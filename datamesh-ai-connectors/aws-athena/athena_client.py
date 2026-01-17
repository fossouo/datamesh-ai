"""
AWS Athena Query Client for DataMesh.AI

Executes SQL queries via AWS Athena with async support,
result streaming, and OpenTelemetry tracing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import boto3
from botocore.exceptions import ClientError
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("datamesh.connectors.athena", "0.1.0")


class QueryState(str, Enum):
    """Athena query execution states."""
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class AthenaQueryResult:
    """Result of an Athena query execution."""
    query_execution_id: str
    state: QueryState
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    bytes_scanned: int = 0
    execution_time_ms: int = 0
    error_message: str | None = None
    s3_output_location: str | None = None


@dataclass
class AthenaConfig:
    """Configuration for Athena client."""
    database: str
    workgroup: str = "primary"
    output_location: str | None = None
    region: str = "eu-west-1"
    max_wait_seconds: int = 300
    poll_interval_seconds: float = 1.0


class AthenaClient:
    """
    AWS Athena client for executing SQL queries.

    Provides synchronous query execution with result polling,
    designed for use in DataMesh.AI SQL Agent.
    """

    def __init__(self, config: AthenaConfig) -> None:
        """
        Initialize the Athena client.

        Args:
            config: Athena configuration with database, workgroup, etc.
        """
        self.config = config
        self._athena = boto3.client("athena", region_name=config.region)
        logger.info(
            "AthenaClient initialized",
            extra={
                "database": config.database,
                "workgroup": config.workgroup,
                "region": config.region,
            }
        )

    def execute_query(
        self,
        query: str,
        max_rows: int = 1000,
        timeout_seconds: int | None = None,
    ) -> AthenaQueryResult:
        """
        Execute a SQL query and wait for results.

        Args:
            query: SQL query string
            max_rows: Maximum rows to return (default 1000)
            timeout_seconds: Query timeout (defaults to config.max_wait_seconds)

        Returns:
            AthenaQueryResult with columns, rows, and metadata

        Raises:
            AthenaQueryError: If query fails or times out
        """
        with tracer.start_as_current_span("athena.execute_query") as span:
            span.set_attribute("db.system", "athena")
            span.set_attribute("db.name", self.config.database)
            span.set_attribute("db.statement", query[:500])  # Truncate for safety

            timeout = timeout_seconds or self.config.max_wait_seconds

            # Start query execution
            execution_id = self._start_query(query)
            span.set_attribute("athena.query_execution_id", execution_id)

            # Wait for completion
            final_state = self._wait_for_completion(execution_id, timeout)

            if final_state == QueryState.SUCCEEDED:
                return self._fetch_results(execution_id, max_rows)
            else:
                error_msg = self._get_error_message(execution_id)
                span.set_attribute("error", True)
                span.set_attribute("error.message", error_msg or "Query failed")
                return AthenaQueryResult(
                    query_execution_id=execution_id,
                    state=final_state,
                    error_message=error_msg,
                )

    def _start_query(self, query: str) -> str:
        """Start an Athena query execution."""
        params: dict[str, Any] = {
            "QueryString": query,
            "QueryExecutionContext": {"Database": self.config.database},
            "WorkGroup": self.config.workgroup,
        }

        if self.config.output_location:
            params["ResultConfiguration"] = {
                "OutputLocation": self.config.output_location
            }

        try:
            response = self._athena.start_query_execution(**params)
            execution_id = response["QueryExecutionId"]
            logger.info(f"Started Athena query: {execution_id}")
            return execution_id
        except ClientError as e:
            logger.error(f"Failed to start query: {e}")
            raise AthenaQueryError(f"Failed to start query: {e}") from e

    def _wait_for_completion(
        self,
        execution_id: str,
        timeout_seconds: int,
    ) -> QueryState:
        """Poll until query completes or times out."""
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                logger.warning(f"Query {execution_id} timed out after {elapsed:.1f}s")
                self._cancel_query(execution_id)
                return QueryState.CANCELLED

            state = self._get_query_state(execution_id)

            if state in (QueryState.SUCCEEDED, QueryState.FAILED, QueryState.CANCELLED):
                logger.info(f"Query {execution_id} completed with state: {state}")
                return state

            time.sleep(self.config.poll_interval_seconds)

    def _get_query_state(self, execution_id: str) -> QueryState:
        """Get current state of a query execution."""
        response = self._athena.get_query_execution(
            QueryExecutionId=execution_id
        )
        state_str = response["QueryExecution"]["Status"]["State"]
        return QueryState(state_str)

    def _get_error_message(self, execution_id: str) -> str | None:
        """Get error message for a failed query."""
        response = self._athena.get_query_execution(
            QueryExecutionId=execution_id
        )
        status = response["QueryExecution"]["Status"]
        return status.get("StateChangeReason")

    def _cancel_query(self, execution_id: str) -> None:
        """Cancel a running query."""
        try:
            self._athena.stop_query_execution(QueryExecutionId=execution_id)
            logger.info(f"Cancelled query {execution_id}")
        except ClientError as e:
            logger.warning(f"Failed to cancel query {execution_id}: {e}")

    def _fetch_results(
        self,
        execution_id: str,
        max_rows: int,
    ) -> AthenaQueryResult:
        """Fetch results from a completed query."""
        # Get execution details for metadata
        exec_response = self._athena.get_query_execution(
            QueryExecutionId=execution_id
        )
        execution = exec_response["QueryExecution"]
        statistics = execution.get("Statistics", {})

        bytes_scanned = statistics.get("DataScannedInBytes", 0)
        execution_time_ms = statistics.get("TotalExecutionTimeInMillis", 0)
        output_location = execution.get("ResultConfiguration", {}).get("OutputLocation")

        # Fetch result rows
        columns: list[str] = []
        rows: list[dict[str, Any]] = []

        paginator = self._athena.get_paginator("get_query_results")

        for page in paginator.paginate(
            QueryExecutionId=execution_id,
            PaginationConfig={"MaxItems": max_rows + 1}  # +1 for header
        ):
            result_set = page["ResultSet"]

            # Extract column names from first page
            if not columns and "ResultSetMetadata" in result_set:
                columns = [
                    col["Name"]
                    for col in result_set["ResultSetMetadata"]["ColumnInfo"]
                ]

            # Extract rows (skip header row on first page)
            for i, row in enumerate(result_set.get("Rows", [])):
                # Skip header row
                if not rows and i == 0:
                    continue

                if len(rows) >= max_rows:
                    break

                row_data = {}
                for j, datum in enumerate(row.get("Data", [])):
                    col_name = columns[j] if j < len(columns) else f"col_{j}"
                    row_data[col_name] = datum.get("VarCharValue")
                rows.append(row_data)

        return AthenaQueryResult(
            query_execution_id=execution_id,
            state=QueryState.SUCCEEDED,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            bytes_scanned=bytes_scanned,
            execution_time_ms=execution_time_ms,
            s3_output_location=output_location,
        )

    def validate_query(self, query: str) -> tuple[bool, str | None]:
        """
        Validate a query without executing it.

        Uses EXPLAIN to check query syntax and semantics.

        Args:
            query: SQL query to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        explain_query = f"EXPLAIN {query}"

        try:
            result = self.execute_query(explain_query, max_rows=10, timeout_seconds=30)
            if result.state == QueryState.SUCCEEDED:
                return True, None
            return False, result.error_message
        except AthenaQueryError as e:
            return False, str(e)

    def get_table_ddl(self, table_name: str) -> str | None:
        """
        Get the DDL (CREATE TABLE statement) for a table.

        Args:
            table_name: Name of the table

        Returns:
            DDL string or None if not found
        """
        query = f"SHOW CREATE TABLE {table_name}"
        result = self.execute_query(query, max_rows=100, timeout_seconds=30)

        if result.state == QueryState.SUCCEEDED and result.rows:
            # DDL is typically in a single column named 'createtab_stmt'
            ddl_parts = [row.get("createtab_stmt", "") for row in result.rows]
            return "\n".join(ddl_parts)
        return None


class AthenaQueryError(Exception):
    """Exception raised for Athena query errors."""
    pass
