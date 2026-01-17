"""
AWS-Enhanced SQL Agent for DataMesh.AI

This agent uses AWS Athena for actual SQL query execution
against Talki metrics data in S3.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

# Add connectors to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../datamesh-ai-connectors/aws-athena"))

from athena_client import AthenaClient, AthenaConfig, AthenaQueryResult, QueryState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sql-agent-aws")


@dataclass
class AgentConfig:
    """Configuration for the AWS SQL Agent."""
    agent_id: str = "sql-agent-aws-001"
    host: str = "0.0.0.0"
    port: int = 8081
    athena_database: str = "talki_metrics_prod"
    athena_workgroup: str = "primary"
    athena_output: str = "s3://talki-athena-results-eu-west-1/datamesh-ai/"
    aws_region: str = "eu-west-1"
    max_rows: int = 1000


class SQLAgentAWS:
    """
    SQL Agent with AWS Athena integration.

    Capabilities:
    - sql.generate: Generate SQL from natural language (placeholder)
    - sql.validate: Validate SQL syntax via EXPLAIN
    - sql.execute: Execute SQL and return results
    - sql.optimize: Suggest query optimizations
    """

    # Allowed operations (read-only for safety)
    ALLOWED_OPERATIONS = {"SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN"}
    BLOCKED_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"}

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        athena_config = AthenaConfig(
            database=config.athena_database,
            workgroup=config.athena_workgroup,
            output_location=config.athena_output,
            region=config.aws_region,
        )
        self.athena = AthenaClient(athena_config)
        logger.info(
            f"Initialized {config.agent_id}",
            extra={
                "database": config.athena_database,
                "region": config.aws_region,
            },
        )

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Route request to appropriate capability handler."""
        capability = request.get("capability", "")
        payload = request.get("payload", {})
        request_id = request.get("requestId", str(uuid.uuid4()))

        logger.info(f"[{request_id}] Handling capability: {capability}")

        try:
            if capability == "sql.generate":
                result = self._handle_generate(payload, request)
            elif capability == "sql.validate":
                result = self._handle_validate(payload)
            elif capability == "sql.execute":
                result = self._handle_execute(payload)
            elif capability == "sql.optimize":
                result = self._handle_optimize(payload)
            else:
                return self._error_response(
                    request_id, f"Unknown capability: {capability}"
                )

            return self._success_response(request_id, capability, result)

        except Exception as e:
            logger.error(f"[{request_id}] Error: {e}", exc_info=True)
            return self._error_response(request_id, str(e))

    def _handle_generate(
        self, payload: dict[str, Any], request: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate SQL from natural language question.

        This is a template-based approach for common Talki queries.
        In production, this would use an LLM for NL-to-SQL.
        """
        question = payload.get("question", "")
        question_lower = question.lower()

        # Template matching for common Talki queries
        if "session" in question_lower and ("count" in question_lower or "how many" in question_lower):
            sql = self._generate_session_count_query(question)
        elif "cost" in question_lower:
            sql = self._generate_cost_query(question)
        elif "language" in question_lower:
            sql = self._generate_language_query(question)
        elif "latency" in question_lower or "performance" in question_lower:
            sql = self._generate_latency_query(question)
        elif "daily" in question_lower or "trend" in question_lower:
            sql = self._generate_daily_trend_query(question)
        else:
            # Default: session overview
            sql = """SELECT
    language,
    region,
    COUNT(*) as sessions,
    COUNT(DISTINCT family_id_hash) as families,
    ROUND(SUM(cost_usd), 4) as total_cost,
    ROUND(AVG(latency_ms), 2) as avg_latency_ms
FROM session_logs
GROUP BY language, region
ORDER BY sessions DESC
LIMIT 20"""

        return {
            "question": question,
            "sql": sql,
            "explanation": f"Generated SQL for: '{question}'",
            "confidence": 0.85,
            "source": "template_matching",
        }

    def _generate_session_count_query(self, question: str) -> str:
        """Generate session count query."""
        return """SELECT
    COUNT(*) as total_sessions,
    COUNT(DISTINCT family_id_hash) as unique_families,
    COUNT(DISTINCT child_id_hash) as unique_children,
    ROUND(SUM(cost_usd), 4) as total_cost_usd
FROM session_logs"""

    def _generate_cost_query(self, question: str) -> str:
        """Generate cost analysis query."""
        return """SELECT
    model_name,
    COUNT(*) as sessions,
    ROUND(SUM(cost_usd), 4) as total_cost,
    ROUND(AVG(cost_usd), 6) as avg_cost,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens
FROM session_logs
GROUP BY model_name
ORDER BY total_cost DESC"""

    def _generate_language_query(self, question: str) -> str:
        """Generate language breakdown query."""
        return """SELECT
    language,
    region,
    COUNT(*) as sessions,
    COUNT(DISTINCT family_id_hash) as families,
    ROUND(SUM(cost_usd), 4) as total_cost,
    ROUND(AVG(cost_usd), 6) as avg_cost_per_session
FROM session_logs
GROUP BY language, region
ORDER BY sessions DESC"""

    def _generate_latency_query(self, question: str) -> str:
        """Generate latency/performance query."""
        return """SELECT
    model_name,
    language,
    COUNT(*) as sessions,
    ROUND(AVG(latency_ms), 2) as avg_latency_ms,
    MIN(latency_ms) as min_latency_ms,
    MAX(latency_ms) as max_latency_ms,
    ROUND(AVG(duration_seconds), 2) as avg_duration_sec
FROM session_logs
WHERE latency_ms > 0
GROUP BY model_name, language
ORDER BY avg_latency_ms DESC"""

    def _generate_daily_trend_query(self, question: str) -> str:
        """Generate daily trend query."""
        return """SELECT
    year, month, day,
    COUNT(*) as sessions,
    COUNT(DISTINCT family_id_hash) as active_families,
    ROUND(SUM(cost_usd), 4) as daily_cost,
    ROUND(AVG(latency_ms), 2) as avg_latency_ms
FROM session_logs
GROUP BY year, month, day
ORDER BY year DESC, month DESC, day DESC
LIMIT 30"""

    def _handle_validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate SQL syntax without executing."""
        sql = payload.get("sql", "")
        if not sql:
            raise ValueError("Missing 'sql' in payload")

        # Security check first
        security_check = self._security_check(sql)
        if not security_check["safe"]:
            return {
                "valid": False,
                "sql": sql,
                "error": security_check["reason"],
                "security_violation": True,
            }

        # Use EXPLAIN to validate
        is_valid, error = self.athena.validate_query(sql)

        return {
            "valid": is_valid,
            "sql": sql,
            "error": error,
        }

    def _handle_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Execute SQL and return results with pagination support.

        Pagination parameters:
        - page_size: Number of rows per page (default: 100, max: 1000)
        - page: Page number (1-indexed, default: 1)
        - cursor: Opaque cursor for cursor-based pagination
        """
        sql = payload.get("sql", "")
        max_rows = payload.get("max_rows", self.config.max_rows)

        # Pagination parameters
        page_size = min(payload.get("page_size", 100), 1000)
        page = max(payload.get("page", 1), 1)
        cursor = payload.get("cursor")  # For cursor-based pagination

        if not sql:
            raise ValueError("Missing 'sql' in payload")

        # Security check
        security_check = self._security_check(sql)
        if not security_check["safe"]:
            raise ValueError(f"Query blocked: {security_check['reason']}")

        # For paginated queries, we need to fetch enough rows
        # Calculate offset for offset-based pagination
        offset = (page - 1) * page_size
        fetch_limit = offset + page_size + 1  # +1 to detect if there are more rows

        # Execute via Athena with enough rows for pagination
        result = self.athena.execute_query(sql, max_rows=min(fetch_limit, max_rows))

        if result.state == QueryState.SUCCEEDED:
            all_rows = result.rows or []
            total_available = len(all_rows)

            # Apply pagination
            if offset < len(all_rows):
                page_rows = all_rows[offset:offset + page_size]
            else:
                page_rows = []

            has_more = total_available > offset + page_size

            # Generate cursor for next page
            next_cursor = None
            if has_more:
                next_cursor = f"page:{page + 1}:size:{page_size}"

            return {
                "status": "success",
                "sql": sql,
                "columns": result.columns,
                "rows": page_rows,
                "row_count": len(page_rows),
                "total_rows_fetched": total_available,
                "bytes_scanned": result.bytes_scanned,
                "execution_time_ms": result.execution_time_ms,
                "query_execution_id": result.query_execution_id,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "has_more": has_more,
                    "next_cursor": next_cursor,
                    "total_pages": (total_available + page_size - 1) // page_size if total_available > 0 else 0,
                },
            }
        else:
            return {
                "status": "failed",
                "sql": sql,
                "error": result.error_message,
                "query_execution_id": result.query_execution_id,
            }

    def _handle_optimize(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Suggest query optimizations."""
        sql = payload.get("sql", "")
        if not sql:
            raise ValueError("Missing 'sql' in payload")

        suggestions = []

        # Check for partition usage
        if "session_logs" in sql.lower():
            if not any(p in sql.lower() for p in ["year", "month", "day"]):
                suggestions.append({
                    "type": "partition_pruning",
                    "severity": "high",
                    "message": "Add partition filters (year, month, day) to reduce data scanned",
                    "example": "WHERE year = '2025' AND month = '01'",
                })

        # Check for SELECT *
        if "select *" in sql.lower():
            suggestions.append({
                "type": "column_projection",
                "severity": "medium",
                "message": "Specify columns instead of SELECT * to reduce data transfer",
            })

        # Check for missing LIMIT
        if "limit" not in sql.lower() and "group by" not in sql.lower():
            suggestions.append({
                "type": "result_limit",
                "severity": "low",
                "message": "Consider adding LIMIT to prevent large result sets",
            })

        return {
            "sql": sql,
            "suggestions": suggestions,
            "suggestion_count": len(suggestions),
        }

    def _security_check(self, sql: str) -> dict[str, Any]:
        """
        Check SQL for dangerous operations.

        Security layers:
        1. Block dangerous keywords (DELETE, DROP, etc.)
        2. Verify first statement is a read operation
        3. Detect SQL injection patterns
        4. Check for suspicious comment patterns
        """
        # Normalize for checking
        sql_upper = sql.upper()

        # Layer 1: Block dangerous keywords anywhere in query
        for keyword in self.BLOCKED_KEYWORDS:
            # Use word boundary to avoid false positives (e.g., "UPDATED_AT")
            if re.search(rf"\b{keyword}\b", sql_upper):
                return {
                    "safe": False,
                    "reason": f"Blocked operation: {keyword}",
                    "security_rule": "blocked_keyword",
                }

        # Layer 2: Detect SQL injection patterns
        injection_patterns = [
            (r";\s*\b(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER|TRUNCATE)\b", "Multi-statement injection"),
            (r"'\s*OR\s+'\d+'\s*=\s*'\d+", "Classic OR injection"),
            (r"'\s*OR\s+1\s*=\s*1", "Tautology injection"),
            (r"UNION\s+ALL\s+SELECT", "UNION injection"),
            (r"--\s*$", "Comment-based injection (end of line)"),
            (r"/\*.*\*/\s*\bDROP\b", "Comment obfuscation"),
            (r"CHAR\s*\(\s*\d+\s*\)", "Character encoding bypass"),
            (r"0x[0-9a-fA-F]+", "Hex encoding bypass"),
        ]

        for pattern, description in injection_patterns:
            if re.search(pattern, sql_upper, re.IGNORECASE):
                return {
                    "safe": False,
                    "reason": f"Potential SQL injection: {description}",
                    "security_rule": "injection_pattern",
                }

        # Layer 3: Remove comments and check first statement
        clean_sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
        clean_sql = re.sub(r"/\*.*?\*/", "", clean_sql, flags=re.DOTALL)
        clean_sql = clean_sql.strip()

        if not clean_sql:
            return {
                "safe": False,
                "reason": "Empty query after removing comments",
                "security_rule": "empty_query",
            }

        first_word = clean_sql.split()[0].upper()
        if first_word not in self.ALLOWED_OPERATIONS:
            return {
                "safe": False,
                "reason": f"Only read operations allowed. Found: {first_word}",
                "security_rule": "disallowed_operation",
            }

        # Layer 4: Check for multiple statements (semicolon)
        # Allow semicolon only at the end
        statements = [s.strip() for s in clean_sql.split(";") if s.strip()]
        if len(statements) > 1:
            return {
                "safe": False,
                "reason": "Multiple statements not allowed",
                "security_rule": "multi_statement",
            }

        return {"safe": True, "reason": None, "security_rule": None}

    def _success_response(
        self, request_id: str, capability: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "status": "SUCCESS",
            "requestId": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": self.config.agent_id,
            "capability": capability,
            "data": data,
        }

    def _error_response(self, request_id: str, error: str) -> dict[str, Any]:
        return {
            "status": "ERROR",
            "requestId": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": self.config.agent_id,
            "error": error,
        }


class SQLAgentHandler(BaseHTTPRequestHandler):
    """HTTP handler for SQL Agent requests."""

    agent: SQLAgentAWS = None

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            request = json.loads(body)
            response = self.agent.handle_request(request)
            self._send_json(200, response)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "healthy",
                "agent": self.agent.config.agent_id,
                "database": self.agent.config.athena_database,
                "capabilities": [
                    "sql.generate",
                    "sql.validate",
                    "sql.execute",
                    "sql.optimize",
                ],
            })
        else:
            self._send_json(404, {"error": "Not found"})

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        logger.debug(f"{self.address_string()} - {format % args}")


def main():
    """Run the AWS SQL Agent."""
    config = AgentConfig(
        athena_database=os.environ.get("ATHENA_DATABASE", "talki_metrics_dev"),
        athena_workgroup=os.environ.get("ATHENA_WORKGROUP", "primary"),
        athena_output=os.environ.get(
            "ATHENA_OUTPUT", "s3://talki-athena-results-eu-west-1/datamesh-ai/"
        ),
        aws_region=os.environ.get("AWS_REGION", "eu-west-1"),
        port=int(os.environ.get("SQL_AGENT_PORT", "8081")),
    )

    agent = SQLAgentAWS(config)
    SQLAgentHandler.agent = agent

    server = HTTPServer((config.host, config.port), SQLAgentHandler)
    logger.info(f"Starting {config.agent_id} on {config.host}:{config.port}")
    logger.info(f"Athena Database: {config.athena_database}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
