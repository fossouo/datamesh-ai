"""
AWS Glue Catalog Client for DataMesh.AI

Provides metadata discovery from AWS Glue Data Catalog,
enabling the Catalog Agent to resolve real table schemas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import ClientError
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("datamesh.connectors.glue", "0.1.0")


@dataclass
class GlueColumn:
    """Represents a column in a Glue table."""
    name: str
    type: str
    comment: str | None = None
    is_partition_key: bool = False

    def to_field_metadata(self) -> dict[str, Any]:
        """Convert to DataMesh.AI FieldMetadata format."""
        return {
            "name": self.name,
            "type": self._normalize_type(self.type),
            "nullable": True,  # Glue doesn't track nullability
            "description": self.comment,
            "is_partition_key": self.is_partition_key,
        }

    def _normalize_type(self, glue_type: str) -> str:
        """Normalize Glue types to standard SQL types."""
        type_map = {
            "string": "STRING",
            "int": "INTEGER",
            "bigint": "BIGINT",
            "double": "DOUBLE",
            "float": "FLOAT",
            "boolean": "BOOLEAN",
            "timestamp": "TIMESTAMP",
            "date": "DATE",
            "decimal": "DECIMAL",
            "binary": "BINARY",
            "array": "ARRAY",
            "map": "MAP",
            "struct": "STRUCT",
        }
        base_type = glue_type.lower().split("(")[0].split("<")[0]
        return type_map.get(base_type, glue_type.upper())


@dataclass
class GlueTable:
    """Represents a table from Glue Data Catalog."""
    database: str
    name: str
    columns: list[GlueColumn] = field(default_factory=list)
    partition_keys: list[GlueColumn] = field(default_factory=list)
    location: str | None = None
    description: str | None = None
    table_type: str | None = None
    parameters: dict[str, str] = field(default_factory=dict)
    created_time: str | None = None
    updated_time: str | None = None

    @property
    def full_name(self) -> str:
        """Get fully qualified table name."""
        return f"{self.database}.{self.name}"

    @property
    def catalog_uri(self) -> str:
        """Get DataMesh.AI catalog URI format."""
        return f"catalog://{self.database}.{self.name}"

    def all_columns(self) -> list[GlueColumn]:
        """Get all columns including partition keys."""
        return self.columns + self.partition_keys

    def to_schema(self) -> list[dict[str, Any]]:
        """Convert to DataMesh.AI schema format."""
        return [col.to_field_metadata() for col in self.all_columns()]


@dataclass
class GlueCatalogConfig:
    """Configuration for Glue Catalog client."""
    region: str = "eu-west-1"
    database_filter: str | None = None  # Regex pattern to filter databases


class GlueCatalogClient:
    """
    AWS Glue Data Catalog client for schema discovery.

    Retrieves table metadata from Glue Catalog to provide
    real schema information to the DataMesh.AI Catalog Agent.
    """

    def __init__(self, config: GlueCatalogConfig | None = None) -> None:
        """
        Initialize the Glue Catalog client.

        Args:
            config: Optional configuration (defaults to eu-west-1)
        """
        self.config = config or GlueCatalogConfig()
        self._glue = boto3.client("glue", region_name=self.config.region)
        self._cache: dict[str, GlueTable] = {}
        logger.info(
            "GlueCatalogClient initialized",
            extra={"region": self.config.region}
        )

    def list_databases(self) -> list[str]:
        """
        List all databases in the Glue Catalog.

        Returns:
            List of database names
        """
        with tracer.start_as_current_span("glue.list_databases") as span:
            databases = []
            paginator = self._glue.get_paginator("get_databases")

            for page in paginator.paginate():
                for db in page.get("DatabaseList", []):
                    db_name = db["Name"]
                    # Apply filter if configured
                    if self.config.database_filter:
                        import re
                        if not re.match(self.config.database_filter, db_name):
                            continue
                    databases.append(db_name)

            span.set_attribute("database_count", len(databases))
            logger.info(f"Found {len(databases)} databases")
            return databases

    def list_tables(self, database: str) -> list[str]:
        """
        List all tables in a database.

        Args:
            database: Database name

        Returns:
            List of table names
        """
        with tracer.start_as_current_span("glue.list_tables") as span:
            span.set_attribute("database", database)
            tables = []

            try:
                paginator = self._glue.get_paginator("get_tables")
                for page in paginator.paginate(DatabaseName=database):
                    for table in page.get("TableList", []):
                        tables.append(table["Name"])

                span.set_attribute("table_count", len(tables))
                logger.info(f"Found {len(tables)} tables in {database}")
                return tables

            except ClientError as e:
                if e.response["Error"]["Code"] == "EntityNotFoundException":
                    logger.warning(f"Database not found: {database}")
                    return []
                raise

    def get_table(self, database: str, table_name: str) -> GlueTable | None:
        """
        Get detailed metadata for a table.

        Args:
            database: Database name
            table_name: Table name

        Returns:
            GlueTable with full schema, or None if not found
        """
        cache_key = f"{database}.{table_name}"

        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        with tracer.start_as_current_span("glue.get_table") as span:
            span.set_attribute("database", database)
            span.set_attribute("table", table_name)

            try:
                response = self._glue.get_table(
                    DatabaseName=database,
                    Name=table_name,
                )
                table_data = response["Table"]

                # Parse columns
                columns = []
                for col in table_data.get("StorageDescriptor", {}).get("Columns", []):
                    columns.append(GlueColumn(
                        name=col["Name"],
                        type=col["Type"],
                        comment=col.get("Comment"),
                        is_partition_key=False,
                    ))

                # Parse partition keys
                partition_keys = []
                for pk in table_data.get("PartitionKeys", []):
                    partition_keys.append(GlueColumn(
                        name=pk["Name"],
                        type=pk["Type"],
                        comment=pk.get("Comment"),
                        is_partition_key=True,
                    ))

                glue_table = GlueTable(
                    database=database,
                    name=table_name,
                    columns=columns,
                    partition_keys=partition_keys,
                    location=table_data.get("StorageDescriptor", {}).get("Location"),
                    description=table_data.get("Description"),
                    table_type=table_data.get("TableType"),
                    parameters=table_data.get("Parameters", {}),
                    created_time=str(table_data.get("CreateTime", "")),
                    updated_time=str(table_data.get("UpdateTime", "")),
                )

                # Cache the result
                self._cache[cache_key] = glue_table

                span.set_attribute("column_count", len(columns))
                span.set_attribute("partition_key_count", len(partition_keys))
                logger.info(
                    f"Retrieved table {cache_key}",
                    extra={
                        "columns": len(columns),
                        "partition_keys": len(partition_keys),
                    }
                )
                return glue_table

            except ClientError as e:
                if e.response["Error"]["Code"] == "EntityNotFoundException":
                    logger.warning(f"Table not found: {cache_key}")
                    return None
                raise

    def resolve_dataset_uri(self, uri: str) -> GlueTable | None:
        """
        Resolve a DataMesh.AI catalog URI to Glue table metadata.

        Args:
            uri: Catalog URI (e.g., catalog://talki_metrics_dev.session_logs)

        Returns:
            GlueTable or None if not found
        """
        # Parse URI: catalog://database.table
        if not uri.startswith("catalog://"):
            logger.warning(f"Invalid catalog URI format: {uri}")
            return None

        path = uri[len("catalog://"):]
        parts = path.split(".", 1)
        if len(parts) != 2:
            logger.warning(f"Invalid catalog URI path: {path}")
            return None

        database, table_name = parts
        return self.get_table(database, table_name)

    def search_tables(self, search_text: str) -> list[GlueTable]:
        """
        Search for tables matching a text pattern.

        Args:
            search_text: Text to search for in table names

        Returns:
            List of matching GlueTable objects
        """
        with tracer.start_as_current_span("glue.search_tables") as span:
            span.set_attribute("search_text", search_text)
            results = []
            search_lower = search_text.lower()

            for database in self.list_databases():
                for table_name in self.list_tables(database):
                    if search_lower in table_name.lower():
                        table = self.get_table(database, table_name)
                        if table:
                            results.append(table)

            span.set_attribute("result_count", len(results))
            return results

    def get_lineage_hint(self, table: GlueTable) -> dict[str, Any]:
        """
        Extract lineage hints from table parameters.

        Glue doesn't have native lineage, but we can extract
        hints from S3 locations and table parameters.

        Args:
            table: GlueTable to analyze

        Returns:
            Lineage hint dictionary
        """
        lineage = {
            "table": table.full_name,
            "location": table.location,
            "upstream": [],
            "downstream": [],
        }

        # Check for lineage in table parameters
        params = table.parameters
        if "source_tables" in params:
            lineage["upstream"] = params["source_tables"].split(",")
        if "downstream_tables" in params:
            lineage["downstream"] = params["downstream_tables"].split(",")

        # Infer from S3 location patterns
        if table.location:
            if "raw" in table.location.lower():
                lineage["tier"] = "raw"
            elif "processed" in table.location.lower():
                lineage["tier"] = "processed"
            elif "curated" in table.location.lower():
                lineage["tier"] = "curated"

        return lineage

    def build_catalog(self, databases: list[str] | None = None) -> dict[str, list[dict]]:
        """
        Build a complete catalog of all tables.

        Args:
            databases: Specific databases to include (None = all)

        Returns:
            Dictionary mapping catalog URIs to schema metadata
        """
        catalog: dict[str, list[dict]] = {}
        db_list = databases or self.list_databases()

        for database in db_list:
            for table_name in self.list_tables(database):
                table = self.get_table(database, table_name)
                if table:
                    catalog[table.catalog_uri] = table.to_schema()

        logger.info(f"Built catalog with {len(catalog)} tables")
        return catalog

    def clear_cache(self) -> None:
        """Clear the internal table cache."""
        self._cache.clear()
        logger.info("Glue catalog cache cleared")
