"""
AWS-Enhanced Schema Resolver for DataMesh.AI Catalog Agent

Extends the base SchemaResolver to use AWS Glue Catalog
for real table schema resolution.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import trace

try:
    from .glue_catalog import GlueCatalogClient, GlueCatalogConfig, GlueTable
except ImportError:
    from glue_catalog import GlueCatalogClient, GlueCatalogConfig, GlueTable

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("datamesh.connectors.aws_resolver", "0.1.0")


class FieldMetadata:
    """Field metadata compatible with DataMesh.AI Catalog Agent."""

    def __init__(
        self,
        name: str,
        type: str,
        nullable: bool = True,
        description: str | None = None,
        is_partition_key: bool = False,
    ):
        self.name = name
        self.type = type
        self.nullable = nullable
        self.description = description
        self.is_partition_key = is_partition_key

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "nullable": self.nullable,
            "description": self.description,
            "is_partition_key": self.is_partition_key,
        }


class AWSSchemaResolver:
    """
    Schema resolver that uses AWS Glue Catalog as the source of truth.

    Drop-in replacement for the mock SchemaResolver in Catalog Agent,
    providing real schema metadata from your AWS data lake.
    """

    def __init__(
        self,
        region: str | None = None,
        database_filter: str | None = None,
        fallback_catalog: dict[str, list[dict]] | None = None,
    ) -> None:
        """
        Initialize the AWS Schema Resolver.

        Args:
            region: AWS region (defaults to AWS_REGION env var or eu-west-1)
            database_filter: Regex pattern to filter databases
            fallback_catalog: Optional mock catalog for datasets not in Glue
        """
        self._region = region or os.environ.get("AWS_REGION", "eu-west-1")
        self._glue_config = GlueCatalogConfig(
            region=self._region,
            database_filter=database_filter,
        )
        self._glue_client = GlueCatalogClient(self._glue_config)
        self._fallback_catalog = fallback_catalog or {}
        self._table_cache: dict[str, GlueTable] = {}

        logger.info(
            "AWSSchemaResolver initialized",
            extra={
                "region": self._region,
                "has_fallback": bool(fallback_catalog),
            }
        )

    def resolve(self, dataset_uri: str) -> list[dict[str, Any]]:
        """
        Resolve a dataset URI to its field metadata.

        First tries AWS Glue Catalog, falls back to local catalog.

        Args:
            dataset_uri: The catalog URI (e.g., catalog://talki_metrics_dev.session_logs)

        Returns:
            List of field metadata dictionaries

        Raises:
            SchemaResolutionError: If the dataset is not found anywhere
        """
        with tracer.start_as_current_span("aws_resolver.resolve") as span:
            span.set_attribute("dataset_uri", dataset_uri)

            # Try Glue Catalog first
            table = self._glue_client.resolve_dataset_uri(dataset_uri)
            if table:
                self._table_cache[dataset_uri] = table
                schema = table.to_schema()
                span.set_attribute("source", "glue")
                span.set_attribute("field_count", len(schema))
                logger.info(
                    f"Resolved from Glue: {dataset_uri}",
                    extra={"fields": len(schema)}
                )
                return schema

            # Fall back to local catalog
            if dataset_uri in self._fallback_catalog:
                schema = self._fallback_catalog[dataset_uri]
                span.set_attribute("source", "fallback")
                span.set_attribute("field_count", len(schema))
                logger.info(
                    f"Resolved from fallback: {dataset_uri}",
                    extra={"fields": len(schema)}
                )
                return schema

            # Not found anywhere
            span.set_attribute("error", True)
            available = self.list_datasets()
            raise SchemaResolutionError(
                f"Dataset not found: {dataset_uri}. "
                f"Available datasets: {available[:10]}{'...' if len(available) > 10 else ''}",
                dataset_uri=dataset_uri,
            )

    def list_datasets(self) -> list[str]:
        """
        List all available datasets (from Glue + fallback).

        Returns:
            List of dataset URIs
        """
        datasets = set()

        # Add Glue datasets
        for database in self._glue_client.list_databases():
            for table_name in self._glue_client.list_tables(database):
                datasets.add(f"catalog://{database}.{table_name}")

        # Add fallback datasets
        datasets.update(self._fallback_catalog.keys())

        return sorted(datasets)

    def list_databases(self) -> list[str]:
        """List all databases from Glue Catalog."""
        return self._glue_client.list_databases()

    def list_tables(self, database: str) -> list[str]:
        """List all tables in a database."""
        return self._glue_client.list_tables(database)

    def get_table_details(self, dataset_uri: str) -> dict[str, Any] | None:
        """
        Get detailed table information including location and metadata.

        Args:
            dataset_uri: The catalog URI

        Returns:
            Dictionary with full table details, or None if not found
        """
        # Check cache first
        if dataset_uri in self._table_cache:
            table = self._table_cache[dataset_uri]
        else:
            table = self._glue_client.resolve_dataset_uri(dataset_uri)
            if table:
                self._table_cache[dataset_uri] = table

        if not table:
            return None

        return {
            "database": table.database,
            "table_name": table.name,
            "full_name": table.full_name,
            "catalog_uri": table.catalog_uri,
            "location": table.location,
            "description": table.description,
            "table_type": table.table_type,
            "columns": [col.to_field_metadata() for col in table.columns],
            "partition_keys": [pk.to_field_metadata() for pk in table.partition_keys],
            "parameters": table.parameters,
            "created_time": table.created_time,
            "updated_time": table.updated_time,
        }

    def get_lineage(self, dataset_uri: str) -> dict[str, Any]:
        """
        Get lineage information for a dataset.

        Args:
            dataset_uri: The catalog URI

        Returns:
            Lineage information dictionary
        """
        table = self._glue_client.resolve_dataset_uri(dataset_uri)
        if not table:
            return {
                "dataset": dataset_uri,
                "error": "Dataset not found",
            }

        return self._glue_client.get_lineage_hint(table)

    def search(self, search_text: str) -> list[dict[str, Any]]:
        """
        Search for datasets matching a pattern.

        Args:
            search_text: Text to search for

        Returns:
            List of matching dataset summaries
        """
        results = []

        for table in self._glue_client.search_tables(search_text):
            results.append({
                "catalog_uri": table.catalog_uri,
                "database": table.database,
                "table_name": table.name,
                "description": table.description,
                "column_count": len(table.columns) + len(table.partition_keys),
            })

        return results

    def refresh_cache(self) -> None:
        """Clear caches and refresh metadata."""
        self._glue_client.clear_cache()
        self._table_cache.clear()
        logger.info("AWS Schema Resolver cache refreshed")


class SchemaResolutionError(Exception):
    """Exception raised when schema resolution fails."""

    def __init__(self, message: str, dataset_uri: str | None = None) -> None:
        super().__init__(message)
        self.dataset_uri = dataset_uri


# Convenience function for quick setup
def create_talki_resolver(stage: str = "dev") -> AWSSchemaResolver:
    """
    Create a resolver pre-configured for Talki metrics databases.

    Args:
        stage: Environment stage (dev, prod)

    Returns:
        Configured AWSSchemaResolver
    """
    database_pattern = f"talki.*{stage}|talki_metrics_{stage}"

    return AWSSchemaResolver(
        region="eu-west-1",
        database_filter=database_pattern,
        fallback_catalog={
            # Add any virtual/computed datasets here
            f"catalog://talki_metrics_{stage}.overview": [
                {"name": "metric_name", "type": "STRING", "nullable": False, "description": "Metric identifier"},
                {"name": "metric_value", "type": "DOUBLE", "nullable": False, "description": "Metric value"},
                {"name": "timestamp", "type": "TIMESTAMP", "nullable": False, "description": "Measurement time"},
            ]
        }
    )
