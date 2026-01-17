"""
Schema Resolution Logic for Catalog Agent

This module provides the SchemaResolver class for resolving dataset URIs
to field metadata including field names, types, and nullable properties.
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("catalog_agent.resolver", "1.0.0")


class FieldMetadata(TypedDict):
    """Type definition for field metadata."""

    name: str
    type: str
    nullable: bool
    description: str | None


class SchemaResolutionError(Exception):
    """Exception raised when schema resolution fails."""

    def __init__(self, message: str, dataset_uri: str | None = None) -> None:
        super().__init__(message)
        self.dataset_uri = dataset_uri


# Mock catalog data for sample schemas
MOCK_CATALOG: dict[str, list[FieldMetadata]] = {
    "catalog://finance.customer_transactions": [
        {
            "name": "transaction_id",
            "type": "STRING",
            "nullable": False,
            "description": "Unique identifier for the transaction"
        },
        {
            "name": "customer_id",
            "type": "STRING",
            "nullable": False,
            "description": "Reference to the customer who made the transaction"
        },
        {
            "name": "amount",
            "type": "DECIMAL(18,2)",
            "nullable": False,
            "description": "Transaction amount in the account currency"
        },
        {
            "name": "currency",
            "type": "STRING",
            "nullable": False,
            "description": "ISO 4217 currency code"
        },
        {
            "name": "transaction_type",
            "type": "STRING",
            "nullable": False,
            "description": "Type of transaction (debit, credit, transfer)"
        },
        {
            "name": "merchant_name",
            "type": "STRING",
            "nullable": True,
            "description": "Name of the merchant (if applicable)"
        },
        {
            "name": "merchant_category",
            "type": "STRING",
            "nullable": True,
            "description": "MCC category of the merchant"
        },
        {
            "name": "transaction_timestamp",
            "type": "TIMESTAMP",
            "nullable": False,
            "description": "UTC timestamp when the transaction occurred"
        },
        {
            "name": "status",
            "type": "STRING",
            "nullable": False,
            "description": "Transaction status (pending, completed, failed, reversed)"
        },
        {
            "name": "created_at",
            "type": "TIMESTAMP",
            "nullable": False,
            "description": "Record creation timestamp"
        },
        {
            "name": "updated_at",
            "type": "TIMESTAMP",
            "nullable": False,
            "description": "Record last update timestamp"
        }
    ],
    "catalog://finance.monthly_revenue": [
        {
            "name": "month",
            "type": "DATE",
            "nullable": False,
            "description": "First day of the month for aggregation"
        },
        {
            "name": "revenue_type",
            "type": "STRING",
            "nullable": False,
            "description": "Category of revenue (subscription, transaction_fees, interest)"
        },
        {
            "name": "gross_revenue",
            "type": "DECIMAL(18,2)",
            "nullable": False,
            "description": "Total gross revenue for the month"
        },
        {
            "name": "net_revenue",
            "type": "DECIMAL(18,2)",
            "nullable": False,
            "description": "Net revenue after deductions"
        },
        {
            "name": "currency",
            "type": "STRING",
            "nullable": False,
            "description": "ISO 4217 currency code for revenue"
        },
        {
            "name": "customer_segment",
            "type": "STRING",
            "nullable": True,
            "description": "Customer segment (retail, business, enterprise)"
        },
        {
            "name": "region",
            "type": "STRING",
            "nullable": True,
            "description": "Geographic region"
        },
        {
            "name": "transaction_count",
            "type": "INTEGER",
            "nullable": False,
            "description": "Number of transactions contributing to revenue"
        },
        {
            "name": "active_customers",
            "type": "INTEGER",
            "nullable": False,
            "description": "Count of unique active customers"
        },
        {
            "name": "calculated_at",
            "type": "TIMESTAMP",
            "nullable": False,
            "description": "Timestamp when the metrics were calculated"
        }
    ],
    "catalog://analytics.user_churn": [
        {
            "name": "user_id",
            "type": "STRING",
            "nullable": False,
            "description": "Unique identifier for the user"
        },
        {
            "name": "churn_date",
            "type": "DATE",
            "nullable": True,
            "description": "Date when the user churned (null if active)"
        },
        {
            "name": "is_churned",
            "type": "BOOLEAN",
            "nullable": False,
            "description": "Whether the user has churned"
        },
        {
            "name": "churn_probability",
            "type": "FLOAT",
            "nullable": False,
            "description": "ML-predicted probability of churn (0.0 to 1.0)"
        },
        {
            "name": "risk_segment",
            "type": "STRING",
            "nullable": False,
            "description": "Risk segment (low, medium, high, critical)"
        },
        {
            "name": "days_since_last_activity",
            "type": "INTEGER",
            "nullable": False,
            "description": "Number of days since last user activity"
        },
        {
            "name": "lifetime_value",
            "type": "DECIMAL(18,2)",
            "nullable": False,
            "description": "Calculated customer lifetime value"
        },
        {
            "name": "tenure_months",
            "type": "INTEGER",
            "nullable": False,
            "description": "Number of months as a customer"
        },
        {
            "name": "product_count",
            "type": "INTEGER",
            "nullable": False,
            "description": "Number of active products/subscriptions"
        },
        {
            "name": "last_transaction_date",
            "type": "DATE",
            "nullable": True,
            "description": "Date of the last transaction"
        },
        {
            "name": "avg_monthly_transactions",
            "type": "FLOAT",
            "nullable": False,
            "description": "Average number of transactions per month"
        },
        {
            "name": "prediction_date",
            "type": "DATE",
            "nullable": False,
            "description": "Date when the churn prediction was made"
        },
        {
            "name": "model_version",
            "type": "STRING",
            "nullable": False,
            "description": "Version of the ML model used for prediction"
        }
    ]
}


class SchemaResolver:
    """
    Resolves dataset URIs to field metadata.

    The SchemaResolver maintains a catalog of dataset schemas and provides
    resolution capabilities for the catalog.resolve capability.
    """

    # URI pattern for validation
    URI_PATTERN = re.compile(r"^catalog://([a-z_]+)\.([a-z_]+)$")

    def __init__(self, catalog: dict[str, list[FieldMetadata]] | None = None) -> None:
        """
        Initialize the SchemaResolver.

        Args:
            catalog: Optional custom catalog data. If not provided,
                    uses the mock catalog.
        """
        self._catalog = catalog if catalog is not None else MOCK_CATALOG
        logger.info(
            "SchemaResolver initialized",
            extra={"catalog_size": len(self._catalog)}
        )

    def resolve(self, dataset_uri: str) -> list[FieldMetadata]:
        """
        Resolve a dataset URI to its field metadata.

        Args:
            dataset_uri: The catalog URI of the dataset
                        (e.g., catalog://finance.customer_transactions)

        Returns:
            List of field metadata dictionaries.

        Raises:
            SchemaResolutionError: If the URI is invalid or dataset not found.
        """
        with tracer.start_as_current_span("resolve") as span:
            span.set_attribute("dataset_uri", dataset_uri)

            # Validate URI format
            if not self._validate_uri(dataset_uri):
                raise SchemaResolutionError(
                    f"Invalid dataset URI format: {dataset_uri}. "
                    f"Expected format: catalog://domain.dataset_name",
                    dataset_uri=dataset_uri
                )

            # Look up schema in catalog
            schema = self._catalog.get(dataset_uri)
            if schema is None:
                available = list(self._catalog.keys())
                raise SchemaResolutionError(
                    f"Dataset not found: {dataset_uri}. "
                    f"Available datasets: {available}",
                    dataset_uri=dataset_uri
                )

            span.set_attribute("field_count", len(schema))
            logger.debug(
                "Schema resolved",
                extra={
                    "dataset_uri": dataset_uri,
                    "field_count": len(schema)
                }
            )

            return schema

    def _validate_uri(self, uri: str) -> bool:
        """
        Validate the format of a dataset URI.

        Args:
            uri: The URI to validate.

        Returns:
            True if the URI is valid, False otherwise.
        """
        return bool(self.URI_PATTERN.match(uri))

    def list_datasets(self) -> list[str]:
        """
        List all available datasets in the catalog.

        Returns:
            List of dataset URIs.
        """
        return list(self._catalog.keys())

    def get_dataset_domain(self, dataset_uri: str) -> str | None:
        """
        Extract the domain from a dataset URI.

        Args:
            dataset_uri: The catalog URI of the dataset.

        Returns:
            The domain name, or None if the URI is invalid.
        """
        match = self.URI_PATTERN.match(dataset_uri)
        if match:
            return match.group(1)
        return None

    def get_dataset_name(self, dataset_uri: str) -> str | None:
        """
        Extract the dataset name from a dataset URI.

        Args:
            dataset_uri: The catalog URI of the dataset.

        Returns:
            The dataset name, or None if the URI is invalid.
        """
        match = self.URI_PATTERN.match(dataset_uri)
        if match:
            return match.group(2)
        return None

    def get_field_by_name(
        self,
        dataset_uri: str,
        field_name: str
    ) -> FieldMetadata | None:
        """
        Get metadata for a specific field in a dataset.

        Args:
            dataset_uri: The catalog URI of the dataset.
            field_name: The name of the field to retrieve.

        Returns:
            Field metadata if found, None otherwise.

        Raises:
            SchemaResolutionError: If the dataset is not found.
        """
        schema = self.resolve(dataset_uri)
        for field in schema:
            if field["name"] == field_name:
                return field
        return None

    def dataset_exists(self, dataset_uri: str) -> bool:
        """
        Check if a dataset exists in the catalog.

        Args:
            dataset_uri: The catalog URI to check.

        Returns:
            True if the dataset exists, False otherwise.
        """
        return dataset_uri in self._catalog
