"""
Lineage Tracking Logic for Catalog Agent

This module provides the LineageTracker class for tracking upstream and
downstream data lineage relationships between datasets.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("catalog_agent.lineage", "1.0.0")


class LineageNode(TypedDict):
    """Type definition for a lineage relationship node."""

    uri: str
    relationship: str
    transformation: str | None


class LineageResult(TypedDict):
    """Type definition for lineage query results."""

    upstream: list[LineageNode]
    downstream: list[LineageNode]


class LineageTrackingError(Exception):
    """Exception raised when lineage tracking fails."""

    def __init__(self, message: str, dataset_uri: str | None = None) -> None:
        super().__init__(message)
        self.dataset_uri = dataset_uri


# Mock lineage data representing data flow relationships
# Structure: source_uri -> list of downstream nodes
MOCK_LINEAGE_DOWNSTREAM: dict[str, list[LineageNode]] = {
    "catalog://finance.customer_transactions": [
        {
            "uri": "catalog://finance.monthly_revenue",
            "relationship": "aggregation",
            "transformation": "SUM(amount) GROUP BY month, revenue_type"
        },
        {
            "uri": "catalog://analytics.user_churn",
            "relationship": "feature_engineering",
            "transformation": "COUNT(*), AVG(amount) per customer"
        }
    ],
    "catalog://finance.monthly_revenue": [
        {
            "uri": "catalog://analytics.revenue_forecast",
            "relationship": "ml_training",
            "transformation": "Time series forecasting model input"
        },
        {
            "uri": "catalog://reporting.executive_dashboard",
            "relationship": "visualization",
            "transformation": "Monthly aggregates and YoY comparison"
        }
    ],
    "catalog://analytics.user_churn": [
        {
            "uri": "catalog://marketing.retention_campaigns",
            "relationship": "targeting",
            "transformation": "Filter by churn_probability > 0.7"
        },
        {
            "uri": "catalog://reporting.customer_health_dashboard",
            "relationship": "visualization",
            "transformation": "Risk segment distribution"
        }
    ]
}

# Structure: target_uri -> list of upstream nodes
MOCK_LINEAGE_UPSTREAM: dict[str, list[LineageNode]] = {
    "catalog://finance.customer_transactions": [
        {
            "uri": "catalog://raw.payment_events",
            "relationship": "transformation",
            "transformation": "Deduplication, enrichment, validation"
        },
        {
            "uri": "catalog://raw.merchant_data",
            "relationship": "join",
            "transformation": "LEFT JOIN on merchant_id"
        }
    ],
    "catalog://finance.monthly_revenue": [
        {
            "uri": "catalog://finance.customer_transactions",
            "relationship": "aggregation",
            "transformation": "SUM(amount) GROUP BY month, revenue_type"
        },
        {
            "uri": "catalog://finance.subscription_payments",
            "relationship": "union",
            "transformation": "UNION ALL subscription revenue"
        }
    ],
    "catalog://analytics.user_churn": [
        {
            "uri": "catalog://finance.customer_transactions",
            "relationship": "feature_engineering",
            "transformation": "Transaction behavior features"
        },
        {
            "uri": "catalog://analytics.user_activity",
            "relationship": "join",
            "transformation": "LEFT JOIN on user_id for activity metrics"
        },
        {
            "uri": "catalog://ml.churn_model_predictions",
            "relationship": "prediction",
            "transformation": "ML model inference output"
        }
    ]
}


class LineageTracker:
    """
    Tracks upstream and downstream data lineage relationships.

    The LineageTracker maintains a graph of data lineage relationships
    and provides traversal capabilities for the catalog.lineage capability.
    """

    def __init__(
        self,
        upstream_graph: dict[str, list[LineageNode]] | None = None,
        downstream_graph: dict[str, list[LineageNode]] | None = None
    ) -> None:
        """
        Initialize the LineageTracker.

        Args:
            upstream_graph: Optional custom upstream lineage data.
            downstream_graph: Optional custom downstream lineage data.
        """
        self._upstream = upstream_graph if upstream_graph is not None else MOCK_LINEAGE_UPSTREAM
        self._downstream = downstream_graph if downstream_graph is not None else MOCK_LINEAGE_DOWNSTREAM

        logger.info(
            "LineageTracker initialized",
            extra={
                "upstream_datasets": len(self._upstream),
                "downstream_datasets": len(self._downstream)
            }
        )

    def get_lineage(
        self,
        dataset_uri: str,
        depth: int = 3
    ) -> LineageResult:
        """
        Get the full lineage (upstream and downstream) for a dataset.

        Args:
            dataset_uri: The catalog URI of the dataset.
            depth: Maximum depth of lineage traversal (default 3).

        Returns:
            LineageResult containing upstream and downstream relationships.

        Raises:
            LineageTrackingError: If the lineage cannot be tracked.
        """
        with tracer.start_as_current_span("get_lineage") as span:
            span.set_attribute("dataset_uri", dataset_uri)
            span.set_attribute("depth", depth)

            try:
                upstream = self._traverse_upstream(dataset_uri, depth)
                downstream = self._traverse_downstream(dataset_uri, depth)

                span.set_attribute("upstream_count", len(upstream))
                span.set_attribute("downstream_count", len(downstream))

                logger.info(
                    "Lineage traced",
                    extra={
                        "dataset_uri": dataset_uri,
                        "upstream_count": len(upstream),
                        "downstream_count": len(downstream),
                        "depth": depth
                    }
                )

                return LineageResult(
                    upstream=upstream,
                    downstream=downstream
                )

            except Exception as e:
                logger.exception("Error tracing lineage")
                raise LineageTrackingError(
                    f"Failed to trace lineage for {dataset_uri}: {str(e)}",
                    dataset_uri=dataset_uri
                ) from e

    def get_upstream(
        self,
        dataset_uri: str,
        depth: int = 3
    ) -> list[LineageNode]:
        """
        Get upstream lineage for a dataset.

        Args:
            dataset_uri: The catalog URI of the dataset.
            depth: Maximum depth of upstream traversal.

        Returns:
            List of upstream lineage nodes.
        """
        with tracer.start_as_current_span("get_upstream") as span:
            span.set_attribute("dataset_uri", dataset_uri)
            span.set_attribute("depth", depth)

            return self._traverse_upstream(dataset_uri, depth)

    def get_downstream(
        self,
        dataset_uri: str,
        depth: int = 3
    ) -> list[LineageNode]:
        """
        Get downstream lineage for a dataset.

        Args:
            dataset_uri: The catalog URI of the dataset.
            depth: Maximum depth of downstream traversal.

        Returns:
            List of downstream lineage nodes.
        """
        with tracer.start_as_current_span("get_downstream") as span:
            span.set_attribute("dataset_uri", dataset_uri)
            span.set_attribute("depth", depth)

            return self._traverse_downstream(dataset_uri, depth)

    def _traverse_upstream(
        self,
        dataset_uri: str,
        depth: int,
        visited: set[str] | None = None
    ) -> list[LineageNode]:
        """
        Recursively traverse upstream lineage.

        Args:
            dataset_uri: Current dataset URI.
            depth: Remaining depth to traverse.
            visited: Set of already visited URIs (for cycle detection).

        Returns:
            Flattened list of upstream lineage nodes.
        """
        if visited is None:
            visited = set()

        if depth <= 0 or dataset_uri in visited:
            return []

        visited.add(dataset_uri)
        result: list[LineageNode] = []

        # Get direct upstream dependencies
        direct_upstream = self._upstream.get(dataset_uri, [])
        result.extend(direct_upstream)

        # Recursively get upstream of upstream
        for node in direct_upstream:
            indirect = self._traverse_upstream(
                node["uri"],
                depth - 1,
                visited.copy()
            )
            result.extend(indirect)

        return result

    def _traverse_downstream(
        self,
        dataset_uri: str,
        depth: int,
        visited: set[str] | None = None
    ) -> list[LineageNode]:
        """
        Recursively traverse downstream lineage.

        Args:
            dataset_uri: Current dataset URI.
            depth: Remaining depth to traverse.
            visited: Set of already visited URIs (for cycle detection).

        Returns:
            Flattened list of downstream lineage nodes.
        """
        if visited is None:
            visited = set()

        if depth <= 0 or dataset_uri in visited:
            return []

        visited.add(dataset_uri)
        result: list[LineageNode] = []

        # Get direct downstream dependencies
        direct_downstream = self._downstream.get(dataset_uri, [])
        result.extend(direct_downstream)

        # Recursively get downstream of downstream
        for node in direct_downstream:
            indirect = self._traverse_downstream(
                node["uri"],
                depth - 1,
                visited.copy()
            )
            result.extend(indirect)

        return result

    def has_lineage(self, dataset_uri: str) -> bool:
        """
        Check if a dataset has any lineage information.

        Args:
            dataset_uri: The catalog URI to check.

        Returns:
            True if the dataset has upstream or downstream lineage.
        """
        return (
            dataset_uri in self._upstream or
            dataset_uri in self._downstream
        )

    def get_direct_upstream(self, dataset_uri: str) -> list[LineageNode]:
        """
        Get only direct (depth=1) upstream dependencies.

        Args:
            dataset_uri: The catalog URI of the dataset.

        Returns:
            List of direct upstream lineage nodes.
        """
        return self._upstream.get(dataset_uri, [])

    def get_direct_downstream(self, dataset_uri: str) -> list[LineageNode]:
        """
        Get only direct (depth=1) downstream dependencies.

        Args:
            dataset_uri: The catalog URI of the dataset.

        Returns:
            List of direct downstream lineage nodes.
        """
        return self._downstream.get(dataset_uri, [])

    def find_path(
        self,
        source_uri: str,
        target_uri: str,
        max_depth: int = 10
    ) -> list[str] | None:
        """
        Find a lineage path between two datasets.

        Args:
            source_uri: Starting dataset URI.
            target_uri: Target dataset URI.
            max_depth: Maximum path length to search.

        Returns:
            List of URIs representing the path, or None if no path exists.
        """
        with tracer.start_as_current_span("find_path") as span:
            span.set_attribute("source_uri", source_uri)
            span.set_attribute("target_uri", target_uri)
            span.set_attribute("max_depth", max_depth)

            # BFS to find shortest path
            from collections import deque

            queue: deque[tuple[str, list[str]]] = deque([(source_uri, [source_uri])])
            visited: set[str] = {source_uri}

            while queue:
                current, path = queue.popleft()

                if len(path) > max_depth:
                    continue

                if current == target_uri:
                    span.set_attribute("path_length", len(path))
                    return path

                # Check downstream connections
                for node in self._downstream.get(current, []):
                    if node["uri"] not in visited:
                        visited.add(node["uri"])
                        queue.append((node["uri"], path + [node["uri"]]))

            return None

    def get_impact_analysis(
        self,
        dataset_uri: str,
        depth: int = 3
    ) -> dict[str, list[LineageNode]]:
        """
        Get impact analysis showing all datasets affected by changes.

        Args:
            dataset_uri: The dataset to analyze.
            depth: Maximum depth for impact analysis.

        Returns:
            Dictionary with downstream datasets grouped by depth level.
        """
        with tracer.start_as_current_span("impact_analysis") as span:
            span.set_attribute("dataset_uri", dataset_uri)
            span.set_attribute("depth", depth)

            result: dict[str, list[LineageNode]] = {}
            visited: set[str] = {dataset_uri}

            current_level = [dataset_uri]
            for level in range(1, depth + 1):
                next_level: list[LineageNode] = []

                for uri in current_level:
                    for node in self._downstream.get(uri, []):
                        if node["uri"] not in visited:
                            visited.add(node["uri"])
                            next_level.append(node)

                if next_level:
                    result[f"level_{level}"] = next_level
                    current_level = [n["uri"] for n in next_level]
                else:
                    break

            return result
