"""
Unified Discovery - Combined discovery for catalogs, connectors, and data sources.

Provides a single entry point for comprehensive environment discovery.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .catalogs import (
    CatalogDiscovery,
    CatalogType,
    DiscoveredCatalog,
    DiscoveredDataset,
)
from .connectors import (
    ConnectorDiscovery,
    ConnectorType,
    ConnectorCapability,
    DiscoveredConnector,
)
from .sources import (
    DataSourceDiscovery,
    DataSourceType,
    DiscoveredDataSource,
)

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryReport:
    """Complete discovery report."""
    # Summary
    total_catalogs: int = 0
    total_schemas: int = 0
    total_datasets: int = 0
    total_connectors: int = 0
    available_connectors: int = 0
    ready_connectors: int = 0
    total_data_sources: int = 0
    configured_data_sources: int = 0

    # Details
    catalogs: list[DiscoveredCatalog] = field(default_factory=list)
    connectors: list[DiscoveredConnector] = field(default_factory=list)
    data_sources: list[DiscoveredDataSource] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    missing_dependencies: dict[str, list[str]] = field(default_factory=dict)

    # Metadata
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    discovery_duration_ms: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "summary": {
                "total_catalogs": self.total_catalogs,
                "total_schemas": self.total_schemas,
                "total_datasets": self.total_datasets,
                "total_connectors": self.total_connectors,
                "available_connectors": self.available_connectors,
                "ready_connectors": self.ready_connectors,
                "total_data_sources": self.total_data_sources,
                "configured_data_sources": self.configured_data_sources,
            },
            "catalogs": [
                {
                    "name": c.name,
                    "type": c.catalog_type.value,
                    "provider": c.provider,
                    "schemas": c.total_schemas,
                    "datasets": c.total_datasets,
                }
                for c in self.catalogs
            ],
            "connectors": [
                {
                    "name": c.name,
                    "type": c.connector_type.value,
                    "available": c.available,
                    "ready": c.ready,
                    "capabilities": [cap.value for cap in c.capabilities],
                }
                for c in self.connectors
            ],
            "data_sources": [
                {
                    "name": s.name,
                    "type": s.source_type.value,
                    "provider": s.provider,
                    "host": s.host,
                    "database": s.database,
                    "has_credentials": s.has_credentials,
                }
                for s in self.data_sources
            ],
            "recommendations": self.recommendations,
            "missing_dependencies": self.missing_dependencies,
            "discovered_at": self.discovered_at.isoformat(),
            "discovery_duration_ms": self.discovery_duration_ms,
            "errors": self.errors,
        }

    def print_summary(self) -> str:
        """Generate a printable summary."""
        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("  DATAMESH.AI - DISCOVERY REPORT")
        lines.append(f"  {self.discovered_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)

        # Catalogs
        lines.append("")
        lines.append("  DATA CATALOGS")
        lines.append("  " + "-" * 40)
        if self.catalogs:
            for catalog in self.catalogs:
                lines.append(f"    {catalog.name}")
                lines.append(f"      Type: {catalog.catalog_type.value}")
                lines.append(f"      Schemas: {catalog.total_schemas}")
                lines.append(f"      Datasets: {catalog.total_datasets}")
        else:
            lines.append("    No catalogs discovered")

        # Connectors
        lines.append("")
        lines.append("  CONNECTORS")
        lines.append("  " + "-" * 40)
        lines.append(f"    Total: {self.total_connectors}")
        lines.append(f"    Available: {self.available_connectors}")
        lines.append(f"    Ready to use: {self.ready_connectors}")
        lines.append("")

        ready = [c for c in self.connectors if c.ready]
        if ready:
            lines.append("    Ready connectors:")
            for c in ready[:5]:
                lines.append(f"      - {c.name} ({c.connector_type.value})")

        available_not_ready = [c for c in self.connectors if c.available and not c.ready]
        if available_not_ready:
            lines.append("")
            lines.append("    Installed but not configured:")
            for c in available_not_ready[:5]:
                lines.append(f"      - {c.name}")

        # Data Sources
        lines.append("")
        lines.append("  DATA SOURCES")
        lines.append("  " + "-" * 40)
        lines.append(f"    Total: {self.total_data_sources}")
        lines.append(f"    Configured: {self.configured_data_sources}")
        lines.append("")

        if self.data_sources:
            for source in self.data_sources[:5]:
                status = "configured" if source.has_credentials else "needs config"
                lines.append(f"    - {source.display_name} ({source.source_type.value}) [{status}]")

        # Recommendations
        if self.recommendations:
            lines.append("")
            lines.append("  RECOMMENDATIONS")
            lines.append("  " + "-" * 40)
            for rec in self.recommendations[:5]:
                lines.append(f"    - {rec}")

        # Missing Dependencies
        if self.missing_dependencies:
            lines.append("")
            lines.append("  MISSING DEPENDENCIES")
            lines.append("  " + "-" * 40)
            for connector, deps in list(self.missing_dependencies.items())[:3]:
                lines.append(f"    {connector}: pip install {' '.join(deps)}")

        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  Discovery completed in {self.discovery_duration_ms}ms")
        lines.append("=" * 70)
        lines.append("")

        return "\n".join(lines)


class UnifiedDiscovery:
    """
    Unified discovery service.

    Combines catalog, connector, and data source discovery into
    a single comprehensive scan.
    """

    def __init__(self):
        self._catalog_discovery = CatalogDiscovery()
        self._connector_discovery = ConnectorDiscovery()
        self._source_discovery = DataSourceDiscovery()
        self._report: Optional[DiscoveryReport] = None

    async def discover(
        self,
        catalogs: bool = True,
        connectors: bool = True,
        data_sources: bool = True,
        catalog_configs: Optional[dict[CatalogType, dict]] = None,
        verbose: bool = False,
    ) -> DiscoveryReport:
        """
        Run comprehensive discovery.

        Args:
            catalogs: Discover data catalogs
            connectors: Discover available connectors
            data_sources: Discover data sources
            catalog_configs: Custom catalog configurations
            verbose: Enable verbose logging

        Returns:
            Complete discovery report
        """
        import time

        start_time = time.time()
        report = DiscoveryReport()

        # Discover connectors first (helps with catalog discovery)
        if connectors:
            try:
                if verbose:
                    logger.info("Discovering connectors...")
                discovered_connectors = await self._connector_discovery.discover_all()
                report.connectors = discovered_connectors
                report.total_connectors = len(discovered_connectors)
                report.available_connectors = sum(1 for c in discovered_connectors if c.available)
                report.ready_connectors = sum(1 for c in discovered_connectors if c.ready)
                report.missing_dependencies = self._connector_discovery.get_missing_dependencies()
            except Exception as e:
                report.errors.append(f"Connector discovery error: {str(e)}")
                logger.error(f"Connector discovery failed: {e}")

        # Discover data sources
        if data_sources:
            try:
                if verbose:
                    logger.info("Discovering data sources...")
                discovered_sources = await self._source_discovery.discover_all()
                report.data_sources = discovered_sources
                report.total_data_sources = len(discovered_sources)
                report.configured_data_sources = sum(1 for s in discovered_sources if s.has_credentials)
            except Exception as e:
                report.errors.append(f"Data source discovery error: {str(e)}")
                logger.error(f"Data source discovery failed: {e}")

        # Discover catalogs
        if catalogs:
            try:
                if verbose:
                    logger.info("Discovering catalogs...")
                discovered_catalogs = await self._catalog_discovery.discover_all(
                    configs=catalog_configs,
                    auto_detect=True,
                )
                report.catalogs = discovered_catalogs
                report.total_catalogs = len(discovered_catalogs)
                report.total_schemas = sum(c.total_schemas for c in discovered_catalogs)
                report.total_datasets = sum(c.total_datasets for c in discovered_catalogs)
            except Exception as e:
                report.errors.append(f"Catalog discovery error: {str(e)}")
                logger.error(f"Catalog discovery failed: {e}")

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        # Calculate duration
        report.discovery_duration_ms = int((time.time() - start_time) * 1000)

        self._report = report
        return report

    def _generate_recommendations(self, report: DiscoveryReport) -> list[str]:
        """Generate recommendations based on discovery results."""
        recommendations = []

        # Check for missing connector dependencies
        if report.missing_dependencies:
            top_missing = list(report.missing_dependencies.keys())[:3]
            recommendations.append(
                f"Install missing dependencies for: {', '.join(top_missing)}"
            )

        # Check for unconfigured data sources
        unconfigured = [
            s for s in report.data_sources
            if not s.has_credentials
        ]
        if unconfigured:
            recommendations.append(
                f"Configure credentials for {len(unconfigured)} data source(s)"
            )

        # Check for catalogs without datasets
        empty_catalogs = [
            c for c in report.catalogs
            if c.total_datasets == 0
        ]
        if empty_catalogs:
            recommendations.append(
                "Some catalogs have no discoverable datasets - check permissions"
            )

        # Suggest governance if not detected
        governance_connectors = [
            c for c in report.connectors
            if c.connector_type == ConnectorType.CATALOG and c.ready
        ]
        if not governance_connectors:
            recommendations.append(
                "Consider setting up a data catalog for governance (Glue, Unity Catalog, Purview)"
            )

        # Suggest query engines if not detected
        query_connectors = [
            c for c in report.connectors
            if c.connector_type in (ConnectorType.SQL_ENGINE, ConnectorType.DATA_WAREHOUSE)
            and c.ready
        ]
        if not query_connectors:
            recommendations.append(
                "No query engines configured - set up Trino, Spark, or a data warehouse"
            )

        return recommendations

    @property
    def report(self) -> Optional[DiscoveryReport]:
        """Get the latest discovery report."""
        return self._report

    @property
    def catalogs(self) -> list[DiscoveredCatalog]:
        """Get discovered catalogs."""
        return self._catalog_discovery.catalogs

    @property
    def connectors(self) -> list[DiscoveredConnector]:
        """Get discovered connectors."""
        return self._connector_discovery.connectors

    @property
    def data_sources(self) -> list[DiscoveredDataSource]:
        """Get discovered data sources."""
        return self._source_discovery.sources

    def get_all_datasets(self) -> list[DiscoveredDataset]:
        """Get all discovered datasets across all catalogs."""
        return self._catalog_discovery.get_all_datasets()

    def search_datasets(
        self,
        pattern: str,
        catalog: Optional[str] = None,
    ) -> list[DiscoveredDataset]:
        """Search for datasets by name pattern."""
        return self._catalog_discovery.search_datasets(pattern, catalog)

    def get_ready_connectors(self) -> list[DiscoveredConnector]:
        """Get connectors that are ready to use."""
        return self._connector_discovery.get_ready_connectors()

    def get_connectors_for_source(
        self,
        source: DiscoveredDataSource,
    ) -> list[DiscoveredConnector]:
        """Get compatible connectors for a data source."""
        # Map source types to connector types
        type_mapping = {
            DataSourceType.DATABASE: [ConnectorType.SQL_ENGINE],
            DataSourceType.DATA_WAREHOUSE: [ConnectorType.DATA_WAREHOUSE, ConnectorType.SQL_ENGINE],
            DataSourceType.DATA_LAKE: [ConnectorType.DATA_LAKE, ConnectorType.SQL_ENGINE],
            DataSourceType.OBJECT_STORAGE: [ConnectorType.OBJECT_STORAGE],
            DataSourceType.STREAMING: [ConnectorType.STREAMING],
        }

        compatible_types = type_mapping.get(source.source_type, [])

        # Also match by provider name
        provider_match = source.provider.lower() if source.provider else ""

        compatible = []
        for connector in self._connector_discovery.connectors:
            if connector.connector_type in compatible_types:
                compatible.append(connector)
            elif provider_match and provider_match in connector.name.lower():
                compatible.append(connector)

        return compatible


# CLI entry point
async def run_discovery_cli(
    output_format: str = "text",
    catalogs: bool = True,
    connectors: bool = True,
    sources: bool = True,
) -> int:
    """
    Run discovery from command line.

    Args:
        output_format: Output format (text, json, yaml)
        catalogs: Discover catalogs
        connectors: Discover connectors
        sources: Discover data sources

    Returns:
        Exit code (0 = success)
    """
    discovery = UnifiedDiscovery()

    try:
        report = await discovery.discover(
            catalogs=catalogs,
            connectors=connectors,
            data_sources=sources,
            verbose=True,
        )

        if output_format == "json":
            import json
            print(json.dumps(report.to_dict(), indent=2))
        elif output_format == "yaml":
            import yaml
            print(yaml.dump(report.to_dict(), default_flow_style=False))
        else:
            print(report.print_summary())

        return 0

    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        return 1


def main():
    """CLI entry point."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="DataMesh.AI Discovery - Scan your data environment"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--no-catalogs",
        action="store_true",
        help="Skip catalog discovery",
    )
    parser.add_argument(
        "--no-connectors",
        action="store_true",
        help="Skip connector discovery",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Skip data source discovery",
    )

    args = parser.parse_args()

    exit_code = asyncio.run(run_discovery_cli(
        output_format=args.format,
        catalogs=not args.no_catalogs,
        connectors=not args.no_connectors,
        sources=not args.no_sources,
    ))

    exit(exit_code)


if __name__ == "__main__":
    main()
