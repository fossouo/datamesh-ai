#!/usr/bin/env python3
"""
Test DataMesh.AI Discovery on current AWS account.
"""

import asyncio
import sys
import os

# Add the package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'datamesh-ai-core/src'))

from datamesh_ai_core.discovery import (
    UnifiedDiscovery,
    CatalogDiscovery,
    ConnectorDiscovery,
    DataSourceDiscovery,
)
from datamesh_ai_core.discovery.catalogs import CatalogType
from datamesh_ai_core.discovery.aws_enhancements import (
    AWSEnhancedDiscovery,
    print_aws_enhancements,
)


async def main():
    print("\n" + "=" * 70)
    print("  DATAMESH.AI DISCOVERY TEST - AWS Account 874735685088")
    print("=" * 70)

    # Set region
    region = "eu-west-1"
    os.environ.setdefault("AWS_REGION", region)

    # Run unified discovery
    discovery = UnifiedDiscovery()

    print("\n Running discovery...\n")

    report = await discovery.discover(
        catalogs=True,
        connectors=True,
        data_sources=True,
        catalog_configs={
            CatalogType.AWS_GLUE: {
                "region": region,
                "scan_tables": True,
            }
        },
        verbose=True,
    )

    # Print the report
    print(report.print_summary())

    # Run AWS enhanced discovery
    print("\n Running AWS enhanced discovery (S3, Crawlers, Lake Formation)...\n")

    aws_discovery = AWSEnhancedDiscovery(region=region)
    aws_result = await aws_discovery.discover()

    # Print AWS enhancements
    print(print_aws_enhancements(aws_result))

    # Print detailed catalog info
    if report.catalogs:
        print("\n" + "=" * 70)
        print("  DETAILED CATALOG INFO")
        print("=" * 70)

        for catalog in report.catalogs:
            print(f"\n  Catalog: {catalog.name}")
            print(f"  Type: {catalog.catalog_type.value}")
            print(f"  Region: {catalog.region}")

            for schema in catalog.schemas[:10]:  # Limit to first 10
                print(f"\n    Database: {schema.name}")
                print(f"    Tables: {len(schema.datasets)}")

                for dataset in schema.datasets[:5]:  # Limit to first 5 tables
                    print(f"      - {dataset.name}")
                    if dataset.columns:
                        col_names = [c.name for c in dataset.columns[:5]]
                        print(f"        Columns: {', '.join(col_names)}...")
                    if dataset.location:
                        print(f"        Location: {dataset.location[:60]}...")

                if len(schema.datasets) > 5:
                    print(f"      ... and {len(schema.datasets) - 5} more tables")

            if len(catalog.schemas) > 10:
                print(f"\n    ... and {len(catalog.schemas) - 10} more databases")

    # Print data sources
    if report.data_sources:
        print("\n" + "=" * 70)
        print("  DETAILED DATA SOURCES")
        print("=" * 70)

        for source in report.data_sources:
            print(f"\n  {source.name}")
            print(f"    Type: {source.source_type.value}")
            print(f"    Provider: {source.provider}")
            if source.host:
                print(f"    Host: {source.host}")
            if source.database:
                print(f"    Database: {source.database}")
            if source.region:
                print(f"    Region: {source.region}")
            print(f"    Credentials: {'Configured' if source.has_credentials else 'Not configured'}")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
