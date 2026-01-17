"""
DATAMESH.AI Discovery - Auto-discovery for catalogs, connectors, and data sources.

This module provides automatic discovery capabilities to scan and identify:
- Data catalogs across cloud providers
- Available connectors and their configurations
- Data sources (databases, data lakes, warehouses)
- Schemas, tables, and metadata
- Cloud-specific features (crawlers, scans, assets)
"""

from .catalogs import (
    CatalogDiscovery,
    DiscoveredCatalog,
    DiscoveredDataset,
    DiscoveredSchema,
)
from .connectors import (
    ConnectorDiscovery,
    DiscoveredConnector,
    ConnectorCapability,
)
from .sources import (
    DataSourceDiscovery,
    DiscoveredDataSource,
    DataSourceType,
)
from .unified import UnifiedDiscovery, DiscoveryReport

# Cloud-specific enhanced discovery
from .aws_enhancements import (
    AWSEnhancedDiscovery,
    AWSDiscoveryEnhancements,
    DiscoveredS3Bucket,
    DiscoveredGlueCrawler,
    CrawlerSuggestion,
    print_aws_enhancements,
)
from .azure_enhancements import (
    AzureEnhancedDiscovery,
    AzureDiscoveryEnhancements,
    DiscoveredStorageAccount,
    DiscoveredPurviewAsset,
    DiscoveredDataFactory,
    DiscoveredSynapseWorkspace,
    PurviewScanSuggestion,
    print_azure_enhancements,
)
from .gcp_enhancements import (
    GCPEnhancedDiscovery,
    GCPDiscoveryEnhancements,
    DiscoveredGCSBucket,
    DiscoveredBigQueryDataset,
    DiscoveredDataplexLake,
    DataplexAssetSuggestion,
    print_gcp_enhancements,
)

__all__ = [
    # Catalog discovery
    "CatalogDiscovery",
    "DiscoveredCatalog",
    "DiscoveredDataset",
    "DiscoveredSchema",
    # Connector discovery
    "ConnectorDiscovery",
    "DiscoveredConnector",
    "ConnectorCapability",
    # Data source discovery
    "DataSourceDiscovery",
    "DiscoveredDataSource",
    "DataSourceType",
    # Unified
    "UnifiedDiscovery",
    "DiscoveryReport",
    # AWS Enhanced
    "AWSEnhancedDiscovery",
    "AWSDiscoveryEnhancements",
    "DiscoveredS3Bucket",
    "DiscoveredGlueCrawler",
    "CrawlerSuggestion",
    "print_aws_enhancements",
    # Azure Enhanced
    "AzureEnhancedDiscovery",
    "AzureDiscoveryEnhancements",
    "DiscoveredStorageAccount",
    "DiscoveredPurviewAsset",
    "DiscoveredDataFactory",
    "DiscoveredSynapseWorkspace",
    "PurviewScanSuggestion",
    "print_azure_enhancements",
    # GCP Enhanced
    "GCPEnhancedDiscovery",
    "GCPDiscoveryEnhancements",
    "DiscoveredGCSBucket",
    "DiscoveredBigQueryDataset",
    "DiscoveredDataplexLake",
    "DataplexAssetSuggestion",
    "print_gcp_enhancements",
]
