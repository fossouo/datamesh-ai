"""
DATAMESH.AI Discovery - Auto-discovery for catalogs, connectors, and data sources.

This module provides automatic discovery capabilities to scan and identify:
- Data catalogs across cloud providers
- Available connectors and their configurations
- Data sources (databases, data lakes, warehouses)
- Schemas, tables, and metadata
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
]
