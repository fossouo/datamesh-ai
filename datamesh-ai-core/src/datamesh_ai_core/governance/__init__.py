"""
DATAMESH.AI Governance - Cloud-agnostic governance abstraction layer.

This module provides a unified interface for interacting with various cloud
provider governance solutions:
- AWS Lake Formation
- Azure Purview
- GCP Dataplex
- Databricks Unity Catalog

The governance layer automatically discovers and adapts to the user's cloud
provider based on configured credentials.
"""

from .interfaces import (
    GovernanceProvider,
    UserContext,
    Resource,
    Permission,
    Classification,
    MaskingRule,
    RowFilter,
    AccessDecision,
    AuditEntry,
)
from .config import GovernanceConfig, CloudProvider
from .factory import GovernanceFactory
from .unified import UnifiedGovernance

__all__ = [
    # Interfaces
    "GovernanceProvider",
    "UserContext",
    "Resource",
    "Permission",
    "Classification",
    "MaskingRule",
    "RowFilter",
    "AccessDecision",
    "AuditEntry",
    # Configuration
    "GovernanceConfig",
    "CloudProvider",
    # Factory & Unified
    "GovernanceFactory",
    "UnifiedGovernance",
]
