"""
DATAMESH.AI Core - Agent lifecycle, registry, A2A messaging, tracing, governance, and discovery.

This module provides the foundational components for building and orchestrating
AI agents in a Data Mesh architecture.
"""

__version__ = "1.2.0"

from .registry import AgentRegistry, AgentContract
from .a2a import A2AClient, A2AMessage, A2AResponse
from .tracing import TracingContext, SpanManager
from .lifecycle import AgentLifecycle, AgentState

# Governance module
from .governance import (
    GovernanceProvider,
    GovernanceConfig,
    GovernanceFactory,
    UnifiedGovernance,
    CloudProvider,
    UserContext,
    Resource,
    Permission,
    Classification,
    AccessDecision,
)

# Discovery module
from .discovery import (
    UnifiedDiscovery,
    DiscoveryReport,
    CatalogDiscovery,
    ConnectorDiscovery,
    DataSourceDiscovery,
    DiscoveredCatalog,
    DiscoveredConnector,
    DiscoveredDataSource,
)

__all__ = [
    # Registry
    "AgentRegistry",
    "AgentContract",
    # A2A
    "A2AClient",
    "A2AMessage",
    "A2AResponse",
    # Tracing
    "TracingContext",
    "SpanManager",
    # Lifecycle
    "AgentLifecycle",
    "AgentState",
    # Governance
    "GovernanceProvider",
    "GovernanceConfig",
    "GovernanceFactory",
    "UnifiedGovernance",
    "CloudProvider",
    "UserContext",
    "Resource",
    "Permission",
    "Classification",
    "AccessDecision",
    # Discovery
    "UnifiedDiscovery",
    "DiscoveryReport",
    "CatalogDiscovery",
    "ConnectorDiscovery",
    "DataSourceDiscovery",
    "DiscoveredCatalog",
    "DiscoveredConnector",
    "DiscoveredDataSource",
]
