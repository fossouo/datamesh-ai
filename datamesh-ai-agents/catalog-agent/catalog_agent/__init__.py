"""
Catalog Agent for DATAMESH.AI

This module provides schema resolution and lineage tracking capabilities
for the DATAMESH.AI framework.

Capabilities:
- catalog.resolve: Resolves dataset URIs to field metadata
- catalog.lineage: Tracks upstream/downstream data lineage
"""

from catalog_agent.resolver import SchemaResolver
from catalog_agent.lineage import LineageTracker
from catalog_agent.handler import handle_request

__version__ = "1.0.0"
__all__ = ["SchemaResolver", "LineageTracker", "handle_request"]
