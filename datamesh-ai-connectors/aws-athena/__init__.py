"""
AWS Athena Connector for DataMesh.AI

Provides Athena query execution and Glue Catalog integration
for the DataMesh.AI agent ecosystem.
"""

from .athena_client import AthenaClient, AthenaQueryResult
from .glue_catalog import GlueCatalogClient, GlueTable, GlueColumn

__all__ = [
    "AthenaClient",
    "AthenaQueryResult",
    "GlueCatalogClient",
    "GlueTable",
    "GlueColumn",
]

__version__ = "0.1.0"
