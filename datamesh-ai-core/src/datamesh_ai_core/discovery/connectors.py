"""
Connector Discovery - Auto-discover available connectors and their capabilities.

Scans for:
- Installed Python packages (database drivers, cloud SDKs)
- Configuration files
- Environment variables indicating available connections
- Running services
"""

import importlib
import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ConnectorType(Enum):
    """Types of connectors."""
    SQL_ENGINE = "sql_engine"
    DATA_WAREHOUSE = "data_warehouse"
    DATA_LAKE = "data_lake"
    STREAMING = "streaming"
    CATALOG = "catalog"
    GOVERNANCE = "governance"
    OBJECT_STORAGE = "object_storage"
    FILE_SYSTEM = "file_system"
    API = "api"


class ConnectorCapability(Enum):
    """Capabilities a connector may support."""
    QUERY = "query"
    WRITE = "write"
    STREAM = "stream"
    SCHEMA_DISCOVERY = "schema_discovery"
    LINEAGE = "lineage"
    CLASSIFICATION = "classification"
    ACCESS_CONTROL = "access_control"
    PARTITIONING = "partitioning"
    TRANSACTIONS = "transactions"


@dataclass
class ConnectorDependency:
    """Dependency information for a connector."""
    package: str
    min_version: Optional[str] = None
    installed: bool = False
    installed_version: Optional[str] = None


@dataclass
class DiscoveredConnector:
    """Discovered connector metadata."""
    name: str
    connector_type: ConnectorType
    description: str
    capabilities: list[ConnectorCapability] = field(default_factory=list)
    dependencies: list[ConnectorDependency] = field(default_factory=list)
    available: bool = False
    configured: bool = False
    config_source: Optional[str] = None  # env, file, auto
    connection_string_template: Optional[str] = None
    documentation_url: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """Check if connector is ready to use."""
        return self.available and self.configured


# Registry of known connectors
KNOWN_CONNECTORS = {
    # SQL Engines
    "trino": {
        "type": ConnectorType.SQL_ENGINE,
        "description": "Distributed SQL query engine for big data analytics",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.SCHEMA_DISCOVERY,
        ],
        "dependencies": ["trino"],
        "env_vars": ["TRINO_HOST", "TRINO_URI"],
        "connection_template": "trino://{user}@{host}:{port}/{catalog}",
        "docs": "https://trino.io/docs/current/",
    },
    "presto": {
        "type": ConnectorType.SQL_ENGINE,
        "description": "Distributed SQL query engine",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.SCHEMA_DISCOVERY,
        ],
        "dependencies": ["presto-python-client"],
        "env_vars": ["PRESTO_HOST"],
        "connection_template": "presto://{user}@{host}:{port}/{catalog}",
        "docs": "https://prestodb.io/docs/current/",
    },
    "spark": {
        "type": ConnectorType.SQL_ENGINE,
        "description": "Apache Spark SQL engine",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.WRITE,
            ConnectorCapability.SCHEMA_DISCOVERY,
            ConnectorCapability.PARTITIONING,
        ],
        "dependencies": ["pyspark"],
        "env_vars": ["SPARK_HOME", "SPARK_MASTER"],
        "docs": "https://spark.apache.org/docs/latest/",
    },
    # Data Warehouses
    "snowflake": {
        "type": ConnectorType.DATA_WAREHOUSE,
        "description": "Snowflake cloud data warehouse",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.WRITE,
            ConnectorCapability.SCHEMA_DISCOVERY,
            ConnectorCapability.ACCESS_CONTROL,
        ],
        "dependencies": ["snowflake-connector-python"],
        "env_vars": ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER"],
        "connection_template": "snowflake://{user}:{password}@{account}/{database}/{schema}",
        "docs": "https://docs.snowflake.com/",
    },
    "bigquery": {
        "type": ConnectorType.DATA_WAREHOUSE,
        "description": "Google BigQuery serverless data warehouse",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.WRITE,
            ConnectorCapability.SCHEMA_DISCOVERY,
            ConnectorCapability.PARTITIONING,
        ],
        "dependencies": ["google-cloud-bigquery"],
        "env_vars": ["GOOGLE_CLOUD_PROJECT", "BIGQUERY_PROJECT"],
        "docs": "https://cloud.google.com/bigquery/docs",
    },
    "redshift": {
        "type": ConnectorType.DATA_WAREHOUSE,
        "description": "Amazon Redshift data warehouse",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.WRITE,
            ConnectorCapability.SCHEMA_DISCOVERY,
        ],
        "dependencies": ["redshift_connector"],
        "env_vars": ["REDSHIFT_HOST", "REDSHIFT_CLUSTER"],
        "connection_template": "redshift+redshift_connector://{user}:{password}@{host}:{port}/{database}",
        "docs": "https://docs.aws.amazon.com/redshift/",
    },
    "athena": {
        "type": ConnectorType.SQL_ENGINE,
        "description": "Amazon Athena serverless SQL",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.SCHEMA_DISCOVERY,
        ],
        "dependencies": ["pyathena", "boto3"],
        "env_vars": ["AWS_REGION", "ATHENA_S3_OUTPUT"],
        "docs": "https://docs.aws.amazon.com/athena/",
    },
    "synapse": {
        "type": ConnectorType.DATA_WAREHOUSE,
        "description": "Azure Synapse Analytics",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.WRITE,
            ConnectorCapability.SCHEMA_DISCOVERY,
        ],
        "dependencies": ["azure-synapse", "pyodbc"],
        "env_vars": ["SYNAPSE_WORKSPACE", "AZURE_SYNAPSE_CONNECTION"],
        "docs": "https://docs.microsoft.com/azure/synapse-analytics/",
    },
    "databricks": {
        "type": ConnectorType.DATA_WAREHOUSE,
        "description": "Databricks Lakehouse Platform",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.WRITE,
            ConnectorCapability.SCHEMA_DISCOVERY,
            ConnectorCapability.ACCESS_CONTROL,
            ConnectorCapability.LINEAGE,
        ],
        "dependencies": ["databricks-sdk", "databricks-sql-connector"],
        "env_vars": ["DATABRICKS_HOST", "DATABRICKS_TOKEN"],
        "docs": "https://docs.databricks.com/",
    },
    # Object Storage
    "s3": {
        "type": ConnectorType.OBJECT_STORAGE,
        "description": "Amazon S3 object storage",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.WRITE,
        ],
        "dependencies": ["boto3"],
        "env_vars": ["AWS_ACCESS_KEY_ID", "AWS_REGION"],
        "docs": "https://docs.aws.amazon.com/s3/",
    },
    "gcs": {
        "type": ConnectorType.OBJECT_STORAGE,
        "description": "Google Cloud Storage",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.WRITE,
        ],
        "dependencies": ["google-cloud-storage"],
        "env_vars": ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"],
        "docs": "https://cloud.google.com/storage/docs",
    },
    "azure_blob": {
        "type": ConnectorType.OBJECT_STORAGE,
        "description": "Azure Blob Storage",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.WRITE,
        ],
        "dependencies": ["azure-storage-blob"],
        "env_vars": ["AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT"],
        "docs": "https://docs.microsoft.com/azure/storage/blobs/",
    },
    # Streaming
    "kafka": {
        "type": ConnectorType.STREAMING,
        "description": "Apache Kafka streaming platform",
        "capabilities": [
            ConnectorCapability.STREAM,
            ConnectorCapability.WRITE,
        ],
        "dependencies": ["confluent-kafka", "kafka-python"],
        "env_vars": ["KAFKA_BOOTSTRAP_SERVERS", "KAFKA_BROKERS"],
        "docs": "https://kafka.apache.org/documentation/",
    },
    "kinesis": {
        "type": ConnectorType.STREAMING,
        "description": "Amazon Kinesis data streaming",
        "capabilities": [
            ConnectorCapability.STREAM,
            ConnectorCapability.WRITE,
        ],
        "dependencies": ["boto3"],
        "env_vars": ["AWS_REGION", "KINESIS_STREAM"],
        "docs": "https://docs.aws.amazon.com/kinesis/",
    },
    # Traditional Databases
    "postgresql": {
        "type": ConnectorType.SQL_ENGINE,
        "description": "PostgreSQL database",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.WRITE,
            ConnectorCapability.SCHEMA_DISCOVERY,
            ConnectorCapability.TRANSACTIONS,
        ],
        "dependencies": ["psycopg2-binary", "asyncpg"],
        "env_vars": ["POSTGRES_HOST", "PGHOST", "DATABASE_URL"],
        "connection_template": "postgresql://{user}:{password}@{host}:{port}/{database}",
        "docs": "https://www.postgresql.org/docs/",
    },
    "mysql": {
        "type": ConnectorType.SQL_ENGINE,
        "description": "MySQL database",
        "capabilities": [
            ConnectorCapability.QUERY,
            ConnectorCapability.WRITE,
            ConnectorCapability.SCHEMA_DISCOVERY,
            ConnectorCapability.TRANSACTIONS,
        ],
        "dependencies": ["mysql-connector-python", "pymysql"],
        "env_vars": ["MYSQL_HOST", "MYSQL_DATABASE"],
        "connection_template": "mysql://{user}:{password}@{host}:{port}/{database}",
        "docs": "https://dev.mysql.com/doc/",
    },
    # Catalogs
    "glue": {
        "type": ConnectorType.CATALOG,
        "description": "AWS Glue Data Catalog",
        "capabilities": [
            ConnectorCapability.SCHEMA_DISCOVERY,
            ConnectorCapability.LINEAGE,
        ],
        "dependencies": ["boto3"],
        "env_vars": ["AWS_REGION"],
        "docs": "https://docs.aws.amazon.com/glue/",
    },
    "unity_catalog": {
        "type": ConnectorType.CATALOG,
        "description": "Databricks Unity Catalog",
        "capabilities": [
            ConnectorCapability.SCHEMA_DISCOVERY,
            ConnectorCapability.ACCESS_CONTROL,
            ConnectorCapability.LINEAGE,
            ConnectorCapability.CLASSIFICATION,
        ],
        "dependencies": ["databricks-sdk"],
        "env_vars": ["DATABRICKS_HOST", "DATABRICKS_TOKEN"],
        "docs": "https://docs.databricks.com/data-governance/unity-catalog/",
    },
    "purview": {
        "type": ConnectorType.CATALOG,
        "description": "Microsoft Purview (Azure)",
        "capabilities": [
            ConnectorCapability.SCHEMA_DISCOVERY,
            ConnectorCapability.LINEAGE,
            ConnectorCapability.CLASSIFICATION,
        ],
        "dependencies": ["azure-purview-catalog"],
        "env_vars": ["AZURE_PURVIEW_ACCOUNT"],
        "docs": "https://docs.microsoft.com/azure/purview/",
    },
}


class ConnectorDiscovery:
    """
    Connector discovery service.

    Automatically discovers available connectors based on:
    - Installed Python packages
    - Environment variables
    - Configuration files
    """

    def __init__(self):
        self._discovered_connectors: list[DiscoveredConnector] = []
        self._installed_packages: dict[str, str] = {}

    async def discover_all(self) -> list[DiscoveredConnector]:
        """
        Discover all available connectors.

        Returns:
            List of discovered connectors
        """
        self._discovered_connectors = []
        self._installed_packages = self._get_installed_packages()

        for name, info in KNOWN_CONNECTORS.items():
            connector = self._check_connector(name, info)
            self._discovered_connectors.append(connector)

        # Sort by availability and readiness
        self._discovered_connectors.sort(
            key=lambda c: (not c.ready, not c.available, c.name)
        )

        available_count = sum(1 for c in self._discovered_connectors if c.available)
        ready_count = sum(1 for c in self._discovered_connectors if c.ready)
        logger.info(
            f"Discovered {len(self._discovered_connectors)} connectors: "
            f"{available_count} available, {ready_count} ready"
        )

        return self._discovered_connectors

    def _check_connector(
        self, name: str, info: dict
    ) -> DiscoveredConnector:
        """Check if a connector is available and configured."""
        dependencies = []
        all_deps_installed = True

        # Check dependencies
        for dep in info.get("dependencies", []):
            installed = dep in self._installed_packages
            version = self._installed_packages.get(dep)

            dependencies.append(ConnectorDependency(
                package=dep,
                installed=installed,
                installed_version=version,
            ))

            if not installed:
                all_deps_installed = False

        # Check configuration (environment variables)
        configured = False
        config_source = None
        env_vars = info.get("env_vars", [])

        for env_var in env_vars:
            if os.environ.get(env_var):
                configured = True
                config_source = "env"
                break

        connector = DiscoveredConnector(
            name=name,
            connector_type=info["type"],
            description=info["description"],
            capabilities=[
                ConnectorCapability(cap) if isinstance(cap, str) else cap
                for cap in info.get("capabilities", [])
            ],
            dependencies=dependencies,
            available=all_deps_installed,
            configured=configured,
            config_source=config_source,
            connection_string_template=info.get("connection_template"),
            documentation_url=info.get("docs"),
        )

        return connector

    def _get_installed_packages(self) -> dict[str, str]:
        """Get installed Python packages and versions."""
        packages = {}

        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                import json
                for pkg in json.loads(result.stdout):
                    packages[pkg["name"].lower().replace("_", "-")] = pkg["version"]

        except Exception as e:
            logger.warning(f"Could not list installed packages: {e}")

            # Fallback: try importing known packages
            for name, info in KNOWN_CONNECTORS.items():
                for dep in info.get("dependencies", []):
                    try:
                        # Convert package name to module name
                        module_name = dep.replace("-", "_").split("[")[0]
                        mod = importlib.import_module(module_name)
                        version = getattr(mod, "__version__", "unknown")
                        packages[dep] = version
                    except ImportError:
                        pass

        return packages

    @property
    def connectors(self) -> list[DiscoveredConnector]:
        """Get discovered connectors."""
        return self._discovered_connectors

    def get_available_connectors(self) -> list[DiscoveredConnector]:
        """Get connectors that have dependencies installed."""
        return [c for c in self._discovered_connectors if c.available]

    def get_ready_connectors(self) -> list[DiscoveredConnector]:
        """Get connectors that are ready to use."""
        return [c for c in self._discovered_connectors if c.ready]

    def get_connectors_by_type(
        self, connector_type: ConnectorType
    ) -> list[DiscoveredConnector]:
        """Get connectors of a specific type."""
        return [
            c for c in self._discovered_connectors
            if c.connector_type == connector_type
        ]

    def get_connectors_with_capability(
        self, capability: ConnectorCapability
    ) -> list[DiscoveredConnector]:
        """Get connectors that support a specific capability."""
        return [
            c for c in self._discovered_connectors
            if capability in c.capabilities
        ]

    def get_missing_dependencies(self) -> dict[str, list[str]]:
        """Get missing dependencies for each unavailable connector."""
        missing = {}
        for connector in self._discovered_connectors:
            if not connector.available:
                missing_deps = [
                    d.package for d in connector.dependencies if not d.installed
                ]
                if missing_deps:
                    missing[connector.name] = missing_deps
        return missing

    def get_installation_commands(self) -> list[str]:
        """Get pip install commands for missing dependencies."""
        commands = []
        missing = self.get_missing_dependencies()

        for connector_name, deps in missing.items():
            commands.append(f"# For {connector_name}")
            commands.append(f"pip install {' '.join(deps)}")

        return commands

    @classmethod
    def register_connector(
        cls,
        name: str,
        connector_type: ConnectorType,
        description: str,
        capabilities: list[ConnectorCapability],
        dependencies: list[str],
        env_vars: list[str],
        connection_template: Optional[str] = None,
        docs_url: Optional[str] = None,
    ) -> None:
        """Register a custom connector."""
        KNOWN_CONNECTORS[name] = {
            "type": connector_type,
            "description": description,
            "capabilities": capabilities,
            "dependencies": dependencies,
            "env_vars": env_vars,
            "connection_template": connection_template,
            "docs": docs_url,
        }
