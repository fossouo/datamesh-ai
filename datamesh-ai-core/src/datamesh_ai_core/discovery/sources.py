"""
Data Source Discovery - Auto-discover data sources and their configurations.

Scans for:
- Database connections
- Data lake locations
- Streaming sources
- API endpoints
- File system paths
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of data sources."""
    DATABASE = "database"
    DATA_WAREHOUSE = "data_warehouse"
    DATA_LAKE = "data_lake"
    OBJECT_STORAGE = "object_storage"
    STREAMING = "streaming"
    FILE_SYSTEM = "file_system"
    API = "api"
    UNKNOWN = "unknown"


class DataSourceStatus(Enum):
    """Status of a data source."""
    AVAILABLE = "available"
    UNREACHABLE = "unreachable"
    AUTH_REQUIRED = "auth_required"
    UNKNOWN = "unknown"


@dataclass
class DataSourceCredential:
    """Credential information for a data source."""
    credential_type: str  # env_var, file, iam, token
    source: str  # Name of env var, path to file, etc.
    is_set: bool = False
    expires_at: Optional[datetime] = None


@dataclass
class DiscoveredDataSource:
    """Discovered data source metadata."""
    name: str
    source_type: DataSourceType
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    schema: Optional[str] = None
    path: Optional[str] = None
    uri: Optional[str] = None
    provider: Optional[str] = None  # aws, azure, gcp, etc.
    region: Optional[str] = None
    credentials: list[DataSourceCredential] = field(default_factory=list)
    status: DataSourceStatus = DataSourceStatus.UNKNOWN
    discovered_from: str = ""  # env, config_file, connection_string
    connection_string: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_credentials(self) -> bool:
        """Check if credentials are configured."""
        return any(cred.is_set for cred in self.credentials)

    @property
    def display_name(self) -> str:
        """Get display name for the data source."""
        if self.database:
            return f"{self.name}:{self.database}"
        if self.path:
            return f"{self.name}:{self.path}"
        return self.name


# Environment variable patterns for data sources
ENV_PATTERNS = {
    # Database URLs
    "DATABASE_URL": {
        "type": DataSourceType.DATABASE,
        "pattern": r"^(?P<scheme>\w+)://(?P<user>[^:]+):(?P<password>[^@]+)@(?P<host>[^:/]+)(?::(?P<port>\d+))?/(?P<database>\w+)",
    },
    "POSTGRES_*": {
        "type": DataSourceType.DATABASE,
        "provider": "postgresql",
        "vars": ["POSTGRES_HOST", "PGHOST", "POSTGRES_DB", "PGDATABASE"],
    },
    "MYSQL_*": {
        "type": DataSourceType.DATABASE,
        "provider": "mysql",
        "vars": ["MYSQL_HOST", "MYSQL_DATABASE"],
    },
    # Cloud Data Warehouses
    "SNOWFLAKE_*": {
        "type": DataSourceType.DATA_WAREHOUSE,
        "provider": "snowflake",
        "vars": ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_DATABASE", "SNOWFLAKE_WAREHOUSE"],
    },
    "REDSHIFT_*": {
        "type": DataSourceType.DATA_WAREHOUSE,
        "provider": "redshift",
        "vars": ["REDSHIFT_HOST", "REDSHIFT_DATABASE", "REDSHIFT_CLUSTER"],
    },
    "BIGQUERY_*": {
        "type": DataSourceType.DATA_WAREHOUSE,
        "provider": "bigquery",
        "vars": ["BIGQUERY_PROJECT", "GOOGLE_CLOUD_PROJECT"],
    },
    "DATABRICKS_*": {
        "type": DataSourceType.DATA_WAREHOUSE,
        "provider": "databricks",
        "vars": ["DATABRICKS_HOST", "DATABRICKS_HTTP_PATH"],
    },
    # Object Storage
    "AWS_*": {
        "type": DataSourceType.OBJECT_STORAGE,
        "provider": "aws",
        "vars": ["AWS_ACCESS_KEY_ID", "AWS_REGION", "S3_BUCKET"],
    },
    "AZURE_STORAGE_*": {
        "type": DataSourceType.OBJECT_STORAGE,
        "provider": "azure",
        "vars": ["AZURE_STORAGE_CONNECTION_STRING", "AZURE_STORAGE_ACCOUNT"],
    },
    "GOOGLE_*": {
        "type": DataSourceType.OBJECT_STORAGE,
        "provider": "gcp",
        "vars": ["GOOGLE_APPLICATION_CREDENTIALS", "GCS_BUCKET"],
    },
    # Streaming
    "KAFKA_*": {
        "type": DataSourceType.STREAMING,
        "provider": "kafka",
        "vars": ["KAFKA_BOOTSTRAP_SERVERS", "KAFKA_BROKERS"],
    },
    "KINESIS_*": {
        "type": DataSourceType.STREAMING,
        "provider": "kinesis",
        "vars": ["KINESIS_STREAM_NAME", "AWS_REGION"],
    },
}


class DataSourceDiscovery:
    """
    Data source discovery service.

    Discovers data sources from:
    - Environment variables
    - Connection strings
    - Configuration files
    - Cloud provider metadata
    """

    def __init__(self):
        self._discovered_sources: list[DiscoveredDataSource] = []

    async def discover_all(
        self,
        scan_env: bool = True,
        scan_files: bool = True,
        scan_cloud: bool = True,
    ) -> list[DiscoveredDataSource]:
        """
        Discover all available data sources.

        Args:
            scan_env: Scan environment variables
            scan_files: Scan configuration files
            scan_cloud: Scan cloud provider metadata

        Returns:
            List of discovered data sources
        """
        self._discovered_sources = []

        if scan_env:
            self._discover_from_env()

        if scan_files:
            await self._discover_from_files()

        if scan_cloud:
            await self._discover_from_cloud()

        # Deduplicate by URI or host+database
        self._deduplicate()

        logger.info(f"Discovered {len(self._discovered_sources)} data sources")
        return self._discovered_sources

    def _discover_from_env(self) -> None:
        """Discover data sources from environment variables."""
        env = os.environ

        # Check for DATABASE_URL style connection strings
        for key, value in env.items():
            if "URL" in key and "://" in value:
                source = self._parse_connection_string(key, value)
                if source:
                    self._discovered_sources.append(source)

        # Check for known environment variable patterns
        discovered_providers = set()

        for pattern_name, pattern_info in ENV_PATTERNS.items():
            vars_to_check = pattern_info.get("vars", [])
            provider = pattern_info.get("provider", pattern_name.split("_")[0].lower())

            # Check if any of the pattern's variables are set
            found_vars = {v: env.get(v) for v in vars_to_check if env.get(v)}

            if found_vars and provider not in discovered_providers:
                discovered_providers.add(provider)

                source = DiscoveredDataSource(
                    name=provider,
                    source_type=pattern_info["type"],
                    provider=provider,
                    discovered_from="env",
                    credentials=[
                        DataSourceCredential(
                            credential_type="env_var",
                            source=var,
                            is_set=True,
                        )
                        for var in found_vars.keys()
                    ],
                    metadata={"env_vars": found_vars},
                )

                # Extract specific details based on provider
                self._enrich_source_from_env(source, found_vars)
                self._discovered_sources.append(source)

    def _parse_connection_string(
        self, key: str, value: str
    ) -> Optional[DiscoveredDataSource]:
        """Parse a connection string URL."""
        try:
            parsed = urlparse(value)
            scheme = parsed.scheme.lower()

            # Map schemes to source types
            scheme_mapping = {
                "postgresql": (DataSourceType.DATABASE, "postgresql"),
                "postgres": (DataSourceType.DATABASE, "postgresql"),
                "mysql": (DataSourceType.DATABASE, "mysql"),
                "mysql+pymysql": (DataSourceType.DATABASE, "mysql"),
                "redshift": (DataSourceType.DATA_WAREHOUSE, "redshift"),
                "redshift+redshift_connector": (DataSourceType.DATA_WAREHOUSE, "redshift"),
                "snowflake": (DataSourceType.DATA_WAREHOUSE, "snowflake"),
                "bigquery": (DataSourceType.DATA_WAREHOUSE, "bigquery"),
                "trino": (DataSourceType.DATABASE, "trino"),
                "presto": (DataSourceType.DATABASE, "presto"),
                "databricks": (DataSourceType.DATA_WAREHOUSE, "databricks"),
            }

            source_type, provider = scheme_mapping.get(
                scheme, (DataSourceType.DATABASE, scheme)
            )

            # Extract path components
            path_parts = parsed.path.strip("/").split("/")
            database = path_parts[0] if path_parts else None
            schema = path_parts[1] if len(path_parts) > 1 else None

            return DiscoveredDataSource(
                name=f"{provider}-{key.lower()}",
                source_type=source_type,
                host=parsed.hostname,
                port=parsed.port,
                database=database,
                schema=schema,
                provider=provider,
                discovered_from="env",
                connection_string=value.replace(parsed.password or "", "***") if parsed.password else value,
                credentials=[
                    DataSourceCredential(
                        credential_type="connection_string",
                        source=key,
                        is_set=True,
                    )
                ],
                status=DataSourceStatus.UNKNOWN,
            )

        except Exception as e:
            logger.debug(f"Could not parse connection string {key}: {e}")
            return None

    def _enrich_source_from_env(
        self,
        source: DiscoveredDataSource,
        env_vars: dict[str, str],
    ) -> None:
        """Enrich source with details from environment variables."""
        provider = source.provider

        if provider == "postgresql":
            source.host = env_vars.get("POSTGRES_HOST") or env_vars.get("PGHOST")
            source.port = int(env_vars.get("POSTGRES_PORT", env_vars.get("PGPORT", "5432")))
            source.database = env_vars.get("POSTGRES_DB") or env_vars.get("PGDATABASE")

        elif provider == "mysql":
            source.host = env_vars.get("MYSQL_HOST")
            source.port = int(env_vars.get("MYSQL_PORT", "3306"))
            source.database = env_vars.get("MYSQL_DATABASE")

        elif provider == "snowflake":
            source.host = f"{env_vars.get('SNOWFLAKE_ACCOUNT', '')}.snowflakecomputing.com"
            source.database = env_vars.get("SNOWFLAKE_DATABASE")
            source.metadata["warehouse"] = env_vars.get("SNOWFLAKE_WAREHOUSE")

        elif provider == "redshift":
            source.host = env_vars.get("REDSHIFT_HOST")
            source.port = int(env_vars.get("REDSHIFT_PORT", "5439"))
            source.database = env_vars.get("REDSHIFT_DATABASE")

        elif provider == "bigquery":
            source.database = env_vars.get("BIGQUERY_PROJECT") or env_vars.get("GOOGLE_CLOUD_PROJECT")
            source.region = env_vars.get("BIGQUERY_LOCATION", "US")

        elif provider == "databricks":
            source.host = env_vars.get("DATABRICKS_HOST")
            source.metadata["http_path"] = env_vars.get("DATABRICKS_HTTP_PATH")

        elif provider == "aws":
            source.region = env_vars.get("AWS_REGION", env_vars.get("AWS_DEFAULT_REGION"))
            source.path = env_vars.get("S3_BUCKET")

        elif provider == "kafka":
            brokers = env_vars.get("KAFKA_BOOTSTRAP_SERVERS") or env_vars.get("KAFKA_BROKERS")
            if brokers:
                source.host = brokers.split(",")[0].split(":")[0]
                source.metadata["brokers"] = brokers

    async def _discover_from_files(self) -> None:
        """Discover data sources from configuration files."""
        config_locations = [
            os.path.expanduser("~/.datamesh/connections.yaml"),
            os.path.expanduser("~/.datamesh/connections.json"),
            ".datamesh/connections.yaml",
            ".datamesh/connections.json",
            "connections.yaml",
            "connections.json",
        ]

        for config_path in config_locations:
            if os.path.exists(config_path):
                await self._parse_config_file(config_path)

    async def _parse_config_file(self, path: str) -> None:
        """Parse a configuration file for data sources."""
        try:
            with open(path, "r") as f:
                content = f.read()

            if path.endswith(".yaml") or path.endswith(".yml"):
                import yaml
                config = yaml.safe_load(content)
            else:
                import json
                config = json.loads(content)

            connections = config.get("connections", config.get("sources", []))

            for conn in connections:
                if isinstance(conn, dict):
                    source = DiscoveredDataSource(
                        name=conn.get("name", "unnamed"),
                        source_type=DataSourceType(conn.get("type", "database")),
                        host=conn.get("host"),
                        port=conn.get("port"),
                        database=conn.get("database"),
                        schema=conn.get("schema"),
                        provider=conn.get("provider"),
                        discovered_from=f"file:{path}",
                    )
                    self._discovered_sources.append(source)

            logger.info(f"Loaded {len(connections)} connections from {path}")

        except Exception as e:
            logger.debug(f"Could not parse config file {path}: {e}")

    async def _discover_from_cloud(self) -> None:
        """Discover data sources from cloud provider metadata."""
        # AWS: Check for RDS, Redshift clusters
        await self._discover_aws_sources()

        # Azure: Check for SQL databases, Synapse
        await self._discover_azure_sources()

        # GCP: Check for Cloud SQL, BigQuery
        await self._discover_gcp_sources()

    async def _discover_aws_sources(self) -> None:
        """Discover AWS data sources."""
        if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_PROFILE"):
            return

        try:
            import boto3

            region = os.environ.get("AWS_REGION", "us-east-1")
            session = boto3.Session()

            # Check for RDS instances
            try:
                rds = session.client("rds", region_name=region)
                instances = rds.describe_db_instances()

                for instance in instances.get("DBInstances", []):
                    source = DiscoveredDataSource(
                        name=f"rds-{instance['DBInstanceIdentifier']}",
                        source_type=DataSourceType.DATABASE,
                        host=instance.get("Endpoint", {}).get("Address"),
                        port=instance.get("Endpoint", {}).get("Port"),
                        database=instance.get("DBName"),
                        provider=f"aws-rds-{instance.get('Engine', 'unknown')}",
                        region=region,
                        discovered_from="aws-api",
                        status=DataSourceStatus.AVAILABLE if instance.get("DBInstanceStatus") == "available" else DataSourceStatus.UNREACHABLE,
                        metadata={
                            "engine": instance.get("Engine"),
                            "engine_version": instance.get("EngineVersion"),
                            "instance_class": instance.get("DBInstanceClass"),
                        },
                    )
                    self._discovered_sources.append(source)

            except Exception as e:
                logger.debug(f"Could not list RDS instances: {e}")

            # Check for Redshift clusters
            try:
                redshift = session.client("redshift", region_name=region)
                clusters = redshift.describe_clusters()

                for cluster in clusters.get("Clusters", []):
                    endpoint = cluster.get("Endpoint", {})
                    source = DiscoveredDataSource(
                        name=f"redshift-{cluster['ClusterIdentifier']}",
                        source_type=DataSourceType.DATA_WAREHOUSE,
                        host=endpoint.get("Address"),
                        port=endpoint.get("Port"),
                        database=cluster.get("DBName"),
                        provider="redshift",
                        region=region,
                        discovered_from="aws-api",
                        status=DataSourceStatus.AVAILABLE if cluster.get("ClusterStatus") == "available" else DataSourceStatus.UNREACHABLE,
                        metadata={
                            "node_type": cluster.get("NodeType"),
                            "number_of_nodes": cluster.get("NumberOfNodes"),
                        },
                    )
                    self._discovered_sources.append(source)

            except Exception as e:
                logger.debug(f"Could not list Redshift clusters: {e}")

        except ImportError:
            logger.debug("boto3 not installed, skipping AWS source discovery")
        except Exception as e:
            logger.warning(f"Error discovering AWS sources: {e}")

    async def _discover_azure_sources(self) -> None:
        """Discover Azure data sources."""
        if not os.environ.get("AZURE_SUBSCRIPTION_ID"):
            return

        # Azure discovery would go here
        pass

    async def _discover_gcp_sources(self) -> None:
        """Discover GCP data sources."""
        if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
            return

        # GCP discovery would go here
        pass

    def _deduplicate(self) -> None:
        """Remove duplicate data sources."""
        seen = set()
        unique = []

        for source in self._discovered_sources:
            key = (source.provider, source.host, source.database)
            if key not in seen:
                seen.add(key)
                unique.append(source)

        self._discovered_sources = unique

    @property
    def sources(self) -> list[DiscoveredDataSource]:
        """Get discovered data sources."""
        return self._discovered_sources

    def get_sources_by_type(
        self, source_type: DataSourceType
    ) -> list[DiscoveredDataSource]:
        """Get data sources of a specific type."""
        return [s for s in self._discovered_sources if s.source_type == source_type]

    def get_sources_by_provider(
        self, provider: str
    ) -> list[DiscoveredDataSource]:
        """Get data sources for a specific provider."""
        return [s for s in self._discovered_sources if s.provider == provider]

    def get_available_sources(self) -> list[DiscoveredDataSource]:
        """Get data sources with credentials configured."""
        return [s for s in self._discovered_sources if s.has_credentials]
