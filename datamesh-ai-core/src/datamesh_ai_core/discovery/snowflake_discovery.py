"""
DATAMESH.AI Snowflake Discovery - Auto-discovery for Snowflake data warehouses.

This module provides comprehensive discovery capabilities for Snowflake:
- Account and region discovery
- Warehouse discovery and sizing
- Database, schema, and table discovery
- Stage discovery (internal and external)
- Data sharing discovery
- Clustering and partitioning analysis
- Access control and governance
- Integration with other cloud providers (S3, Azure, GCS)
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime


class SnowflakeEdition(Enum):
    """Snowflake account editions."""
    STANDARD = "standard"
    ENTERPRISE = "enterprise"
    BUSINESS_CRITICAL = "business_critical"
    VIRTUAL_PRIVATE = "virtual_private"


class StageType(Enum):
    """Types of Snowflake stages."""
    INTERNAL = "internal"
    EXTERNAL_S3 = "external_s3"
    EXTERNAL_AZURE = "external_azure"
    EXTERNAL_GCS = "external_gcs"


class TableType(Enum):
    """Types of Snowflake tables."""
    PERMANENT = "permanent"
    TRANSIENT = "transient"
    TEMPORARY = "temporary"
    EXTERNAL = "external"
    DYNAMIC = "dynamic"
    ICEBERG = "iceberg"


@dataclass
class DiscoveredSnowflakeAccount:
    """Discovered Snowflake account information."""
    account_identifier: str
    account_locator: str
    region: str
    cloud_provider: str  # aws, azure, gcp
    edition: Optional[SnowflakeEdition] = None
    organization_name: Optional[str] = None
    account_name: Optional[str] = None
    is_reader: bool = False
    created_on: Optional[datetime] = None


@dataclass
class DiscoveredWarehouse:
    """Discovered Snowflake warehouse."""
    name: str
    size: str  # X-Small, Small, Medium, Large, etc.
    state: str  # STARTED, SUSPENDED
    type: str  # STANDARD, SNOWPARK_OPTIMIZED
    auto_suspend: Optional[int] = None  # seconds
    auto_resume: bool = True
    min_cluster_count: int = 1
    max_cluster_count: int = 1
    scaling_policy: str = "STANDARD"
    resource_monitor: Optional[str] = None
    query_tag: Optional[str] = None
    created_on: Optional[datetime] = None
    owner: Optional[str] = None


@dataclass
class DiscoveredDatabase:
    """Discovered Snowflake database."""
    name: str
    owner: Optional[str] = None
    comment: Optional[str] = None
    is_transient: bool = False
    is_default: bool = False
    retention_time: int = 1  # days
    created_on: Optional[datetime] = None
    schemas: List["DiscoveredSnowflakeSchema"] = field(default_factory=list)
    origin: Optional[str] = None  # For shared databases
    is_shared: bool = False


@dataclass
class DiscoveredSnowflakeSchema:
    """Discovered Snowflake schema."""
    name: str
    database: str
    owner: Optional[str] = None
    comment: Optional[str] = None
    is_transient: bool = False
    is_managed_access: bool = False
    retention_time: int = 1
    created_on: Optional[datetime] = None
    table_count: int = 0
    view_count: int = 0
    stage_count: int = 0
    procedure_count: int = 0
    function_count: int = 0


@dataclass
class DiscoveredSnowflakeTable:
    """Discovered Snowflake table."""
    name: str
    database: str
    schema: str
    table_type: TableType = TableType.PERMANENT
    owner: Optional[str] = None
    comment: Optional[str] = None
    row_count: Optional[int] = None
    bytes: Optional[int] = None
    created_on: Optional[datetime] = None
    last_altered: Optional[datetime] = None
    columns: List[Dict[str, Any]] = field(default_factory=list)
    clustering_key: Optional[str] = None
    is_clustered: bool = False
    change_tracking: bool = False
    search_optimization: bool = False
    # External table specific
    location: Optional[str] = None
    file_format: Optional[str] = None
    # Iceberg specific
    iceberg_catalog: Optional[str] = None
    iceberg_base_location: Optional[str] = None


@dataclass
class DiscoveredStage:
    """Discovered Snowflake stage."""
    name: str
    database: str
    schema: str
    stage_type: StageType
    url: Optional[str] = None  # External stage URL
    storage_integration: Optional[str] = None
    file_format: Optional[str] = None
    owner: Optional[str] = None
    comment: Optional[str] = None
    created_on: Optional[datetime] = None
    # Cloud-specific
    cloud_provider: Optional[str] = None
    bucket_name: Optional[str] = None
    prefix: Optional[str] = None


@dataclass
class DiscoveredStorageIntegration:
    """Discovered Snowflake storage integration."""
    name: str
    type: str  # EXTERNAL_STAGE
    enabled: bool = True
    storage_provider: Optional[str] = None  # S3, AZURE, GCS
    storage_allowed_locations: List[str] = field(default_factory=list)
    storage_blocked_locations: List[str] = field(default_factory=list)
    # AWS specific
    storage_aws_role_arn: Optional[str] = None
    storage_aws_external_id: Optional[str] = None
    # Azure specific
    azure_tenant_id: Optional[str] = None
    # GCP specific
    storage_gcp_service_account: Optional[str] = None
    created_on: Optional[datetime] = None


@dataclass
class DiscoveredShare:
    """Discovered Snowflake data share."""
    name: str
    kind: str  # INBOUND, OUTBOUND
    owner: Optional[str] = None
    comment: Optional[str] = None
    created_on: Optional[datetime] = None
    # Outbound share specific
    database_name: Optional[str] = None
    to_accounts: List[str] = field(default_factory=list)
    # Inbound share specific
    from_account: Optional[str] = None
    from_organization: Optional[str] = None


@dataclass
class ClusteringSuggestion:
    """Suggestion for table clustering."""
    database: str
    schema: str
    table: str
    current_clustering_key: Optional[str]
    suggested_clustering_key: str
    reason: str
    estimated_benefit: str
    ddl_command: str


@dataclass
class SnowflakeDiscoveryResult:
    """Complete Snowflake discovery result."""
    account: Optional[DiscoveredSnowflakeAccount] = None
    warehouses: List[DiscoveredWarehouse] = field(default_factory=list)
    databases: List[DiscoveredDatabase] = field(default_factory=list)
    tables: List[DiscoveredSnowflakeTable] = field(default_factory=list)
    stages: List[DiscoveredStage] = field(default_factory=list)
    storage_integrations: List[DiscoveredStorageIntegration] = field(default_factory=list)
    shares: List[DiscoveredShare] = field(default_factory=list)
    clustering_suggestions: List[ClusteringSuggestion] = field(default_factory=list)
    discovery_time: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)


class SnowflakeDiscovery:
    """
    Snowflake discovery implementation.

    Discovers Snowflake account resources including warehouses, databases,
    schemas, tables, stages, and data shares.
    """

    def __init__(
        self,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        private_key_path: Optional[str] = None,
        warehouse: Optional[str] = None,
        role: Optional[str] = None,
        authenticator: Optional[str] = None,
    ):
        """
        Initialize Snowflake discovery.

        Args:
            account: Snowflake account identifier
            user: Username
            password: Password (or use private key)
            private_key_path: Path to private key file
            warehouse: Default warehouse to use
            role: Role to use
            authenticator: Authentication method (externalbrowser, snowflake, etc.)
        """
        self.account = account or os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = user or os.getenv("SNOWFLAKE_USER")
        self.password = password or os.getenv("SNOWFLAKE_PASSWORD")
        self.private_key_path = private_key_path or os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
        self.warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE")
        self.role = role or os.getenv("SNOWFLAKE_ROLE")
        self.authenticator = authenticator or os.getenv("SNOWFLAKE_AUTHENTICATOR")

        self._connection = None

    def _get_connection(self):
        """Get or create Snowflake connection."""
        if self._connection:
            return self._connection

        try:
            import snowflake.connector
        except ImportError:
            raise ImportError(
                "snowflake-connector-python is required. "
                "Install with: pip install snowflake-connector-python"
            )

        connect_params = {
            "account": self.account,
            "user": self.user,
        }

        if self.password:
            connect_params["password"] = self.password
        elif self.private_key_path:
            with open(self.private_key_path, "rb") as key_file:
                from cryptography.hazmat.backends import default_backend
                from cryptography.hazmat.primitives import serialization

                p_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                    backend=default_backend(),
                )
                connect_params["private_key"] = p_key

        if self.warehouse:
            connect_params["warehouse"] = self.warehouse
        if self.role:
            connect_params["role"] = self.role
        if self.authenticator:
            connect_params["authenticator"] = self.authenticator

        self._connection = snowflake.connector.connect(**connect_params)
        return self._connection

    def _execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dicts."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()

    async def discover(
        self,
        discover_warehouses: bool = True,
        discover_databases: bool = True,
        discover_tables: bool = True,
        discover_stages: bool = True,
        discover_integrations: bool = True,
        discover_shares: bool = True,
        analyze_clustering: bool = True,
        database_filter: Optional[List[str]] = None,
        exclude_system_databases: bool = True,
    ) -> SnowflakeDiscoveryResult:
        """
        Discover Snowflake resources.

        Args:
            discover_warehouses: Discover warehouses
            discover_databases: Discover databases and schemas
            discover_tables: Discover tables (requires discover_databases)
            discover_stages: Discover stages
            discover_integrations: Discover storage integrations
            discover_shares: Discover data shares
            analyze_clustering: Analyze and suggest clustering keys
            database_filter: Only discover specific databases
            exclude_system_databases: Exclude SNOWFLAKE and SNOWFLAKE_SAMPLE_DATA

        Returns:
            SnowflakeDiscoveryResult
        """
        result = SnowflakeDiscoveryResult()

        try:
            # Discover account info
            result.account = await self._discover_account()

            # Discover warehouses
            if discover_warehouses:
                result.warehouses = await self._discover_warehouses()

            # Discover databases
            if discover_databases:
                result.databases = await self._discover_databases(
                    database_filter, exclude_system_databases
                )

            # Discover tables
            if discover_tables and result.databases:
                result.tables = await self._discover_tables(result.databases)

            # Discover stages
            if discover_stages:
                result.stages = await self._discover_stages(result.databases)

            # Discover storage integrations
            if discover_integrations:
                result.storage_integrations = await self._discover_storage_integrations()

            # Discover shares
            if discover_shares:
                result.shares = await self._discover_shares()

            # Analyze clustering
            if analyze_clustering and result.tables:
                result.clustering_suggestions = await self._analyze_clustering(result.tables)

        except Exception as e:
            result.errors.append(str(e))

        return result

    async def _discover_account(self) -> Optional[DiscoveredSnowflakeAccount]:
        """Discover account information."""
        try:
            # Get current account info
            account_info = self._execute_query("SELECT CURRENT_ACCOUNT(), CURRENT_REGION()")

            if account_info:
                row = account_info[0]
                account_locator = row.get("CURRENT_ACCOUNT()", "")
                region = row.get("CURRENT_REGION()", "")

                # Parse region to get cloud provider
                cloud_provider = "aws"
                if "azure" in region.lower():
                    cloud_provider = "azure"
                elif "gcp" in region.lower():
                    cloud_provider = "gcp"

                # Get account details
                try:
                    details = self._execute_query(
                        "SELECT * FROM SNOWFLAKE.ORGANIZATION_USAGE.ACCOUNTS "
                        "WHERE ACCOUNT_LOCATOR = CURRENT_ACCOUNT() LIMIT 1"
                    )
                    if details:
                        d = details[0]
                        return DiscoveredSnowflakeAccount(
                            account_identifier=self.account or "",
                            account_locator=account_locator,
                            region=region,
                            cloud_provider=cloud_provider,
                            organization_name=d.get("ORGANIZATION_NAME"),
                            account_name=d.get("ACCOUNT_NAME"),
                            created_on=d.get("CREATED_ON"),
                        )
                except Exception:
                    pass

                return DiscoveredSnowflakeAccount(
                    account_identifier=self.account or "",
                    account_locator=account_locator,
                    region=region,
                    cloud_provider=cloud_provider,
                )

        except Exception:
            pass

        return None

    async def _discover_warehouses(self) -> List[DiscoveredWarehouse]:
        """Discover warehouses."""
        warehouses = []

        try:
            results = self._execute_query("SHOW WAREHOUSES")

            for row in results:
                warehouses.append(DiscoveredWarehouse(
                    name=row.get("name", ""),
                    size=row.get("size", ""),
                    state=row.get("state", ""),
                    type=row.get("type", "STANDARD"),
                    auto_suspend=row.get("auto_suspend"),
                    auto_resume=row.get("auto_resume", "true").lower() == "true",
                    min_cluster_count=row.get("min_cluster_count", 1),
                    max_cluster_count=row.get("max_cluster_count", 1),
                    scaling_policy=row.get("scaling_policy", "STANDARD"),
                    resource_monitor=row.get("resource_monitor"),
                    created_on=row.get("created_on"),
                    owner=row.get("owner"),
                ))

        except Exception:
            pass

        return warehouses

    async def _discover_databases(
        self,
        database_filter: Optional[List[str]] = None,
        exclude_system: bool = True,
    ) -> List[DiscoveredDatabase]:
        """Discover databases."""
        databases = []
        system_dbs = {"SNOWFLAKE", "SNOWFLAKE_SAMPLE_DATA"}

        try:
            results = self._execute_query("SHOW DATABASES")

            for row in results:
                db_name = row.get("name", "")

                # Filter
                if exclude_system and db_name in system_dbs:
                    continue
                if database_filter and db_name not in database_filter:
                    continue

                db = DiscoveredDatabase(
                    name=db_name,
                    owner=row.get("owner"),
                    comment=row.get("comment"),
                    is_transient=row.get("is_transient", "N") == "Y",
                    is_default=row.get("is_default", "N") == "Y",
                    retention_time=row.get("retention_time", 1),
                    created_on=row.get("created_on"),
                    origin=row.get("origin"),
                    is_shared=bool(row.get("origin")),
                )

                # Get schemas for this database
                try:
                    schemas = self._execute_query(f"SHOW SCHEMAS IN DATABASE {db_name}")
                    for schema_row in schemas:
                        schema_name = schema_row.get("name", "")
                        if schema_name not in ("INFORMATION_SCHEMA", "PUBLIC") or not exclude_system:
                            db.schemas.append(DiscoveredSnowflakeSchema(
                                name=schema_name,
                                database=db_name,
                                owner=schema_row.get("owner"),
                                comment=schema_row.get("comment"),
                                is_transient=schema_row.get("is_transient", "N") == "Y",
                                is_managed_access=schema_row.get("is_managed_access", "N") == "Y",
                                retention_time=schema_row.get("retention_time", 1),
                                created_on=schema_row.get("created_on"),
                            ))
                except Exception:
                    pass

                databases.append(db)

        except Exception:
            pass

        return databases

    async def _discover_tables(
        self,
        databases: List[DiscoveredDatabase],
    ) -> List[DiscoveredSnowflakeTable]:
        """Discover tables in databases."""
        tables = []

        for db in databases:
            for schema in db.schemas:
                try:
                    results = self._execute_query(
                        f"SHOW TABLES IN {db.name}.{schema.name}"
                    )

                    for row in results:
                        table_kind = row.get("kind", "TABLE")
                        table_type = TableType.PERMANENT
                        if table_kind == "TRANSIENT":
                            table_type = TableType.TRANSIENT
                        elif table_kind == "TEMPORARY":
                            table_type = TableType.TEMPORARY
                        elif table_kind == "EXTERNAL":
                            table_type = TableType.EXTERNAL
                        elif "DYNAMIC" in table_kind:
                            table_type = TableType.DYNAMIC
                        elif "ICEBERG" in table_kind:
                            table_type = TableType.ICEBERG

                        table = DiscoveredSnowflakeTable(
                            name=row.get("name", ""),
                            database=db.name,
                            schema=schema.name,
                            table_type=table_type,
                            owner=row.get("owner"),
                            comment=row.get("comment"),
                            row_count=row.get("rows"),
                            bytes=row.get("bytes"),
                            created_on=row.get("created_on"),
                            clustering_key=row.get("cluster_by"),
                            is_clustered=bool(row.get("cluster_by")),
                            change_tracking=row.get("change_tracking", "OFF") == "ON",
                        )

                        # Get columns
                        try:
                            cols = self._execute_query(
                                f"DESCRIBE TABLE {db.name}.{schema.name}.{table.name}"
                            )
                            table.columns = [
                                {
                                    "name": c.get("name"),
                                    "type": c.get("type"),
                                    "nullable": c.get("null?", "Y") == "Y",
                                    "default": c.get("default"),
                                    "comment": c.get("comment"),
                                }
                                for c in cols
                            ]
                        except Exception:
                            pass

                        tables.append(table)

                except Exception:
                    pass

        return tables

    async def _discover_stages(
        self,
        databases: List[DiscoveredDatabase],
    ) -> List[DiscoveredStage]:
        """Discover stages."""
        stages = []

        for db in databases:
            for schema in db.schemas:
                try:
                    results = self._execute_query(
                        f"SHOW STAGES IN {db.name}.{schema.name}"
                    )

                    for row in results:
                        url = row.get("url", "")
                        stage_type = StageType.INTERNAL

                        if url:
                            if url.startswith("s3://"):
                                stage_type = StageType.EXTERNAL_S3
                            elif url.startswith("azure://"):
                                stage_type = StageType.EXTERNAL_AZURE
                            elif url.startswith("gcs://"):
                                stage_type = StageType.EXTERNAL_GCS

                        # Parse bucket and prefix from URL
                        bucket_name = None
                        prefix = None
                        if stage_type != StageType.INTERNAL and url:
                            parts = url.replace("s3://", "").replace("azure://", "").replace("gcs://", "").split("/", 1)
                            bucket_name = parts[0] if parts else None
                            prefix = parts[1] if len(parts) > 1 else None

                        stages.append(DiscoveredStage(
                            name=row.get("name", ""),
                            database=db.name,
                            schema=schema.name,
                            stage_type=stage_type,
                            url=url,
                            storage_integration=row.get("storage_integration"),
                            owner=row.get("owner"),
                            comment=row.get("comment"),
                            created_on=row.get("created_on"),
                            cloud_provider=stage_type.value.replace("external_", "") if stage_type != StageType.INTERNAL else None,
                            bucket_name=bucket_name,
                            prefix=prefix,
                        ))

                except Exception:
                    pass

        return stages

    async def _discover_storage_integrations(self) -> List[DiscoveredStorageIntegration]:
        """Discover storage integrations."""
        integrations = []

        try:
            results = self._execute_query("SHOW STORAGE INTEGRATIONS")

            for row in results:
                integration = DiscoveredStorageIntegration(
                    name=row.get("name", ""),
                    type=row.get("type", ""),
                    enabled=row.get("enabled", "true").lower() == "true",
                    created_on=row.get("created_on"),
                )

                # Get detailed info
                try:
                    details = self._execute_query(
                        f"DESCRIBE STORAGE INTEGRATION {integration.name}"
                    )
                    detail_dict = {d.get("property"): d.get("property_value") for d in details}

                    integration.storage_provider = detail_dict.get("STORAGE_PROVIDER")
                    integration.storage_allowed_locations = (
                        detail_dict.get("STORAGE_ALLOWED_LOCATIONS", "").split(",")
                        if detail_dict.get("STORAGE_ALLOWED_LOCATIONS") else []
                    )
                    integration.storage_blocked_locations = (
                        detail_dict.get("STORAGE_BLOCKED_LOCATIONS", "").split(",")
                        if detail_dict.get("STORAGE_BLOCKED_LOCATIONS") else []
                    )
                    integration.storage_aws_role_arn = detail_dict.get("STORAGE_AWS_ROLE_ARN")
                    integration.storage_aws_external_id = detail_dict.get("STORAGE_AWS_EXTERNAL_ID")
                    integration.azure_tenant_id = detail_dict.get("AZURE_TENANT_ID")
                    integration.storage_gcp_service_account = detail_dict.get("STORAGE_GCP_SERVICE_ACCOUNT")

                except Exception:
                    pass

                integrations.append(integration)

        except Exception:
            pass

        return integrations

    async def _discover_shares(self) -> List[DiscoveredShare]:
        """Discover data shares."""
        shares = []

        # Outbound shares
        try:
            results = self._execute_query("SHOW SHARES")
            for row in results:
                shares.append(DiscoveredShare(
                    name=row.get("name", ""),
                    kind=row.get("kind", ""),
                    owner=row.get("owner"),
                    comment=row.get("comment"),
                    created_on=row.get("created_on"),
                    database_name=row.get("database_name"),
                    to_accounts=row.get("to", "").split(",") if row.get("to") else [],
                ))
        except Exception:
            pass

        return shares

    async def _analyze_clustering(
        self,
        tables: List[DiscoveredSnowflakeTable],
    ) -> List[ClusteringSuggestion]:
        """Analyze tables and suggest clustering keys."""
        suggestions = []

        for table in tables:
            # Skip small tables and already clustered tables
            if table.bytes and table.bytes < 1_000_000_000:  # < 1GB
                continue

            if not table.columns:
                continue

            # Look for good clustering candidates
            date_columns = []
            high_cardinality_columns = []
            id_columns = []

            for col in table.columns:
                col_name = col.get("name", "").lower()
                col_type = col.get("type", "").lower()

                if "date" in col_type or "timestamp" in col_type:
                    date_columns.append(col.get("name"))
                elif col_name in ("id", "user_id", "customer_id", "account_id", "event_id"):
                    id_columns.append(col.get("name"))
                elif col_name in ("region", "country", "type", "status", "category"):
                    high_cardinality_columns.append(col.get("name"))

            # Generate suggestion
            suggested_key = None
            reason = ""

            if date_columns and not table.is_clustered:
                # Date-based clustering is usually best
                suggested_key = date_columns[0]
                if high_cardinality_columns:
                    suggested_key = f"{date_columns[0]}, {high_cardinality_columns[0]}"
                reason = "Date columns provide good clustering for time-series queries"

            elif id_columns and not table.is_clustered:
                suggested_key = id_columns[0]
                reason = "ID column clustering improves point lookups"

            if suggested_key and suggested_key != table.clustering_key:
                size_gb = table.bytes / (1024 ** 3) if table.bytes else 0
                suggestions.append(ClusteringSuggestion(
                    database=table.database,
                    schema=table.schema,
                    table=table.name,
                    current_clustering_key=table.clustering_key,
                    suggested_clustering_key=suggested_key,
                    reason=reason,
                    estimated_benefit=f"Table size: {size_gb:.1f} GB - clustering can reduce scan by 50-90%",
                    ddl_command=f"ALTER TABLE {table.database}.{table.schema}.{table.name} CLUSTER BY ({suggested_key});",
                ))

        return suggestions

    def close(self):
        """Close the connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


def print_snowflake_discovery(result: SnowflakeDiscoveryResult) -> str:
    """Print Snowflake discovery results in a readable format."""
    lines = [
        "",
        "=" * 70,
        "  SNOWFLAKE DISCOVERY RESULTS",
        "=" * 70,
    ]

    # Account info
    if result.account:
        lines.append("\n  ACCOUNT:")
        lines.append(f"    Identifier: {result.account.account_identifier}")
        lines.append(f"    Locator: {result.account.account_locator}")
        lines.append(f"    Region: {result.account.region}")
        lines.append(f"    Cloud: {result.account.cloud_provider.upper()}")
        if result.account.organization_name:
            lines.append(f"    Organization: {result.account.organization_name}")

    # Warehouses
    if result.warehouses:
        lines.append(f"\n  WAREHOUSES ({len(result.warehouses)}):")
        for wh in result.warehouses:
            state_icon = "●" if wh.state == "STARTED" else "○"
            lines.append(f"    {state_icon} {wh.name} ({wh.size}) - {wh.state}")
            if wh.max_cluster_count > 1:
                lines.append(f"      Multi-cluster: {wh.min_cluster_count}-{wh.max_cluster_count}")

    # Databases
    if result.databases:
        lines.append(f"\n  DATABASES ({len(result.databases)}):")
        for db in result.databases:
            shared = " [SHARED]" if db.is_shared else ""
            lines.append(f"    {db.name}{shared}")
            lines.append(f"      Schemas: {len(db.schemas)}")
            for schema in db.schemas[:5]:
                lines.append(f"        - {schema.name}")
            if len(db.schemas) > 5:
                lines.append(f"        ... and {len(db.schemas) - 5} more")

    # Tables summary
    if result.tables:
        lines.append(f"\n  TABLES ({len(result.tables)}):")

        # Group by type
        by_type = {}
        for t in result.tables:
            by_type.setdefault(t.table_type.value, []).append(t)

        for type_name, tables in by_type.items():
            lines.append(f"    {type_name}: {len(tables)}")

        # Show largest tables
        sorted_tables = sorted(
            [t for t in result.tables if t.bytes],
            key=lambda t: t.bytes or 0,
            reverse=True,
        )[:5]

        if sorted_tables:
            lines.append("\n    Largest tables:")
            for t in sorted_tables:
                size_gb = t.bytes / (1024 ** 3) if t.bytes else 0
                clustered = " [CLUSTERED]" if t.is_clustered else ""
                lines.append(f"      {t.database}.{t.schema}.{t.name}: {size_gb:.2f} GB{clustered}")

    # Stages
    if result.stages:
        lines.append(f"\n  STAGES ({len(result.stages)}):")

        by_type = {}
        for s in result.stages:
            by_type.setdefault(s.stage_type.value, []).append(s)

        for type_name, stages in by_type.items():
            lines.append(f"    {type_name}: {len(stages)}")
            for stage in stages[:3]:
                loc = f" -> {stage.url}" if stage.url else ""
                lines.append(f"      - {stage.database}.{stage.schema}.{stage.name}{loc}")

    # Storage integrations
    if result.storage_integrations:
        lines.append(f"\n  STORAGE INTEGRATIONS ({len(result.storage_integrations)}):")
        for si in result.storage_integrations:
            status = "●" if si.enabled else "○"
            provider = si.storage_provider or "Unknown"
            lines.append(f"    {status} {si.name} ({provider})")
            if si.storage_allowed_locations:
                for loc in si.storage_allowed_locations[:2]:
                    lines.append(f"      Allowed: {loc}")

    # Shares
    if result.shares:
        lines.append(f"\n  DATA SHARES ({len(result.shares)}):")
        for share in result.shares:
            direction = "←" if share.kind == "INBOUND" else "→"
            lines.append(f"    {direction} {share.name} ({share.kind})")
            if share.database_name:
                lines.append(f"      Database: {share.database_name}")
            if share.to_accounts:
                lines.append(f"      To: {', '.join(share.to_accounts[:3])}")

    # Clustering suggestions
    if result.clustering_suggestions:
        lines.append(f"\n  CLUSTERING SUGGESTIONS ({len(result.clustering_suggestions)}):")
        for sug in result.clustering_suggestions[:5]:
            lines.append(f"\n    {sug.database}.{sug.schema}.{sug.table}")
            if sug.current_clustering_key:
                lines.append(f"      Current: {sug.current_clustering_key}")
            lines.append(f"      Suggested: {sug.suggested_clustering_key}")
            lines.append(f"      Reason: {sug.reason}")
            lines.append(f"      DDL: {sug.ddl_command}")

    # Errors
    if result.errors:
        lines.append("\n  ERRORS:")
        for err in result.errors:
            lines.append(f"    ⚠ {err}")

    lines.append("")
    return "\n".join(lines)
