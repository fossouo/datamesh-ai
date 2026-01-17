"""
Catalog Discovery - Auto-discover data catalogs across cloud providers.

Supports:
- AWS Glue Data Catalog
- Azure Purview
- GCP Data Catalog
- Databricks Unity Catalog
- Apache Hive Metastore
- Open Metadata
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CatalogType(Enum):
    """Supported catalog types."""
    AWS_GLUE = "aws_glue"
    AZURE_PURVIEW = "azure_purview"
    GCP_DATACATALOG = "gcp_datacatalog"
    DATABRICKS_UNITY = "databricks_unity"
    HIVE_METASTORE = "hive_metastore"
    OPEN_METADATA = "open_metadata"
    CUSTOM = "custom"


@dataclass
class DiscoveredColumn:
    """Discovered column metadata."""
    name: str
    data_type: str
    nullable: bool = True
    comment: Optional[str] = None
    is_partition: bool = False
    classification: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class DiscoveredDataset:
    """Discovered dataset (table/view) metadata."""
    name: str
    database: str
    catalog: str
    dataset_type: str  # table, view, external_table, etc.
    columns: list[DiscoveredColumn] = field(default_factory=list)
    location: Optional[str] = None
    format: Optional[str] = None  # parquet, csv, json, etc.
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    owner: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    classifications: list[str] = field(default_factory=list)
    partition_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fully_qualified_name(self) -> str:
        return f"{self.catalog}.{self.database}.{self.name}"


@dataclass
class DiscoveredSchema:
    """Discovered schema/database metadata."""
    name: str
    catalog: str
    datasets: list[DiscoveredDataset] = field(default_factory=list)
    description: Optional[str] = None
    owner: Optional[str] = None
    location: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveredCatalog:
    """Discovered catalog metadata."""
    name: str
    catalog_type: CatalogType
    provider: str
    schemas: list[DiscoveredSchema] = field(default_factory=list)
    endpoint: Optional[str] = None
    region: Optional[str] = None
    description: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_datasets(self) -> int:
        return sum(len(schema.datasets) for schema in self.schemas)

    @property
    def total_schemas(self) -> int:
        return len(self.schemas)


class CatalogScanner(ABC):
    """Abstract base class for catalog scanners."""

    @property
    @abstractmethod
    def catalog_type(self) -> CatalogType:
        """Return the catalog type."""
        pass

    @abstractmethod
    async def scan(self, config: dict) -> list[DiscoveredCatalog]:
        """Scan and discover catalogs."""
        pass

    @abstractmethod
    async def scan_datasets(
        self,
        catalog: str,
        schema: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> list[DiscoveredDataset]:
        """Scan datasets in a catalog/schema."""
        pass


class AWSGlueCatalogScanner(CatalogScanner):
    """Scanner for AWS Glue Data Catalog."""

    @property
    def catalog_type(self) -> CatalogType:
        return CatalogType.AWS_GLUE

    async def scan(self, config: dict) -> list[DiscoveredCatalog]:
        """Scan AWS Glue catalogs."""
        catalogs = []

        try:
            import boto3

            region = config.get("region", "us-east-1")
            profile = config.get("profile")

            session_kwargs = {}
            if profile:
                session_kwargs["profile_name"] = profile

            session = boto3.Session(**session_kwargs)
            glue = session.client("glue", region_name=region)

            # Get account ID for catalog name
            sts = session.client("sts", region_name=region)
            account_id = sts.get_caller_identity()["Account"]

            catalog = DiscoveredCatalog(
                name=f"aws-glue-{account_id}",
                catalog_type=CatalogType.AWS_GLUE,
                provider="aws",
                endpoint=f"glue.{region}.amazonaws.com",
                region=region,
                metadata={"account_id": account_id},
            )

            # List databases
            paginator = glue.get_paginator("get_databases")
            for page in paginator.paginate():
                for db in page.get("DatabaseList", []):
                    schema = DiscoveredSchema(
                        name=db["Name"],
                        catalog=catalog.name,
                        description=db.get("Description"),
                        location=db.get("LocationUri"),
                        metadata={
                            "create_time": str(db.get("CreateTime")),
                            "parameters": db.get("Parameters", {}),
                        },
                    )

                    # Optionally scan tables (can be slow for large catalogs)
                    if config.get("scan_tables", True):
                        schema.datasets = await self._scan_database_tables(
                            glue, db["Name"], catalog.name
                        )

                    catalog.schemas.append(schema)

            catalogs.append(catalog)
            logger.info(f"Discovered AWS Glue catalog with {catalog.total_schemas} schemas")

        except ImportError:
            logger.warning("boto3 not installed, skipping AWS Glue discovery")
        except Exception as e:
            logger.error(f"Error scanning AWS Glue: {e}")

        return catalogs

    async def _scan_database_tables(
        self, glue, database: str, catalog_name: str
    ) -> list[DiscoveredDataset]:
        """Scan tables in a database."""
        datasets = []

        try:
            paginator = glue.get_paginator("get_tables")
            for page in paginator.paginate(DatabaseName=database):
                for table in page.get("TableList", []):
                    storage = table.get("StorageDescriptor", {})

                    columns = [
                        DiscoveredColumn(
                            name=col["Name"],
                            data_type=col["Type"],
                            comment=col.get("Comment"),
                        )
                        for col in storage.get("Columns", [])
                    ]

                    # Add partition keys as columns
                    for pk in table.get("PartitionKeys", []):
                        columns.append(DiscoveredColumn(
                            name=pk["Name"],
                            data_type=pk["Type"],
                            comment=pk.get("Comment"),
                            is_partition=True,
                        ))

                    dataset = DiscoveredDataset(
                        name=table["Name"],
                        database=database,
                        catalog=catalog_name,
                        dataset_type=table.get("TableType", "EXTERNAL_TABLE"),
                        columns=columns,
                        location=storage.get("Location"),
                        format=storage.get("InputFormat", "").split(".")[-1],
                        created_at=table.get("CreateTime"),
                        updated_at=table.get("UpdateTime"),
                        owner=table.get("Owner"),
                        description=table.get("Description"),
                        partition_keys=[pk["Name"] for pk in table.get("PartitionKeys", [])],
                        metadata={
                            "parameters": table.get("Parameters", {}),
                            "serde": storage.get("SerdeInfo", {}),
                        },
                    )

                    # Extract classifications from parameters
                    params = table.get("Parameters", {})
                    if "classification" in params:
                        dataset.classifications.append(params["classification"])

                    datasets.append(dataset)

        except Exception as e:
            logger.warning(f"Error scanning tables in {database}: {e}")

        return datasets

    async def scan_datasets(
        self,
        catalog: str,
        schema: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> list[DiscoveredDataset]:
        """Scan datasets matching criteria."""
        # Implementation for targeted scanning
        return []


class DatabricksUnityCatalogScanner(CatalogScanner):
    """Scanner for Databricks Unity Catalog."""

    @property
    def catalog_type(self) -> CatalogType:
        return CatalogType.DATABRICKS_UNITY

    async def scan(self, config: dict) -> list[DiscoveredCatalog]:
        """Scan Databricks Unity Catalogs."""
        catalogs = []

        try:
            from databricks.sdk import WorkspaceClient

            workspace_url = config.get("workspace_url")
            token = config.get("token")

            client = WorkspaceClient(host=workspace_url, token=token)

            # List all catalogs
            for cat in client.catalogs.list():
                catalog = DiscoveredCatalog(
                    name=cat.name,
                    catalog_type=CatalogType.DATABRICKS_UNITY,
                    provider="databricks",
                    endpoint=workspace_url,
                    description=cat.comment,
                    metadata={
                        "owner": cat.owner,
                        "metastore_id": cat.metastore_id,
                    },
                )

                # List schemas
                if config.get("scan_schemas", True):
                    for schema in client.schemas.list(catalog_name=cat.name):
                        discovered_schema = DiscoveredSchema(
                            name=schema.name,
                            catalog=cat.name,
                            description=schema.comment,
                            owner=schema.owner,
                        )

                        # List tables
                        if config.get("scan_tables", True):
                            for table in client.tables.list(
                                catalog_name=cat.name,
                                schema_name=schema.name,
                            ):
                                columns = [
                                    DiscoveredColumn(
                                        name=col.name,
                                        data_type=col.type_text,
                                        nullable=col.nullable,
                                        comment=col.comment,
                                    )
                                    for col in (table.columns or [])
                                ]

                                dataset = DiscoveredDataset(
                                    name=table.name,
                                    database=schema.name,
                                    catalog=cat.name,
                                    dataset_type=table.table_type.value if table.table_type else "TABLE",
                                    columns=columns,
                                    location=table.storage_location,
                                    format=table.data_source_format.value if table.data_source_format else None,
                                    owner=table.owner,
                                    description=table.comment,
                                    created_at=datetime.fromtimestamp(table.created_at / 1000) if table.created_at else None,
                                    updated_at=datetime.fromtimestamp(table.updated_at / 1000) if table.updated_at else None,
                                )
                                discovered_schema.datasets.append(dataset)

                        catalog.schemas.append(discovered_schema)

                catalogs.append(catalog)

            logger.info(f"Discovered {len(catalogs)} Databricks Unity Catalogs")

        except ImportError:
            logger.warning("databricks-sdk not installed, skipping Unity Catalog discovery")
        except Exception as e:
            logger.error(f"Error scanning Databricks Unity Catalog: {e}")

        return catalogs

    async def scan_datasets(
        self,
        catalog: str,
        schema: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> list[DiscoveredDataset]:
        """Scan datasets matching criteria."""
        return []


class AzurePurviewCatalogScanner(CatalogScanner):
    """Scanner for Azure Purview."""

    @property
    def catalog_type(self) -> CatalogType:
        return CatalogType.AZURE_PURVIEW

    async def scan(self, config: dict) -> list[DiscoveredCatalog]:
        """Scan Azure Purview catalog."""
        catalogs = []

        try:
            from azure.identity import DefaultAzureCredential
            from azure.purview.catalog import PurviewCatalogClient

            purview_account = config.get("purview_account")
            endpoint = config.get("endpoint") or f"https://{purview_account}.purview.azure.com"

            credential = DefaultAzureCredential()
            client = PurviewCatalogClient(endpoint=endpoint, credential=credential)

            catalog = DiscoveredCatalog(
                name=purview_account,
                catalog_type=CatalogType.AZURE_PURVIEW,
                provider="azure",
                endpoint=endpoint,
            )

            # Search for all assets
            if config.get("scan_assets", True):
                search_request = {"keywords": "*", "limit": 1000}
                results = client.discovery.query(search_request)

                # Group by qualified name structure
                schemas_map = {}
                for entity in results.get("value", []):
                    qn = entity.get("qualifiedName", "")
                    entity_type = entity.get("entityType", "")

                    # Parse qualified name to extract database/schema
                    parts = qn.split("/")
                    if len(parts) >= 2:
                        schema_name = parts[-2] if len(parts) > 2 else "default"
                        table_name = parts[-1]

                        if schema_name not in schemas_map:
                            schemas_map[schema_name] = DiscoveredSchema(
                                name=schema_name,
                                catalog=purview_account,
                            )

                        dataset = DiscoveredDataset(
                            name=table_name,
                            database=schema_name,
                            catalog=purview_account,
                            dataset_type=entity_type,
                            metadata={
                                "purview_id": entity.get("id"),
                                "qualified_name": qn,
                            },
                        )
                        schemas_map[schema_name].datasets.append(dataset)

                catalog.schemas = list(schemas_map.values())

            catalogs.append(catalog)
            logger.info(f"Discovered Azure Purview catalog with {catalog.total_datasets} assets")

        except ImportError:
            logger.warning("azure-purview-catalog not installed, skipping Purview discovery")
        except Exception as e:
            logger.error(f"Error scanning Azure Purview: {e}")

        return catalogs

    async def scan_datasets(
        self,
        catalog: str,
        schema: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> list[DiscoveredDataset]:
        """Scan datasets matching criteria."""
        return []


class GCPDataCatalogScanner(CatalogScanner):
    """Scanner for GCP Data Catalog."""

    @property
    def catalog_type(self) -> CatalogType:
        return CatalogType.GCP_DATACATALOG

    async def scan(self, config: dict) -> list[DiscoveredCatalog]:
        """Scan GCP Data Catalog."""
        catalogs = []

        try:
            from google.cloud import datacatalog_v1

            project_id = config.get("project_id")
            location = config.get("location", "us")

            client = datacatalog_v1.DataCatalogClient()

            catalog = DiscoveredCatalog(
                name=f"gcp-datacatalog-{project_id}",
                catalog_type=CatalogType.GCP_DATACATALOG,
                provider="gcp",
                region=location,
                metadata={"project_id": project_id},
            )

            # Search for entries
            scope = datacatalog_v1.SearchCatalogRequest.Scope(
                include_project_ids=[project_id],
            )

            request = datacatalog_v1.SearchCatalogRequest(
                scope=scope,
                query="*",
                page_size=100,
            )

            schemas_map = {}
            for result in client.search_catalog(request=request):
                linked_resource = result.linked_resource or ""

                # Parse BigQuery resource paths
                if "bigquery.googleapis.com" in linked_resource:
                    parts = linked_resource.split("/")
                    if "datasets" in parts:
                        dataset_idx = parts.index("datasets")
                        dataset_name = parts[dataset_idx + 1]

                        if dataset_name not in schemas_map:
                            schemas_map[dataset_name] = DiscoveredSchema(
                                name=dataset_name,
                                catalog=catalog.name,
                            )

                        if "tables" in parts:
                            table_idx = parts.index("tables")
                            table_name = parts[table_idx + 1]

                            ds = DiscoveredDataset(
                                name=table_name,
                                database=dataset_name,
                                catalog=catalog.name,
                                dataset_type="table",
                                metadata={
                                    "linked_resource": linked_resource,
                                    "entry_name": result.relative_resource_name,
                                },
                            )
                            schemas_map[dataset_name].datasets.append(ds)

            catalog.schemas = list(schemas_map.values())
            catalogs.append(catalog)
            logger.info(f"Discovered GCP Data Catalog with {catalog.total_datasets} entries")

        except ImportError:
            logger.warning("google-cloud-datacatalog not installed, skipping GCP discovery")
        except Exception as e:
            logger.error(f"Error scanning GCP Data Catalog: {e}")

        return catalogs

    async def scan_datasets(
        self,
        catalog: str,
        schema: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> list[DiscoveredDataset]:
        """Scan datasets matching criteria."""
        return []


class CatalogDiscovery:
    """
    Multi-catalog discovery service.

    Automatically scans and discovers data catalogs across multiple
    cloud providers and systems.
    """

    _scanners: dict[CatalogType, type] = {
        CatalogType.AWS_GLUE: AWSGlueCatalogScanner,
        CatalogType.DATABRICKS_UNITY: DatabricksUnityCatalogScanner,
        CatalogType.AZURE_PURVIEW: AzurePurviewCatalogScanner,
        CatalogType.GCP_DATACATALOG: GCPDataCatalogScanner,
    }

    def __init__(self):
        self._discovered_catalogs: list[DiscoveredCatalog] = []

    async def discover_all(
        self,
        configs: Optional[dict[CatalogType, dict]] = None,
        auto_detect: bool = True,
    ) -> list[DiscoveredCatalog]:
        """
        Discover all available catalogs.

        Args:
            configs: Per-catalog-type configuration
            auto_detect: Auto-detect available catalogs from environment

        Returns:
            List of discovered catalogs
        """
        configs = configs or {}
        self._discovered_catalogs = []

        if auto_detect:
            detected = self._auto_detect_catalogs()
            for catalog_type in detected:
                if catalog_type not in configs:
                    configs[catalog_type] = detected[catalog_type]

        for catalog_type, config in configs.items():
            scanner_class = self._scanners.get(catalog_type)
            if scanner_class:
                try:
                    scanner = scanner_class()
                    catalogs = await scanner.scan(config)
                    self._discovered_catalogs.extend(catalogs)
                except Exception as e:
                    logger.error(f"Error scanning {catalog_type.value}: {e}")

        return self._discovered_catalogs

    async def discover_catalog(
        self,
        catalog_type: CatalogType,
        config: dict,
    ) -> list[DiscoveredCatalog]:
        """
        Discover a specific catalog type.

        Args:
            catalog_type: Type of catalog to discover
            config: Configuration for the scanner

        Returns:
            List of discovered catalogs
        """
        scanner_class = self._scanners.get(catalog_type)
        if not scanner_class:
            raise ValueError(f"No scanner for catalog type: {catalog_type}")

        scanner = scanner_class()
        return await scanner.scan(config)

    def _auto_detect_catalogs(self) -> dict[CatalogType, dict]:
        """Auto-detect available catalogs from environment."""
        import os

        detected = {}

        # Check AWS
        if any([
            os.environ.get("AWS_ACCESS_KEY_ID"),
            os.environ.get("AWS_PROFILE"),
            os.path.exists(os.path.expanduser("~/.aws/credentials")),
        ]):
            detected[CatalogType.AWS_GLUE] = {
                "region": os.environ.get("AWS_REGION", "us-east-1"),
                "profile": os.environ.get("AWS_PROFILE"),
            }

        # Check Databricks
        if os.environ.get("DATABRICKS_HOST"):
            detected[CatalogType.DATABRICKS_UNITY] = {
                "workspace_url": os.environ.get("DATABRICKS_HOST"),
                "token": os.environ.get("DATABRICKS_TOKEN"),
            }

        # Check Azure
        if os.environ.get("AZURE_PURVIEW_ACCOUNT"):
            detected[CatalogType.AZURE_PURVIEW] = {
                "purview_account": os.environ.get("AZURE_PURVIEW_ACCOUNT"),
            }

        # Check GCP
        if os.environ.get("GOOGLE_CLOUD_PROJECT"):
            detected[CatalogType.GCP_DATACATALOG] = {
                "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT"),
            }

        return detected

    @property
    def catalogs(self) -> list[DiscoveredCatalog]:
        """Get discovered catalogs."""
        return self._discovered_catalogs

    def get_all_datasets(self) -> list[DiscoveredDataset]:
        """Get all datasets across all catalogs."""
        datasets = []
        for catalog in self._discovered_catalogs:
            for schema in catalog.schemas:
                datasets.extend(schema.datasets)
        return datasets

    def search_datasets(
        self,
        pattern: str,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> list[DiscoveredDataset]:
        """
        Search datasets by name pattern.

        Args:
            pattern: Name pattern (supports * wildcard)
            catalog: Filter by catalog name
            schema: Filter by schema name

        Returns:
            Matching datasets
        """
        import fnmatch

        results = []
        for ds in self.get_all_datasets():
            if catalog and ds.catalog != catalog:
                continue
            if schema and ds.database != schema:
                continue
            if fnmatch.fnmatch(ds.name.lower(), pattern.lower()):
                results.append(ds)

        return results

    @classmethod
    def register_scanner(
        cls,
        catalog_type: CatalogType,
        scanner_class: type,
    ) -> None:
        """Register a custom catalog scanner."""
        cls._scanners[catalog_type] = scanner_class
