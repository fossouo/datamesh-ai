"""
GCP Enhancements - Additional GCP-specific discovery features.

Includes:
- GCS bucket discovery and analysis
- BigQuery dataset and table discovery
- Dataplex lake and zone detection
- Data Catalog entry scanning
- Dataflow job discovery
- Cloud Composer (Airflow) detection
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredGCSBucket:
    """Discovered Google Cloud Storage bucket."""
    name: str
    location: str
    storage_class: str
    created: Optional[datetime] = None
    is_catalogued: bool = False
    catalogued_prefixes: list[str] = field(default_factory=list)
    uncatalogued_prefixes: list[str] = field(default_factory=list)
    data_formats: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveredBigQueryDataset:
    """Discovered BigQuery dataset."""
    dataset_id: str
    project_id: str
    location: str
    tables: list[str] = field(default_factory=list)
    views: list[str] = field(default_factory=list)
    table_count: int = 0
    total_bytes: int = 0
    created: Optional[datetime] = None
    labels: dict[str, str] = field(default_factory=dict)
    is_in_dataplex: bool = False


@dataclass
class DiscoveredBigQueryTable:
    """Discovered BigQuery table."""
    table_id: str
    dataset_id: str
    project_id: str
    table_type: str  # TABLE, VIEW, EXTERNAL
    num_rows: int = 0
    num_bytes: int = 0
    columns: list[str] = field(default_factory=list)
    partition_field: Optional[str] = None
    clustering_fields: list[str] = field(default_factory=list)
    has_policy_tags: bool = False
    policy_tags: list[str] = field(default_factory=list)


@dataclass
class DiscoveredDataplexLake:
    """Discovered Dataplex lake."""
    name: str
    project_id: str
    location: str
    display_name: str
    zones: list[str] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)
    state: str = "UNKNOWN"


@dataclass
class DiscoveredDataCatalogEntry:
    """Discovered Data Catalog entry."""
    name: str
    display_name: str
    entry_type: str
    linked_resource: str
    tags: list[str] = field(default_factory=list)
    has_policy_tags: bool = False


@dataclass
class DataplexAssetSuggestion:
    """Suggested Dataplex asset configuration."""
    name: str
    lake: str
    zone: str
    resource_type: str  # STORAGE_BUCKET, BIGQUERY_DATASET
    resource_name: str
    description: str
    reason: str
    gcloud_command: str = ""
    terraform_snippet: str = ""


@dataclass
class GCPDiscoveryEnhancements:
    """Enhanced GCP discovery results."""
    buckets: list[DiscoveredGCSBucket] = field(default_factory=list)
    bigquery_datasets: list[DiscoveredBigQueryDataset] = field(default_factory=list)
    bigquery_tables: list[DiscoveredBigQueryTable] = field(default_factory=list)
    dataplex_lakes: list[DiscoveredDataplexLake] = field(default_factory=list)
    datacatalog_entries: list[DiscoveredDataCatalogEntry] = field(default_factory=list)
    asset_suggestions: list[DataplexAssetSuggestion] = field(default_factory=list)
    project_id: Optional[str] = None
    recommendations: list[str] = field(default_factory=list)


class GCPEnhancedDiscovery:
    """
    Enhanced GCP discovery with GCS, BigQuery, Dataplex, and Data Catalog.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us",
    ):
        self.project_id = project_id
        self.location = location
        self._credentials = None
        self._storage_client = None
        self._bigquery_client = None
        self._dataplex_client = None
        self._datacatalog_client = None

    async def discover(self) -> GCPDiscoveryEnhancements:
        """Run enhanced GCP discovery."""
        result = GCPDiscoveryEnhancements()

        try:
            from google.auth import default

            self._credentials, detected_project = default()
            self.project_id = self.project_id or detected_project
            result.project_id = self.project_id

            if not self.project_id:
                logger.warning("No GCP project ID found")
                return result

            # Discover GCS buckets
            result.buckets = await self._discover_gcs_buckets()

            # Discover BigQuery
            bq_result = await self._discover_bigquery()
            result.bigquery_datasets = bq_result.get("datasets", [])
            result.bigquery_tables = bq_result.get("tables", [])

            # Discover Dataplex
            result.dataplex_lakes = await self._discover_dataplex()

            # Discover Data Catalog
            result.datacatalog_entries = await self._discover_datacatalog()

            # Generate asset suggestions
            result.asset_suggestions = self._generate_asset_suggestions(
                result.buckets,
                result.bigquery_datasets,
                result.dataplex_lakes,
            )

            # Generate recommendations
            result.recommendations = self._generate_recommendations(result)

        except ImportError as e:
            logger.warning(f"GCP SDK not fully installed: {e}")
            result.recommendations.append(
                "Install GCP SDK: pip install google-cloud-storage google-cloud-bigquery "
                "google-cloud-dataplex google-cloud-datacatalog"
            )
        except Exception as e:
            logger.error(f"Error in GCP discovery: {e}")
            result.recommendations.append(f"GCP discovery error: {str(e)}")

        return result

    async def _discover_gcs_buckets(self) -> list[DiscoveredGCSBucket]:
        """Discover Google Cloud Storage buckets."""
        buckets = []

        try:
            from google.cloud import storage

            self._storage_client = storage.Client(
                project=self.project_id,
                credentials=self._credentials,
            )

            for bucket in self._storage_client.list_buckets():
                gcs_bucket = DiscoveredGCSBucket(
                    name=bucket.name,
                    location=bucket.location,
                    storage_class=bucket.storage_class,
                    created=bucket.time_created,
                    labels=dict(bucket.labels) if bucket.labels else {},
                )

                # Analyze bucket contents
                try:
                    prefixes, formats = await self._analyze_bucket(bucket)
                    gcs_bucket.uncatalogued_prefixes = prefixes
                    gcs_bucket.data_formats = formats
                except Exception as e:
                    logger.debug(f"Could not analyze bucket {bucket.name}: {e}")

                buckets.append(gcs_bucket)

        except ImportError:
            logger.warning("google-cloud-storage not installed")
        except Exception as e:
            logger.error(f"Error discovering GCS buckets: {e}")

        return buckets

    async def _analyze_bucket(self, bucket) -> tuple[list[str], list[str]]:
        """Analyze bucket contents for prefixes and formats."""
        prefixes = set()
        formats = set()

        try:
            # List objects with delimiter to get prefixes
            blobs = bucket.list_blobs(max_results=100, delimiter="/")

            # Get top-level prefixes
            for prefix in blobs.prefixes:
                prefixes.add(prefix)

            # Detect formats from object names
            blobs = bucket.list_blobs(max_results=100)
            for blob in blobs:
                name = blob.name.lower()
                if name.endswith(".parquet"):
                    formats.add("parquet")
                elif name.endswith(".json") or name.endswith(".jsonl"):
                    formats.add("json")
                elif name.endswith(".csv"):
                    formats.add("csv")
                elif name.endswith(".avro"):
                    formats.add("avro")
                elif "_delta_log" in name:
                    formats.add("delta")

        except Exception as e:
            logger.debug(f"Error analyzing bucket: {e}")

        return list(prefixes)[:10], list(formats)

    async def _discover_bigquery(self) -> dict:
        """Discover BigQuery datasets and tables."""
        result = {"datasets": [], "tables": []}

        try:
            from google.cloud import bigquery

            self._bigquery_client = bigquery.Client(
                project=self.project_id,
                credentials=self._credentials,
            )

            for dataset_ref in self._bigquery_client.list_datasets():
                dataset = self._bigquery_client.get_dataset(dataset_ref.dataset_id)

                bq_dataset = DiscoveredBigQueryDataset(
                    dataset_id=dataset.dataset_id,
                    project_id=dataset.project,
                    location=dataset.location,
                    created=dataset.created,
                    labels=dict(dataset.labels) if dataset.labels else {},
                )

                # List tables
                tables_list = []
                views_list = []
                total_bytes = 0

                for table_ref in self._bigquery_client.list_tables(dataset.dataset_id):
                    try:
                        table = self._bigquery_client.get_table(table_ref)

                        if table.table_type == "VIEW":
                            views_list.append(table.table_id)
                        else:
                            tables_list.append(table.table_id)

                        total_bytes += table.num_bytes or 0

                        # Create detailed table entry
                        bq_table = DiscoveredBigQueryTable(
                            table_id=table.table_id,
                            dataset_id=dataset.dataset_id,
                            project_id=self.project_id,
                            table_type=table.table_type,
                            num_rows=table.num_rows or 0,
                            num_bytes=table.num_bytes or 0,
                            columns=[f.name for f in table.schema] if table.schema else [],
                            partition_field=table.time_partitioning.field if table.time_partitioning else None,
                            clustering_fields=list(table.clustering_fields) if table.clustering_fields else [],
                        )

                        # Check for policy tags
                        if table.schema:
                            for field in table.schema:
                                if field.policy_tags:
                                    bq_table.has_policy_tags = True
                                    bq_table.policy_tags.extend(
                                        [pt.name for pt in field.policy_tags.names] if hasattr(field.policy_tags, 'names') else []
                                    )

                        result["tables"].append(bq_table)

                    except Exception as e:
                        logger.debug(f"Could not get table details: {e}")
                        tables_list.append(table_ref.table_id)

                bq_dataset.tables = tables_list
                bq_dataset.views = views_list
                bq_dataset.table_count = len(tables_list) + len(views_list)
                bq_dataset.total_bytes = total_bytes

                result["datasets"].append(bq_dataset)

        except ImportError:
            logger.warning("google-cloud-bigquery not installed")
        except Exception as e:
            logger.error(f"Error discovering BigQuery: {e}")

        return result

    async def _discover_dataplex(self) -> list[DiscoveredDataplexLake]:
        """Discover Dataplex lakes and zones."""
        lakes = []

        try:
            from google.cloud import dataplex_v1

            self._dataplex_client = dataplex_v1.DataplexServiceClient(
                credentials=self._credentials
            )

            # List lakes
            parent = f"projects/{self.project_id}/locations/{self.location}"

            try:
                for lake in self._dataplex_client.list_lakes(parent=parent):
                    dataplex_lake = DiscoveredDataplexLake(
                        name=lake.name,
                        project_id=self.project_id,
                        location=self.location,
                        display_name=lake.display_name,
                        state=lake.state.name if lake.state else "UNKNOWN",
                    )

                    # List zones
                    try:
                        for zone in self._dataplex_client.list_zones(parent=lake.name):
                            dataplex_lake.zones.append(zone.display_name or zone.name)

                            # List assets
                            for asset in self._dataplex_client.list_assets(parent=zone.name):
                                dataplex_lake.assets.append(asset.display_name or asset.name)
                    except Exception:
                        pass

                    lakes.append(dataplex_lake)

            except Exception as e:
                logger.debug(f"Could not list Dataplex lakes: {e}")

        except ImportError:
            logger.debug("google-cloud-dataplex not installed")
        except Exception as e:
            logger.debug(f"Error discovering Dataplex: {e}")

        return lakes

    async def _discover_datacatalog(self) -> list[DiscoveredDataCatalogEntry]:
        """Discover Data Catalog entries."""
        entries = []

        try:
            from google.cloud import datacatalog_v1

            self._datacatalog_client = datacatalog_v1.DataCatalogClient(
                credentials=self._credentials
            )

            # Search for entries
            scope = datacatalog_v1.SearchCatalogRequest.Scope(
                include_project_ids=[self.project_id],
            )

            request = datacatalog_v1.SearchCatalogRequest(
                scope=scope,
                query="*",
                page_size=100,
            )

            for result in self._datacatalog_client.search_catalog(request=request):
                entry = DiscoveredDataCatalogEntry(
                    name=result.relative_resource_name,
                    display_name=result.display_name or "",
                    entry_type=result.search_result_type.name,
                    linked_resource=result.linked_resource or "",
                )
                entries.append(entry)

        except ImportError:
            logger.debug("google-cloud-datacatalog not installed")
        except Exception as e:
            logger.debug(f"Error discovering Data Catalog: {e}")

        return entries

    def _generate_asset_suggestions(
        self,
        buckets: list[DiscoveredGCSBucket],
        datasets: list[DiscoveredBigQueryDataset],
        lakes: list[DiscoveredDataplexLake],
    ) -> list[DataplexAssetSuggestion]:
        """Generate Dataplex asset suggestions."""
        suggestions = []

        # Get existing assets in Dataplex
        existing_assets = set()
        for lake in lakes:
            existing_assets.update(lake.assets)

        # If no lakes exist, suggest creating one first
        if not lakes:
            return suggestions

        lake = lakes[0]  # Use first lake for suggestions

        # Suggest GCS buckets as assets
        for bucket in buckets:
            # Skip if likely already an asset
            if any(bucket.name in asset for asset in existing_assets):
                continue

            # Skip system buckets
            if bucket.name.startswith("gcf-") or "staging" in bucket.name.lower():
                continue

            if bucket.data_formats:  # Only suggest buckets with data
                suggestion = DataplexAssetSuggestion(
                    name=f"asset-{bucket.name}",
                    lake=lake.display_name,
                    zone=lake.zones[0] if lake.zones else "raw-zone",
                    resource_type="STORAGE_BUCKET",
                    resource_name=f"projects/{self.project_id}/buckets/{bucket.name}",
                    description=f"Add {bucket.name} to Dataplex",
                    reason=f"Found {', '.join(bucket.data_formats)} data not managed by Dataplex",
                    gcloud_command=self._generate_dataplex_gcloud(
                        lake.name, bucket.name, "STORAGE_BUCKET"
                    ),
                    terraform_snippet=self._generate_dataplex_terraform(
                        lake.name.split("/")[-1], bucket.name, "STORAGE_BUCKET"
                    ),
                )
                suggestions.append(suggestion)

        # Suggest BigQuery datasets as assets
        for dataset in datasets:
            if dataset.is_in_dataplex:
                continue

            if any(dataset.dataset_id in asset for asset in existing_assets):
                continue

            suggestion = DataplexAssetSuggestion(
                name=f"asset-{dataset.dataset_id}",
                lake=lake.display_name,
                zone=lake.zones[0] if lake.zones else "curated-zone",
                resource_type="BIGQUERY_DATASET",
                resource_name=f"projects/{self.project_id}/datasets/{dataset.dataset_id}",
                description=f"Add BigQuery dataset {dataset.dataset_id} to Dataplex",
                reason=f"Dataset with {dataset.table_count} tables not managed by Dataplex",
                gcloud_command=self._generate_dataplex_gcloud(
                    lake.name, dataset.dataset_id, "BIGQUERY_DATASET"
                ),
            )
            suggestions.append(suggestion)

        return suggestions

    def _generate_dataplex_gcloud(
        self, lake_name: str, resource_name: str, resource_type: str
    ) -> str:
        """Generate gcloud command for Dataplex asset."""
        zone = lake_name.replace("/lakes/", "/zones/raw-zone")  # Simplified

        if resource_type == "STORAGE_BUCKET":
            return f"""gcloud dataplex assets create asset-{resource_name} \\
    --lake={lake_name.split('/')[-1]} \\
    --zone=raw-zone \\
    --location={self.location} \\
    --resource-type=STORAGE_BUCKET \\
    --resource-name=projects/{self.project_id}/buckets/{resource_name} \\
    --discovery-enabled"""
        else:
            return f"""gcloud dataplex assets create asset-{resource_name} \\
    --lake={lake_name.split('/')[-1]} \\
    --zone=curated-zone \\
    --location={self.location} \\
    --resource-type=BIGQUERY_DATASET \\
    --resource-name=projects/{self.project_id}/datasets/{resource_name} \\
    --discovery-enabled"""

    def _generate_dataplex_terraform(
        self, lake_id: str, resource_name: str, resource_type: str
    ) -> str:
        """Generate Terraform for Dataplex asset."""
        return f'''resource "google_dataplex_asset" "asset_{resource_name.replace("-", "_")}" {{
  lake     = google_dataplex_lake.{lake_id}.name
  zone     = google_dataplex_zone.raw_zone.name
  location = "{self.location}"

  name         = "asset-{resource_name}"
  display_name = "Asset {resource_name}"

  resource_spec {{
    type = "{resource_type}"
    name = "projects/{self.project_id}/buckets/{resource_name}"
  }}

  discovery_spec {{
    enabled = true
  }}
}}'''

    def _generate_recommendations(
        self, result: GCPDiscoveryEnhancements
    ) -> list[str]:
        """Generate GCP-specific recommendations."""
        recommendations = []

        # Dataplex recommendations
        if not result.dataplex_lakes:
            if result.buckets or result.bigquery_datasets:
                recommendations.append(
                    "No Dataplex lakes found. Consider setting up Dataplex for "
                    "centralized data governance and discovery across GCS and BigQuery."
                )
        elif result.asset_suggestions:
            recommendations.append(
                f"Found {len(result.asset_suggestions)} data resources not managed by Dataplex. "
                "Consider adding them as assets for unified governance."
            )

        # BigQuery recommendations
        tables_without_partition = [
            t for t in result.bigquery_tables
            if t.table_type == "TABLE" and not t.partition_field and t.num_bytes > 1_000_000_000
        ]
        if tables_without_partition:
            recommendations.append(
                f"Found {len(tables_without_partition)} large BigQuery tables without partitioning. "
                "Consider adding time partitioning for better query performance and cost."
            )

        tables_without_policy_tags = [
            t for t in result.bigquery_tables
            if not t.has_policy_tags and t.num_rows > 0
        ]
        if tables_without_policy_tags and len(tables_without_policy_tags) > 5:
            recommendations.append(
                "Most BigQuery tables don't have policy tags. Consider using "
                "Data Catalog policy tags for column-level security."
            )

        # GCS recommendations
        buckets_with_data = [b for b in result.buckets if b.data_formats]
        if buckets_with_data:
            formats_found = set()
            for bucket in buckets_with_data:
                formats_found.update(bucket.data_formats)

            if "parquet" not in formats_found and "delta" not in formats_found:
                recommendations.append(
                    "Consider using columnar formats (Parquet, Delta) for analytics data. "
                    "They provide better compression and query performance."
                )

        return recommendations


def print_gcp_enhancements(result: GCPDiscoveryEnhancements) -> str:
    """Generate printable output for GCP enhancements."""
    lines = []

    lines.append("")
    lines.append("=" * 70)
    lines.append("  GCP ENHANCED DISCOVERY")
    lines.append(f"  Project: {result.project_id}")
    lines.append("=" * 70)

    # GCS Buckets
    lines.append("")
    lines.append("  CLOUD STORAGE BUCKETS")
    lines.append("  " + "-" * 40)

    if result.buckets:
        lines.append(f"    Total: {len(result.buckets)}")
        lines.append("")

        for bucket in result.buckets[:5]:
            formats = ", ".join(bucket.data_formats) if bucket.data_formats else "unknown"
            lines.append(f"    {bucket.name}")
            lines.append(f"      Location: {bucket.location}")
            lines.append(f"      Storage Class: {bucket.storage_class}")
            lines.append(f"      Data formats: {formats}")
            if bucket.uncatalogued_prefixes:
                lines.append(f"      Prefixes: {len(bucket.uncatalogued_prefixes)}")
    else:
        lines.append("    No buckets found")

    # BigQuery
    lines.append("")
    lines.append("  BIGQUERY")
    lines.append("  " + "-" * 40)

    if result.bigquery_datasets:
        total_tables = sum(d.table_count for d in result.bigquery_datasets)
        total_bytes = sum(d.total_bytes for d in result.bigquery_datasets)
        total_gb = total_bytes / (1024**3)

        lines.append(f"    Datasets: {len(result.bigquery_datasets)}")
        lines.append(f"    Total tables: {total_tables}")
        lines.append(f"    Total size: {total_gb:.2f} GB")
        lines.append("")

        for dataset in result.bigquery_datasets[:5]:
            size_gb = dataset.total_bytes / (1024**3)
            lines.append(f"    {dataset.dataset_id}")
            lines.append(f"      Tables: {len(dataset.tables)}, Views: {len(dataset.views)}")
            lines.append(f"      Size: {size_gb:.2f} GB")
            lines.append(f"      Location: {dataset.location}")
    else:
        lines.append("    No BigQuery datasets found")

    # Dataplex
    lines.append("")
    lines.append("  DATAPLEX")
    lines.append("  " + "-" * 40)

    if result.dataplex_lakes:
        for lake in result.dataplex_lakes:
            lines.append(f"    Lake: {lake.display_name}")
            lines.append(f"      State: {lake.state}")
            lines.append(f"      Zones: {len(lake.zones)}")
            lines.append(f"      Assets: {len(lake.assets)}")
    else:
        lines.append("    No Dataplex lakes found")

    # Data Catalog
    lines.append("")
    lines.append("  DATA CATALOG")
    lines.append("  " + "-" * 40)

    if result.datacatalog_entries:
        lines.append(f"    Total entries: {len(result.datacatalog_entries)}")

        # Group by type
        type_counts = {}
        for entry in result.datacatalog_entries:
            type_counts[entry.entry_type] = type_counts.get(entry.entry_type, 0) + 1

        for entry_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"      - {entry_type}: {count}")
    else:
        lines.append("    No Data Catalog entries found")

    # Asset Suggestions
    if result.asset_suggestions:
        lines.append("")
        lines.append("  DATAPLEX ASSET SUGGESTIONS")
        lines.append("  " + "-" * 40)

        for suggestion in result.asset_suggestions[:5]:
            lines.append(f"    {suggestion.name}")
            lines.append(f"      Type: {suggestion.resource_type}")
            lines.append(f"      Reason: {suggestion.reason}")
            lines.append("")
            lines.append("      Quick create:")
            for line in suggestion.gcloud_command.split("\n"):
                lines.append(f"        {line}")
            lines.append("")

    # Recommendations
    if result.recommendations:
        lines.append("")
        lines.append("  GCP RECOMMENDATIONS")
        lines.append("  " + "-" * 40)
        for rec in result.recommendations:
            words = rec.split()
            current_line = "    - "
            for word in words:
                if len(current_line) + len(word) > 68:
                    lines.append(current_line)
                    current_line = "      " + word + " "
                else:
                    current_line += word + " "
            lines.append(current_line.rstrip())

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)
