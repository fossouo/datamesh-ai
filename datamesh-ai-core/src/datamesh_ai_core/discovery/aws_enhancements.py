"""
AWS Enhancements - Additional AWS-specific discovery features.

Includes:
- S3 bucket discovery and analysis
- Glue Crawler management suggestions
- Athena workgroup detection
- Lake Formation status
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredS3Bucket:
    """Discovered S3 bucket with cataloguing potential."""
    name: str
    region: str
    creation_date: Optional[datetime] = None
    is_catalogued: bool = False
    catalogued_prefixes: list[str] = field(default_factory=list)
    uncatalogued_prefixes: list[str] = field(default_factory=list)
    total_size_bytes: Optional[int] = None
    object_count: Optional[int] = None
    data_formats: list[str] = field(default_factory=list)
    suggested_crawler_name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveredGlueCrawler:
    """Discovered Glue Crawler."""
    name: str
    database: str
    targets: list[str] = field(default_factory=list)
    state: str = "UNKNOWN"
    schedule: Optional[str] = None
    last_run: Optional[datetime] = None
    tables_created: int = 0
    tables_updated: int = 0


@dataclass
class CrawlerSuggestion:
    """Suggested Glue Crawler configuration."""
    name: str
    database: str
    s3_targets: list[str]
    description: str
    reason: str
    estimated_tables: int = 0
    cli_command: str = ""
    terraform_snippet: str = ""


@dataclass
class AWSDiscoveryEnhancements:
    """Enhanced AWS discovery results."""
    buckets: list[DiscoveredS3Bucket] = field(default_factory=list)
    crawlers: list[DiscoveredGlueCrawler] = field(default_factory=list)
    crawler_suggestions: list[CrawlerSuggestion] = field(default_factory=list)
    athena_workgroups: list[str] = field(default_factory=list)
    lake_formation_enabled: bool = False
    recommendations: list[str] = field(default_factory=list)


class AWSEnhancedDiscovery:
    """
    Enhanced AWS discovery with S3, Glue Crawler, and cataloguing suggestions.
    """

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._s3_client = None
        self._glue_client = None
        self._athena_client = None
        self._lf_client = None

    async def discover(self) -> AWSDiscoveryEnhancements:
        """Run enhanced AWS discovery."""
        import boto3

        session = boto3.Session()
        self._s3_client = session.client("s3", region_name=self.region)
        self._glue_client = session.client("glue", region_name=self.region)

        try:
            self._athena_client = session.client("athena", region_name=self.region)
        except Exception:
            pass

        try:
            self._lf_client = session.client("lakeformation", region_name=self.region)
        except Exception:
            pass

        result = AWSDiscoveryEnhancements()

        # Discover S3 buckets
        result.buckets = await self._discover_s3_buckets()

        # Discover existing crawlers
        result.crawlers = await self._discover_crawlers()

        # Check Lake Formation
        result.lake_formation_enabled = await self._check_lake_formation()

        # Discover Athena workgroups
        result.athena_workgroups = await self._discover_athena_workgroups()

        # Generate crawler suggestions
        result.crawler_suggestions = self._generate_crawler_suggestions(
            result.buckets, result.crawlers
        )

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    async def _discover_s3_buckets(self) -> list[DiscoveredS3Bucket]:
        """Discover S3 buckets and analyze their contents."""
        buckets = []

        try:
            response = self._s3_client.list_buckets()

            # Get catalogued locations from Glue
            catalogued_locations = set()
            try:
                paginator = self._glue_client.get_paginator("get_tables")
                for db in self._glue_client.get_databases().get("DatabaseList", []):
                    for page in paginator.paginate(DatabaseName=db["Name"]):
                        for table in page.get("TableList", []):
                            location = table.get("StorageDescriptor", {}).get("Location", "")
                            if location.startswith("s3://"):
                                # Extract bucket and prefix
                                parts = location[5:].split("/", 1)
                                bucket_name = parts[0]
                                prefix = parts[1] if len(parts) > 1 else ""
                                catalogued_locations.add((bucket_name, prefix))
            except Exception as e:
                logger.warning(f"Could not get catalogued locations: {e}")

            for bucket_info in response.get("Buckets", []):
                bucket_name = bucket_info["Name"]

                # Get bucket region
                try:
                    location = self._s3_client.get_bucket_location(Bucket=bucket_name)
                    bucket_region = location.get("LocationConstraint") or "us-east-1"
                except Exception:
                    bucket_region = "unknown"

                # Skip buckets in different regions for performance
                if bucket_region != self.region and bucket_region != "unknown":
                    continue

                bucket = DiscoveredS3Bucket(
                    name=bucket_name,
                    region=bucket_region,
                    creation_date=bucket_info.get("CreationDate"),
                )

                # Analyze bucket contents (sample prefixes)
                try:
                    prefixes = await self._analyze_bucket_prefixes(bucket_name)
                    bucket.data_formats = prefixes.get("formats", [])

                    # Check which prefixes are catalogued
                    for prefix in prefixes.get("prefixes", []):
                        is_catalogued = any(
                            loc[0] == bucket_name and prefix.startswith(loc[1])
                            for loc in catalogued_locations
                        )
                        if is_catalogued:
                            bucket.catalogued_prefixes.append(prefix)
                        else:
                            bucket.uncatalogued_prefixes.append(prefix)

                    bucket.is_catalogued = len(bucket.catalogued_prefixes) > 0

                except Exception as e:
                    logger.debug(f"Could not analyze bucket {bucket_name}: {e}")

                # Generate suggested crawler name
                if bucket.uncatalogued_prefixes:
                    bucket.suggested_crawler_name = f"crawler-{bucket_name}"

                buckets.append(bucket)

        except Exception as e:
            logger.error(f"Error discovering S3 buckets: {e}")

        return buckets

    async def _analyze_bucket_prefixes(self, bucket_name: str) -> dict:
        """Analyze bucket prefixes and detect data formats."""
        result = {"prefixes": [], "formats": []}

        try:
            # List top-level prefixes
            response = self._s3_client.list_objects_v2(
                Bucket=bucket_name,
                Delimiter="/",
                MaxKeys=100,
            )

            prefixes = [p["Prefix"] for p in response.get("CommonPrefixes", [])]
            result["prefixes"] = prefixes[:20]  # Limit

            # Sample objects to detect formats
            formats = set()
            sample = self._s3_client.list_objects_v2(
                Bucket=bucket_name,
                MaxKeys=50,
            )

            for obj in sample.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".parquet"):
                    formats.add("parquet")
                elif key.endswith(".json") or key.endswith(".jsonl"):
                    formats.add("json")
                elif key.endswith(".csv"):
                    formats.add("csv")
                elif key.endswith(".avro"):
                    formats.add("avro")
                elif key.endswith(".orc"):
                    formats.add("orc")

            result["formats"] = list(formats)

        except Exception as e:
            logger.debug(f"Error analyzing bucket {bucket_name}: {e}")

        return result

    async def _discover_crawlers(self) -> list[DiscoveredGlueCrawler]:
        """Discover existing Glue Crawlers."""
        crawlers = []

        try:
            paginator = self._glue_client.get_paginator("get_crawlers")

            for page in paginator.paginate():
                for crawler in page.get("Crawlers", []):
                    targets = []
                    for s3_target in crawler.get("Targets", {}).get("S3Targets", []):
                        targets.append(s3_target.get("Path", ""))

                    last_crawl = crawler.get("LastCrawl", {})

                    crawlers.append(DiscoveredGlueCrawler(
                        name=crawler["Name"],
                        database=crawler.get("DatabaseName", ""),
                        targets=targets,
                        state=crawler.get("State", "UNKNOWN"),
                        schedule=crawler.get("Schedule"),
                        last_run=last_crawl.get("StartTime"),
                        tables_created=last_crawl.get("TablesCreated", 0),
                        tables_updated=last_crawl.get("TablesUpdated", 0),
                    ))

        except Exception as e:
            logger.warning(f"Error discovering crawlers: {e}")

        return crawlers

    async def _check_lake_formation(self) -> bool:
        """Check if Lake Formation is enabled."""
        if not self._lf_client:
            return False

        try:
            settings = self._lf_client.get_data_lake_settings()
            # If we can get settings, Lake Formation is configured
            return True
        except Exception:
            return False

    async def _discover_athena_workgroups(self) -> list[str]:
        """Discover Athena workgroups."""
        workgroups = []

        if not self._athena_client:
            return workgroups

        try:
            response = self._athena_client.list_work_groups()
            workgroups = [wg["Name"] for wg in response.get("WorkGroups", [])]
        except Exception as e:
            logger.debug(f"Error discovering Athena workgroups: {e}")

        return workgroups

    def _generate_crawler_suggestions(
        self,
        buckets: list[DiscoveredS3Bucket],
        existing_crawlers: list[DiscoveredGlueCrawler],
    ) -> list[CrawlerSuggestion]:
        """Generate crawler suggestions for uncatalogued data."""
        suggestions = []

        # Get existing crawler targets
        existing_targets = set()
        for crawler in existing_crawlers:
            for target in crawler.targets:
                existing_targets.add(target)

        for bucket in buckets:
            if not bucket.uncatalogued_prefixes:
                continue

            # Skip buckets that look like system/log buckets
            if any(x in bucket.name.lower() for x in ["log", "trail", "config", "backup"]):
                continue

            for prefix in bucket.uncatalogued_prefixes[:3]:  # Limit suggestions per bucket
                s3_path = f"s3://{bucket.name}/{prefix}"

                # Skip if already covered by existing crawler
                if any(s3_path.startswith(t) or t.startswith(s3_path) for t in existing_targets):
                    continue

                # Generate database name from bucket/prefix
                db_name = self._generate_database_name(bucket.name, prefix)
                crawler_name = f"crawler-{db_name}"

                # Generate CLI command
                cli_command = self._generate_crawler_cli(
                    crawler_name, db_name, s3_path
                )

                # Generate Terraform snippet
                terraform_snippet = self._generate_crawler_terraform(
                    crawler_name, db_name, s3_path
                )

                suggestions.append(CrawlerSuggestion(
                    name=crawler_name,
                    database=db_name,
                    s3_targets=[s3_path],
                    description=f"Catalog data in {s3_path}",
                    reason=f"Found {', '.join(bucket.data_formats) or 'data'} files not yet in Glue Catalog",
                    cli_command=cli_command,
                    terraform_snippet=terraform_snippet,
                ))

        return suggestions

    def _generate_database_name(self, bucket: str, prefix: str) -> str:
        """Generate a database name from bucket and prefix."""
        # Clean up names
        name = f"{bucket}_{prefix}".rstrip("/")
        name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.lower()[:128]  # Glue limit
        return name

    def _generate_crawler_cli(
        self, crawler_name: str, database: str, s3_path: str
    ) -> str:
        """Generate AWS CLI command to create crawler."""
        return f"""aws glue create-crawler \\
    --name {crawler_name} \\
    --role AWSGlueServiceRole \\
    --database-name {database} \\
    --targets '{{"S3Targets": [{{"Path": "{s3_path}"}}]}}'"""

    def _generate_crawler_terraform(
        self, crawler_name: str, database: str, s3_path: str
    ) -> str:
        """Generate Terraform snippet for crawler."""
        return f'''resource "aws_glue_crawler" "{crawler_name.replace("-", "_")}" {{
  name          = "{crawler_name}"
  database_name = "{database}"
  role          = aws_iam_role.glue_role.arn

  s3_target {{
    path = "{s3_path}"
  }}

  schema_change_policy {{
    delete_behavior = "LOG"
    update_behavior = "UPDATE_IN_DATABASE"
  }}
}}'''

    def _generate_recommendations(
        self, result: AWSDiscoveryEnhancements
    ) -> list[str]:
        """Generate AWS-specific recommendations."""
        recommendations = []

        # Uncatalogued data
        uncatalogued_buckets = [b for b in result.buckets if b.uncatalogued_prefixes]
        if uncatalogued_buckets:
            recommendations.append(
                f"Found {len(uncatalogued_buckets)} S3 bucket(s) with data not in Glue Catalog. "
                "Consider creating Glue Crawlers to make this data queryable."
            )

        # Crawler suggestions
        if result.crawler_suggestions:
            recommendations.append(
                f"Generated {len(result.crawler_suggestions)} crawler suggestion(s) "
                "to catalog your S3 data automatically."
            )

        # Lake Formation
        if not result.lake_formation_enabled:
            recommendations.append(
                "Lake Formation is not enabled. Consider enabling it for "
                "fine-grained access control and governance."
            )

        # Athena workgroups
        if not result.athena_workgroups or result.athena_workgroups == ["primary"]:
            recommendations.append(
                "Consider creating dedicated Athena workgroups for "
                "cost tracking and query management."
            )

        # Stale crawlers
        stale_crawlers = [
            c for c in result.crawlers
            if c.last_run and (datetime.utcnow() - c.last_run).days > 30
        ]
        if stale_crawlers:
            recommendations.append(
                f"Found {len(stale_crawlers)} crawler(s) that haven't run in 30+ days. "
                "Consider scheduling them or removing if no longer needed."
            )

        return recommendations


def print_aws_enhancements(result: AWSDiscoveryEnhancements) -> str:
    """Generate printable output for AWS enhancements."""
    lines = []

    lines.append("")
    lines.append("=" * 70)
    lines.append("  AWS ENHANCED DISCOVERY")
    lines.append("=" * 70)

    # S3 Buckets
    lines.append("")
    lines.append("  S3 BUCKETS (in region)")
    lines.append("  " + "-" * 40)

    catalogued = [b for b in result.buckets if b.is_catalogued]
    uncatalogued = [b for b in result.buckets if b.uncatalogued_prefixes and not b.is_catalogued]

    lines.append(f"    Catalogued: {len(catalogued)}")
    lines.append(f"    With uncatalogued data: {len(uncatalogued)}")

    if uncatalogued:
        lines.append("")
        lines.append("    Buckets with uncatalogued data:")
        for bucket in uncatalogued[:5]:
            formats = ", ".join(bucket.data_formats) if bucket.data_formats else "unknown"
            lines.append(f"      - {bucket.name} ({formats})")
            for prefix in bucket.uncatalogued_prefixes[:2]:
                lines.append(f"        └── {prefix}")

    # Existing Crawlers
    lines.append("")
    lines.append("  GLUE CRAWLERS")
    lines.append("  " + "-" * 40)

    if result.crawlers:
        for crawler in result.crawlers:
            status = "Running" if crawler.state == "RUNNING" else crawler.state
            lines.append(f"    {crawler.name} [{status}]")
            lines.append(f"      Database: {crawler.database}")
            if crawler.last_run:
                lines.append(f"      Last run: {crawler.last_run.strftime('%Y-%m-%d')}")
    else:
        lines.append("    No crawlers found")

    # Crawler Suggestions
    if result.crawler_suggestions:
        lines.append("")
        lines.append("  CRAWLER SUGGESTIONS")
        lines.append("  " + "-" * 40)

        for suggestion in result.crawler_suggestions[:5]:
            lines.append(f"    {suggestion.name}")
            lines.append(f"      Target: {suggestion.s3_targets[0]}")
            lines.append(f"      Reason: {suggestion.reason}")
            lines.append("")
            lines.append("      Quick create:")
            for line in suggestion.cli_command.split("\n"):
                lines.append(f"        {line}")
            lines.append("")

    # Lake Formation & Athena
    lines.append("")
    lines.append("  ADDITIONAL SERVICES")
    lines.append("  " + "-" * 40)
    lines.append(f"    Lake Formation: {'Enabled' if result.lake_formation_enabled else 'Not configured'}")
    lines.append(f"    Athena Workgroups: {', '.join(result.athena_workgroups) if result.athena_workgroups else 'None'}")

    # Recommendations
    if result.recommendations:
        lines.append("")
        lines.append("  AWS RECOMMENDATIONS")
        lines.append("  " + "-" * 40)
        for rec in result.recommendations:
            # Wrap long recommendations
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
