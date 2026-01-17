"""
DATAMESH.AI Partitioning Strategy Detection and Recommendations.

This module analyzes data sources and suggests optimal partitioning strategies
based on data patterns, query patterns, and cloud-specific best practices.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Set
from datetime import datetime


class PartitionType(Enum):
    """Types of partitioning strategies."""
    NONE = "none"
    DATE = "date"
    DATE_HOUR = "date_hour"
    YEAR_MONTH_DAY = "year_month_day"
    YEAR_MONTH = "year_month"
    HASH = "hash"
    RANGE = "range"
    LIST = "list"
    COMPOSITE = "composite"
    HIVE_STYLE = "hive_style"


class PartitionGranularity(Enum):
    """Granularity for time-based partitions."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class DetectedPartition:
    """A detected partition in the data."""
    column_name: str
    partition_type: PartitionType
    granularity: Optional[PartitionGranularity] = None
    sample_values: List[str] = field(default_factory=list)
    cardinality: Optional[int] = None
    is_hive_style: bool = False
    path_pattern: Optional[str] = None


@dataclass
class PartitionRecommendation:
    """A recommended partitioning strategy."""
    partition_type: PartitionType
    columns: List[str]
    granularity: Optional[PartitionGranularity] = None
    reason: str = ""
    estimated_partition_count: Optional[int] = None
    cloud_specific: Dict[str, str] = field(default_factory=dict)
    ddl_snippet: Optional[str] = None
    crawler_config: Optional[Dict[str, Any]] = None


@dataclass
class PartitionAnalysis:
    """Complete partition analysis for a data source."""
    source_name: str
    source_type: str  # s3, gcs, bigquery, etc.
    location: str
    detected_partitions: List[DetectedPartition] = field(default_factory=list)
    recommendations: List[PartitionRecommendation] = field(default_factory=list)
    current_strategy: Optional[PartitionType] = None
    estimated_data_size_gb: Optional[float] = None
    estimated_row_count: Optional[int] = None
    query_patterns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PartitionDetector:
    """Detects and analyzes partitioning in data sources."""

    # Common date/time patterns in paths
    DATE_PATTERNS = [
        # Hive-style: year=2024/month=01/day=15
        (r"year=(\d{4})/month=(\d{2})/day=(\d{2})", PartitionType.HIVE_STYLE, PartitionGranularity.DAILY),
        (r"year=(\d{4})/month=(\d{2})", PartitionType.HIVE_STYLE, PartitionGranularity.MONTHLY),
        (r"year=(\d{4})", PartitionType.HIVE_STYLE, PartitionGranularity.YEARLY),
        (r"dt=(\d{4}-\d{2}-\d{2})", PartitionType.HIVE_STYLE, PartitionGranularity.DAILY),
        (r"date=(\d{4}-\d{2}-\d{2})", PartitionType.HIVE_STYLE, PartitionGranularity.DAILY),
        (r"hour=(\d{2})", PartitionType.HIVE_STYLE, PartitionGranularity.HOURLY),

        # Date paths: /2024/01/15/ or /2024-01-15/
        (r"/(\d{4})/(\d{2})/(\d{2})/", PartitionType.YEAR_MONTH_DAY, PartitionGranularity.DAILY),
        (r"/(\d{4})/(\d{2})/", PartitionType.YEAR_MONTH, PartitionGranularity.MONTHLY),
        (r"/(\d{4}-\d{2}-\d{2})/", PartitionType.DATE, PartitionGranularity.DAILY),
        (r"/(\d{4}-\d{2}-\d{2})/(\d{2})/", PartitionType.DATE_HOUR, PartitionGranularity.HOURLY),

        # Compact date: /20240115/
        (r"/(\d{8})/", PartitionType.DATE, PartitionGranularity.DAILY),
        (r"/(\d{8})(\d{2})/", PartitionType.DATE_HOUR, PartitionGranularity.HOURLY),
    ]

    # Common partition column patterns
    PARTITION_COLUMN_PATTERNS = {
        "date": ["date", "dt", "event_date", "created_date", "partition_date", "ingestion_date"],
        "time": ["timestamp", "event_time", "created_at", "updated_at", "ts"],
        "region": ["region", "country", "geo", "location", "country_code"],
        "category": ["category", "type", "status", "source", "channel"],
    }

    def __init__(self):
        self.analysis_cache: Dict[str, PartitionAnalysis] = {}

    def detect_from_path(self, path: str) -> List[DetectedPartition]:
        """
        Detect partitioning from a file/object path.

        Args:
            path: S3, GCS, or file path

        Returns:
            List of detected partitions
        """
        detected = []

        for pattern, partition_type, granularity in self.DATE_PATTERNS:
            matches = re.findall(pattern, path, re.IGNORECASE)
            if matches:
                # Determine column name from pattern
                if "year=" in pattern.lower():
                    col_name = "year/month/day" if "day=" in pattern else "year/month" if "month=" in pattern else "year"
                elif "dt=" in pattern.lower() or "date=" in pattern.lower():
                    col_name = "dt" if "dt=" in pattern else "date"
                elif "hour=" in pattern.lower():
                    col_name = "hour"
                else:
                    col_name = "date_partition"

                detected.append(DetectedPartition(
                    column_name=col_name,
                    partition_type=partition_type,
                    granularity=granularity,
                    sample_values=[str(m) if isinstance(m, str) else "/".join(m) for m in matches[:5]],
                    is_hive_style="=" in pattern,
                    path_pattern=pattern,
                ))
                break  # Use first match

        return detected

    def detect_from_paths(self, paths: List[str]) -> List[DetectedPartition]:
        """
        Detect partitioning from multiple paths.

        Args:
            paths: List of file/object paths

        Returns:
            List of detected partitions with cardinality
        """
        partition_values: Dict[str, Set[str]] = {}
        partition_info: Dict[str, DetectedPartition] = {}

        for path in paths:
            detected = self.detect_from_path(path)
            for d in detected:
                key = f"{d.column_name}:{d.partition_type.value}"
                if key not in partition_info:
                    partition_info[key] = d
                    partition_values[key] = set()
                partition_values[key].update(d.sample_values)

        # Update cardinality
        result = []
        for key, partition in partition_info.items():
            partition.cardinality = len(partition_values[key])
            partition.sample_values = list(partition_values[key])[:10]
            result.append(partition)

        return result

    def detect_from_schema(
        self,
        columns: List[Dict[str, Any]],
        table_name: Optional[str] = None,
    ) -> List[DetectedPartition]:
        """
        Detect potential partition columns from schema.

        Args:
            columns: List of column definitions with 'name' and 'type'
            table_name: Optional table name for context

        Returns:
            List of potential partition columns
        """
        detected = []

        for col in columns:
            col_name = col.get("name", "").lower()
            col_type = col.get("type", "").lower()

            # Check date/timestamp columns
            if any(t in col_type for t in ["date", "timestamp", "datetime"]):
                detected.append(DetectedPartition(
                    column_name=col["name"],
                    partition_type=PartitionType.DATE,
                    granularity=PartitionGranularity.DAILY,
                ))

            # Check for common partition column names
            for category, patterns in self.PARTITION_COLUMN_PATTERNS.items():
                if any(p in col_name for p in patterns):
                    if category == "date":
                        detected.append(DetectedPartition(
                            column_name=col["name"],
                            partition_type=PartitionType.DATE,
                            granularity=PartitionGranularity.DAILY,
                        ))
                    elif category == "time":
                        detected.append(DetectedPartition(
                            column_name=col["name"],
                            partition_type=PartitionType.DATE_HOUR,
                            granularity=PartitionGranularity.HOURLY,
                        ))
                    elif category in ["region", "category"]:
                        detected.append(DetectedPartition(
                            column_name=col["name"],
                            partition_type=PartitionType.LIST,
                        ))

        return detected

    def recommend_strategy(
        self,
        detected_partitions: List[DetectedPartition],
        estimated_size_gb: Optional[float] = None,
        query_patterns: Optional[List[str]] = None,
        cloud_provider: str = "aws",
    ) -> List[PartitionRecommendation]:
        """
        Recommend partitioning strategies based on detected patterns.

        Args:
            detected_partitions: List of detected partitions
            estimated_size_gb: Estimated data size in GB
            query_patterns: Common query patterns (e.g., "WHERE date = ?")
            cloud_provider: Target cloud provider

        Returns:
            List of recommendations
        """
        recommendations = []

        # If we have date partitions detected
        date_partitions = [p for p in detected_partitions
                          if p.partition_type in [PartitionType.DATE, PartitionType.HIVE_STYLE,
                                                   PartitionType.YEAR_MONTH_DAY, PartitionType.DATE_HOUR]]

        if date_partitions:
            # Recommend based on granularity and size
            best_granularity = self._determine_best_granularity(
                date_partitions, estimated_size_gb
            )

            rec = PartitionRecommendation(
                partition_type=PartitionType.HIVE_STYLE,
                columns=[p.column_name for p in date_partitions],
                granularity=best_granularity,
                reason=self._get_granularity_reason(best_granularity, estimated_size_gb),
            )

            # Add cloud-specific config
            rec.cloud_specific = self._get_cloud_config(
                cloud_provider, rec, date_partitions
            )
            rec.ddl_snippet = self._generate_ddl(cloud_provider, rec, date_partitions)
            rec.crawler_config = self._generate_crawler_config(cloud_provider, rec)

            recommendations.append(rec)

        # Check for categorical partitions
        categorical_partitions = [p for p in detected_partitions
                                  if p.partition_type == PartitionType.LIST]

        if categorical_partitions:
            for cat_partition in categorical_partitions:
                if cat_partition.cardinality and cat_partition.cardinality < 100:
                    rec = PartitionRecommendation(
                        partition_type=PartitionType.LIST,
                        columns=[cat_partition.column_name],
                        reason=f"Low cardinality column ({cat_partition.cardinality} values) suitable for list partitioning",
                    )
                    rec.cloud_specific = self._get_cloud_config(
                        cloud_provider, rec, [cat_partition]
                    )
                    recommendations.append(rec)

        # Composite recommendation if both date and categorical
        if date_partitions and categorical_partitions:
            rec = PartitionRecommendation(
                partition_type=PartitionType.COMPOSITE,
                columns=[date_partitions[0].column_name, categorical_partitions[0].column_name],
                granularity=self._determine_best_granularity(date_partitions, estimated_size_gb),
                reason="Composite partitioning on date + category for optimal query performance",
            )
            rec.cloud_specific = self._get_cloud_config(cloud_provider, rec, date_partitions)
            recommendations.append(rec)

        return recommendations

    def _determine_best_granularity(
        self,
        partitions: List[DetectedPartition],
        size_gb: Optional[float],
    ) -> PartitionGranularity:
        """Determine the best partition granularity based on data characteristics."""
        # Check if any partition already has granularity
        existing = [p.granularity for p in partitions if p.granularity]
        if existing:
            return existing[0]

        # Based on size
        if size_gb:
            if size_gb > 1000:  # > 1TB
                return PartitionGranularity.HOURLY
            elif size_gb > 100:  # > 100GB
                return PartitionGranularity.DAILY
            elif size_gb > 10:  # > 10GB
                return PartitionGranularity.DAILY
            else:
                return PartitionGranularity.MONTHLY

        return PartitionGranularity.DAILY  # Default

    def _get_granularity_reason(
        self,
        granularity: PartitionGranularity,
        size_gb: Optional[float],
    ) -> str:
        """Get explanation for recommended granularity."""
        reasons = {
            PartitionGranularity.HOURLY: "High data volume benefits from hourly partitions for efficient pruning",
            PartitionGranularity.DAILY: "Daily partitioning balances query performance with partition management",
            PartitionGranularity.WEEKLY: "Weekly partitioning suitable for moderate data volumes",
            PartitionGranularity.MONTHLY: "Monthly partitioning optimal for smaller datasets or long retention",
            PartitionGranularity.YEARLY: "Yearly partitioning for archival or reference data",
        }
        base_reason = reasons.get(granularity, "")
        if size_gb:
            base_reason += f" (estimated {size_gb:.1f} GB)"
        return base_reason

    def _get_cloud_config(
        self,
        provider: str,
        recommendation: PartitionRecommendation,
        partitions: List[DetectedPartition],
    ) -> Dict[str, str]:
        """Get cloud-specific configuration."""
        config = {}

        if provider == "aws":
            config["glue_table_type"] = "EXTERNAL_TABLE"
            config["serde"] = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
            if recommendation.partition_type == PartitionType.HIVE_STYLE:
                config["partition_projection"] = "enabled"
                config["projection_type"] = "date" if recommendation.granularity else "enum"

        elif provider == "azure":
            config["synapse_distribution"] = "HASH" if recommendation.partition_type == PartitionType.HASH else "ROUND_ROBIN"
            config["partition_function"] = "RANGE"

        elif provider == "gcp":
            config["bigquery_partition_type"] = "TIME" if recommendation.granularity else "RANGE"
            if recommendation.granularity:
                granularity_map = {
                    PartitionGranularity.HOURLY: "HOUR",
                    PartitionGranularity.DAILY: "DAY",
                    PartitionGranularity.MONTHLY: "MONTH",
                    PartitionGranularity.YEARLY: "YEAR",
                }
                config["bigquery_granularity"] = granularity_map.get(
                    recommendation.granularity, "DAY"
                )

        elif provider == "snowflake":
            config["cluster_by"] = ", ".join(recommendation.columns)
            if recommendation.granularity == PartitionGranularity.DAILY:
                config["cluster_expression"] = f"TO_DATE({recommendation.columns[0]})"

        return config

    def _generate_ddl(
        self,
        provider: str,
        recommendation: PartitionRecommendation,
        partitions: List[DetectedPartition],
    ) -> str:
        """Generate DDL snippet for the recommendation."""
        cols = ", ".join(recommendation.columns)

        if provider == "aws":
            return f"""-- AWS Glue/Athena DDL
PARTITIONED BY ({cols})
STORED AS PARQUET
LOCATION 's3://bucket/prefix/'
TBLPROPERTIES (
  'projection.enabled' = 'true',
  'projection.{recommendation.columns[0]}.type' = 'date',
  'projection.{recommendation.columns[0]}.format' = 'yyyy-MM-dd',
  'projection.{recommendation.columns[0]}.range' = '2020-01-01,NOW'
);"""

        elif provider == "gcp":
            gran = recommendation.cloud_specific.get("bigquery_granularity", "DAY")
            return f"""-- BigQuery DDL
CREATE TABLE dataset.table
PARTITION BY {gran}({recommendation.columns[0]})
CLUSTER BY {cols}
AS SELECT * FROM source_table;"""

        elif provider == "azure":
            return f"""-- Azure Synapse DDL
CREATE TABLE schema.table
WITH (
  DISTRIBUTION = HASH({recommendation.columns[0]}),
  PARTITION ({recommendation.columns[0]} RANGE RIGHT FOR VALUES ())
);"""

        elif provider == "snowflake":
            return f"""-- Snowflake DDL
CREATE TABLE schema.table
CLUSTER BY ({cols});

-- Or with date expression
ALTER TABLE schema.table CLUSTER BY (TO_DATE({recommendation.columns[0]}));"""

        return ""

    def _generate_crawler_config(
        self,
        provider: str,
        recommendation: PartitionRecommendation,
    ) -> Dict[str, Any]:
        """Generate crawler configuration for the recommendation."""
        if provider == "aws":
            return {
                "RecrawlPolicy": {"RecrawlBehavior": "CRAWL_NEW_FOLDERS_ONLY"},
                "SchemaChangePolicy": {
                    "UpdateBehavior": "UPDATE_IN_DATABASE",
                    "DeleteBehavior": "LOG",
                },
                "Configuration": {
                    "Version": 1.0,
                    "Grouping": {"TableGroupingPolicy": "CombineCompatibleSchemas"},
                },
            }
        return {}


class PartitionAnalyzer:
    """High-level partition analysis for data sources."""

    def __init__(self):
        self.detector = PartitionDetector()

    async def analyze_s3_bucket(
        self,
        bucket: str,
        prefix: str = "",
        region: str = "us-east-1",
        sample_size: int = 1000,
    ) -> PartitionAnalysis:
        """
        Analyze partitioning in an S3 bucket.

        Args:
            bucket: S3 bucket name
            prefix: Optional prefix to analyze
            region: AWS region
            sample_size: Number of objects to sample

        Returns:
            Partition analysis
        """
        import boto3

        s3 = boto3.client("s3", region_name=region)

        # List objects
        paginator = s3.get_paginator("list_objects_v2")
        paths = []
        total_size = 0

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, MaxKeys=sample_size):
            for obj in page.get("Contents", []):
                paths.append(obj["Key"])
                total_size += obj.get("Size", 0)
                if len(paths) >= sample_size:
                    break
            if len(paths) >= sample_size:
                break

        # Detect partitions from paths
        detected = self.detector.detect_from_paths(paths)

        # Generate recommendations
        estimated_size_gb = total_size / (1024 ** 3) if total_size else None
        recommendations = self.detector.recommend_strategy(
            detected, estimated_size_gb, cloud_provider="aws"
        )

        return PartitionAnalysis(
            source_name=bucket,
            source_type="s3",
            location=f"s3://{bucket}/{prefix}",
            detected_partitions=detected,
            recommendations=recommendations,
            estimated_data_size_gb=estimated_size_gb,
        )

    async def analyze_gcs_bucket(
        self,
        bucket: str,
        prefix: str = "",
        project: str = "",
        sample_size: int = 1000,
    ) -> PartitionAnalysis:
        """
        Analyze partitioning in a GCS bucket.

        Args:
            bucket: GCS bucket name
            prefix: Optional prefix to analyze
            project: GCP project ID
            sample_size: Number of objects to sample

        Returns:
            Partition analysis
        """
        try:
            from google.cloud import storage
        except ImportError:
            return PartitionAnalysis(
                source_name=bucket,
                source_type="gcs",
                location=f"gs://{bucket}/{prefix}",
                warnings=["google-cloud-storage not installed"],
            )

        client = storage.Client(project=project) if project else storage.Client()
        bucket_obj = client.bucket(bucket)

        paths = []
        total_size = 0

        for blob in bucket_obj.list_blobs(prefix=prefix, max_results=sample_size):
            paths.append(blob.name)
            total_size += blob.size or 0

        detected = self.detector.detect_from_paths(paths)
        estimated_size_gb = total_size / (1024 ** 3) if total_size else None
        recommendations = self.detector.recommend_strategy(
            detected, estimated_size_gb, cloud_provider="gcp"
        )

        return PartitionAnalysis(
            source_name=bucket,
            source_type="gcs",
            location=f"gs://{bucket}/{prefix}",
            detected_partitions=detected,
            recommendations=recommendations,
            estimated_data_size_gb=estimated_size_gb,
        )

    async def analyze_glue_table(
        self,
        database: str,
        table: str,
        region: str = "us-east-1",
    ) -> PartitionAnalysis:
        """
        Analyze partitioning of a Glue table.

        Args:
            database: Glue database name
            table: Table name
            region: AWS region

        Returns:
            Partition analysis
        """
        import boto3

        glue = boto3.client("glue", region_name=region)

        try:
            response = glue.get_table(DatabaseName=database, Name=table)
            table_info = response["Table"]

            # Get existing partitions
            partition_keys = table_info.get("PartitionKeys", [])
            storage_location = table_info.get("StorageDescriptor", {}).get("Location", "")
            columns = table_info.get("StorageDescriptor", {}).get("Columns", [])

            # Detect from schema
            schema_detected = self.detector.detect_from_schema(
                [{"name": c["Name"], "type": c["Type"]} for c in columns],
                table_name=table,
            )

            # Current strategy
            current_strategy = PartitionType.HIVE_STYLE if partition_keys else PartitionType.NONE

            detected = []
            for pk in partition_keys:
                detected.append(DetectedPartition(
                    column_name=pk["Name"],
                    partition_type=PartitionType.HIVE_STYLE,
                    is_hive_style=True,
                ))

            # Get partition count for cardinality
            try:
                partitions_response = glue.get_partitions(
                    DatabaseName=database, TableName=table, MaxResults=1000
                )
                partition_count = len(partitions_response.get("Partitions", []))
                if detected:
                    detected[0].cardinality = partition_count
            except Exception:
                partition_count = None

            recommendations = self.detector.recommend_strategy(
                detected + schema_detected,
                cloud_provider="aws",
            )

            return PartitionAnalysis(
                source_name=f"{database}.{table}",
                source_type="glue_table",
                location=storage_location,
                detected_partitions=detected,
                recommendations=recommendations,
                current_strategy=current_strategy,
            )

        except Exception as e:
            return PartitionAnalysis(
                source_name=f"{database}.{table}",
                source_type="glue_table",
                location="",
                warnings=[str(e)],
            )

    async def analyze_bigquery_table(
        self,
        project: str,
        dataset: str,
        table: str,
    ) -> PartitionAnalysis:
        """
        Analyze partitioning of a BigQuery table.

        Args:
            project: GCP project ID
            dataset: BigQuery dataset
            table: Table name

        Returns:
            Partition analysis
        """
        try:
            from google.cloud import bigquery
        except ImportError:
            return PartitionAnalysis(
                source_name=f"{project}.{dataset}.{table}",
                source_type="bigquery",
                location=f"bigquery://{project}/{dataset}/{table}",
                warnings=["google-cloud-bigquery not installed"],
            )

        client = bigquery.Client(project=project)
        table_ref = f"{project}.{dataset}.{table}"

        try:
            table_info = client.get_table(table_ref)

            detected = []
            current_strategy = PartitionType.NONE

            # Check time partitioning
            if table_info.time_partitioning:
                tp = table_info.time_partitioning
                granularity_map = {
                    "HOUR": PartitionGranularity.HOURLY,
                    "DAY": PartitionGranularity.DAILY,
                    "MONTH": PartitionGranularity.MONTHLY,
                    "YEAR": PartitionGranularity.YEARLY,
                }
                detected.append(DetectedPartition(
                    column_name=tp.field or "_PARTITIONTIME",
                    partition_type=PartitionType.DATE,
                    granularity=granularity_map.get(tp.type_, PartitionGranularity.DAILY),
                ))
                current_strategy = PartitionType.DATE

            # Check range partitioning
            if table_info.range_partitioning:
                rp = table_info.range_partitioning
                detected.append(DetectedPartition(
                    column_name=rp.field,
                    partition_type=PartitionType.RANGE,
                ))
                current_strategy = PartitionType.RANGE

            # Check clustering
            if table_info.clustering_fields:
                for field in table_info.clustering_fields:
                    detected.append(DetectedPartition(
                        column_name=field,
                        partition_type=PartitionType.HASH,
                    ))

            # Detect from schema
            schema_columns = [
                {"name": f.name, "type": f.field_type}
                for f in table_info.schema
            ]
            schema_detected = self.detector.detect_from_schema(schema_columns, table)

            # Estimate size
            size_gb = table_info.num_bytes / (1024 ** 3) if table_info.num_bytes else None

            recommendations = self.detector.recommend_strategy(
                detected + schema_detected,
                estimated_size_gb=size_gb,
                cloud_provider="gcp",
            )

            return PartitionAnalysis(
                source_name=table_ref,
                source_type="bigquery",
                location=f"bigquery://{project}/{dataset}/{table}",
                detected_partitions=detected,
                recommendations=recommendations,
                current_strategy=current_strategy,
                estimated_data_size_gb=size_gb,
                estimated_row_count=table_info.num_rows,
            )

        except Exception as e:
            return PartitionAnalysis(
                source_name=table_ref,
                source_type="bigquery",
                location=f"bigquery://{project}/{dataset}/{table}",
                warnings=[str(e)],
            )


def print_partition_analysis(analysis: PartitionAnalysis) -> str:
    """Print partition analysis in a readable format."""
    lines = [
        "",
        "=" * 70,
        f"  PARTITION ANALYSIS: {analysis.source_name}",
        "=" * 70,
        f"  Source Type: {analysis.source_type}",
        f"  Location: {analysis.location}",
    ]

    if analysis.estimated_data_size_gb:
        lines.append(f"  Estimated Size: {analysis.estimated_data_size_gb:.2f} GB")

    if analysis.estimated_row_count:
        lines.append(f"  Estimated Rows: {analysis.estimated_row_count:,}")

    if analysis.current_strategy:
        lines.append(f"  Current Strategy: {analysis.current_strategy.value}")

    # Detected partitions
    if analysis.detected_partitions:
        lines.append("\n  DETECTED PARTITIONS:")
        for dp in analysis.detected_partitions:
            lines.append(f"    - {dp.column_name}")
            lines.append(f"      Type: {dp.partition_type.value}")
            if dp.granularity:
                lines.append(f"      Granularity: {dp.granularity.value}")
            if dp.cardinality:
                lines.append(f"      Cardinality: {dp.cardinality}")
            if dp.sample_values:
                lines.append(f"      Samples: {', '.join(dp.sample_values[:3])}")

    # Recommendations
    if analysis.recommendations:
        lines.append("\n  RECOMMENDATIONS:")
        for i, rec in enumerate(analysis.recommendations, 1):
            lines.append(f"\n    {i}. {rec.partition_type.value.upper()} on ({', '.join(rec.columns)})")
            if rec.granularity:
                lines.append(f"       Granularity: {rec.granularity.value}")
            lines.append(f"       Reason: {rec.reason}")

            if rec.cloud_specific:
                lines.append("       Cloud Config:")
                for k, v in rec.cloud_specific.items():
                    lines.append(f"         {k}: {v}")

            if rec.ddl_snippet:
                lines.append("       DDL:")
                for ddl_line in rec.ddl_snippet.split("\n")[:5]:
                    lines.append(f"         {ddl_line}")

    # Warnings
    if analysis.warnings:
        lines.append("\n  WARNINGS:")
        for w in analysis.warnings:
            lines.append(f"    ⚠ {w}")

    lines.append("")
    return "\n".join(lines)
