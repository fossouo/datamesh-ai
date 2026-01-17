"""
DATAMESH.AI Discovery CLI - Execute discovery suggestions.

This module provides CLI commands to execute crawler, scan, and asset suggestions
discovered by the enhanced discovery modules.
"""

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime


class ExecutionStatus(Enum):
    """Status of suggestion execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionResult:
    """Result of executing a suggestion."""
    suggestion_id: str
    cloud: str
    resource_type: str
    status: ExecutionStatus
    command: str
    output: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    created_resource_id: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Plan for executing multiple suggestions."""
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    dry_run: bool = True
    parallel: bool = False
    results: List[ExecutionResult] = field(default_factory=list)


class DiscoveryCLI:
    """CLI for executing discovery suggestions."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.execution_log: List[ExecutionResult] = []

    def _log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    # =========================================================================
    # AWS Glue Crawler Execution
    # =========================================================================

    async def create_glue_crawler(
        self,
        name: str,
        s3_path: str,
        database: str,
        role: str,
        region: str = "us-east-1",
        table_prefix: str = "",
        schedule: Optional[str] = None,
        classifiers: Optional[List[str]] = None,
        dry_run: bool = True,
    ) -> ExecutionResult:
        """
        Create an AWS Glue Crawler.

        Args:
            name: Crawler name
            s3_path: S3 path to crawl (s3://bucket/prefix)
            database: Target Glue database
            role: IAM role ARN for the crawler
            region: AWS region
            table_prefix: Prefix for created tables
            schedule: Cron schedule (optional)
            classifiers: Custom classifiers (optional)
            dry_run: If True, only show the command without executing

        Returns:
            ExecutionResult with status and output
        """
        start_time = datetime.now()

        # Build the AWS CLI command
        cmd = [
            "aws", "glue", "create-crawler",
            "--name", name,
            "--role", role,
            "--database-name", database,
            "--targets", json.dumps({
                "S3Targets": [{"Path": s3_path}]
            }),
            "--region", region,
        ]

        if table_prefix:
            cmd.extend(["--table-prefix", table_prefix])

        if schedule:
            cmd.extend(["--schedule", schedule])

        if classifiers:
            cmd.extend(["--classifiers", json.dumps(classifiers)])

        command_str = " ".join(cmd)

        if dry_run:
            self._log(f"\n[DRY RUN] Would execute:\n  {command_str}")
            return ExecutionResult(
                suggestion_id=f"crawler-{name}",
                cloud="aws",
                resource_type="glue_crawler",
                status=ExecutionStatus.SKIPPED,
                command=command_str,
                output="Dry run - command not executed",
                duration_seconds=0.0,
            )

        self._log(f"\n Creating Glue Crawler: {name}")
        self._log(f"  S3 Path: {s3_path}")
        self._log(f"  Database: {database}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            duration = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                self._log(f"  ✓ Crawler created successfully")
                return ExecutionResult(
                    suggestion_id=f"crawler-{name}",
                    cloud="aws",
                    resource_type="glue_crawler",
                    status=ExecutionStatus.SUCCESS,
                    command=command_str,
                    output=result.stdout,
                    created_resource_id=name,
                    duration_seconds=duration,
                )
            else:
                self._log(f"  ✗ Failed: {result.stderr}")
                return ExecutionResult(
                    suggestion_id=f"crawler-{name}",
                    cloud="aws",
                    resource_type="glue_crawler",
                    status=ExecutionStatus.FAILED,
                    command=command_str,
                    error=result.stderr,
                    duration_seconds=duration,
                )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                suggestion_id=f"crawler-{name}",
                cloud="aws",
                resource_type="glue_crawler",
                status=ExecutionStatus.FAILED,
                command=command_str,
                error="Command timed out after 60 seconds",
                duration_seconds=60.0,
            )
        except Exception as e:
            return ExecutionResult(
                suggestion_id=f"crawler-{name}",
                cloud="aws",
                resource_type="glue_crawler",
                status=ExecutionStatus.FAILED,
                command=command_str,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    async def start_glue_crawler(
        self,
        name: str,
        region: str = "us-east-1",
        wait: bool = False,
        dry_run: bool = True,
    ) -> ExecutionResult:
        """
        Start an existing Glue Crawler.

        Args:
            name: Crawler name
            region: AWS region
            wait: If True, wait for crawler to complete
            dry_run: If True, only show the command

        Returns:
            ExecutionResult
        """
        start_time = datetime.now()

        cmd = ["aws", "glue", "start-crawler", "--name", name, "--region", region]
        command_str = " ".join(cmd)

        if dry_run:
            self._log(f"\n[DRY RUN] Would execute:\n  {command_str}")
            return ExecutionResult(
                suggestion_id=f"start-crawler-{name}",
                cloud="aws",
                resource_type="glue_crawler",
                status=ExecutionStatus.SKIPPED,
                command=command_str,
                output="Dry run - command not executed",
            )

        self._log(f"\n Starting Glue Crawler: {name}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self._log(f"  ✓ Crawler started")

                if wait:
                    self._log(f"  Waiting for crawler to complete...")
                    await self._wait_for_crawler(name, region)

                return ExecutionResult(
                    suggestion_id=f"start-crawler-{name}",
                    cloud="aws",
                    resource_type="glue_crawler",
                    status=ExecutionStatus.SUCCESS,
                    command=command_str,
                    output=result.stdout,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                )
            else:
                return ExecutionResult(
                    suggestion_id=f"start-crawler-{name}",
                    cloud="aws",
                    resource_type="glue_crawler",
                    status=ExecutionStatus.FAILED,
                    command=command_str,
                    error=result.stderr,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                )

        except Exception as e:
            return ExecutionResult(
                suggestion_id=f"start-crawler-{name}",
                cloud="aws",
                resource_type="glue_crawler",
                status=ExecutionStatus.FAILED,
                command=command_str,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    async def _wait_for_crawler(self, name: str, region: str, timeout: int = 600):
        """Wait for a crawler to complete."""
        import time
        start = time.time()

        while time.time() - start < timeout:
            cmd = ["aws", "glue", "get-crawler", "--name", name, "--region", region]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                data = json.loads(result.stdout)
                state = data.get("Crawler", {}).get("State", "")

                if state == "READY":
                    self._log(f"  ✓ Crawler completed")
                    return
                elif state in ["STOPPING", "RUNNING"]:
                    self._log(f"  ... Crawler state: {state}")
                    await asyncio.sleep(10)
                else:
                    self._log(f"  Crawler state: {state}")
                    return
            else:
                self._log(f"  Warning: Could not get crawler state")
                return

        self._log(f"  Warning: Timeout waiting for crawler")

    # =========================================================================
    # Azure Purview Scan Execution
    # =========================================================================

    async def create_purview_scan(
        self,
        account_name: str,
        data_source_name: str,
        scan_name: str,
        scan_config: Dict[str, Any],
        resource_group: str,
        dry_run: bool = True,
    ) -> ExecutionResult:
        """
        Create an Azure Purview scan.

        Args:
            account_name: Purview account name
            data_source_name: Registered data source name
            scan_name: Name for the scan
            scan_config: Scan configuration
            resource_group: Azure resource group
            dry_run: If True, only show the command

        Returns:
            ExecutionResult
        """
        start_time = datetime.now()

        # Build Azure CLI command
        cmd = [
            "az", "purview", "scan", "create",
            "--account-name", account_name,
            "--data-source-name", data_source_name,
            "--scan-name", scan_name,
            "--resource-group", resource_group,
            "--body", json.dumps(scan_config),
        ]

        command_str = " ".join(cmd)

        if dry_run:
            self._log(f"\n[DRY RUN] Would execute:\n  {command_str}")
            return ExecutionResult(
                suggestion_id=f"purview-scan-{scan_name}",
                cloud="azure",
                resource_type="purview_scan",
                status=ExecutionStatus.SKIPPED,
                command=command_str,
                output="Dry run - command not executed",
            )

        self._log(f"\n Creating Purview Scan: {scan_name}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            duration = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                self._log(f"  ✓ Scan created successfully")
                return ExecutionResult(
                    suggestion_id=f"purview-scan-{scan_name}",
                    cloud="azure",
                    resource_type="purview_scan",
                    status=ExecutionStatus.SUCCESS,
                    command=command_str,
                    output=result.stdout,
                    created_resource_id=scan_name,
                    duration_seconds=duration,
                )
            else:
                return ExecutionResult(
                    suggestion_id=f"purview-scan-{scan_name}",
                    cloud="azure",
                    resource_type="purview_scan",
                    status=ExecutionStatus.FAILED,
                    command=command_str,
                    error=result.stderr,
                    duration_seconds=duration,
                )

        except Exception as e:
            return ExecutionResult(
                suggestion_id=f"purview-scan-{scan_name}",
                cloud="azure",
                resource_type="purview_scan",
                status=ExecutionStatus.FAILED,
                command=command_str,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    async def run_purview_scan(
        self,
        account_name: str,
        data_source_name: str,
        scan_name: str,
        resource_group: str,
        dry_run: bool = True,
    ) -> ExecutionResult:
        """Run an existing Purview scan."""
        start_time = datetime.now()

        cmd = [
            "az", "purview", "scan", "run",
            "--account-name", account_name,
            "--data-source-name", data_source_name,
            "--scan-name", scan_name,
            "--resource-group", resource_group,
        ]

        command_str = " ".join(cmd)

        if dry_run:
            self._log(f"\n[DRY RUN] Would execute:\n  {command_str}")
            return ExecutionResult(
                suggestion_id=f"run-purview-scan-{scan_name}",
                cloud="azure",
                resource_type="purview_scan",
                status=ExecutionStatus.SKIPPED,
                command=command_str,
            )

        self._log(f"\n Running Purview Scan: {scan_name}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            duration = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                self._log(f"  ✓ Scan started")
                return ExecutionResult(
                    suggestion_id=f"run-purview-scan-{scan_name}",
                    cloud="azure",
                    resource_type="purview_scan",
                    status=ExecutionStatus.SUCCESS,
                    command=command_str,
                    output=result.stdout,
                    duration_seconds=duration,
                )
            else:
                return ExecutionResult(
                    suggestion_id=f"run-purview-scan-{scan_name}",
                    cloud="azure",
                    resource_type="purview_scan",
                    status=ExecutionStatus.FAILED,
                    command=command_str,
                    error=result.stderr,
                    duration_seconds=duration,
                )
        except Exception as e:
            return ExecutionResult(
                suggestion_id=f"run-purview-scan-{scan_name}",
                cloud="azure",
                resource_type="purview_scan",
                status=ExecutionStatus.FAILED,
                command=command_str,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    # =========================================================================
    # GCP Dataplex Asset Execution
    # =========================================================================

    async def create_dataplex_asset(
        self,
        project: str,
        location: str,
        lake: str,
        zone: str,
        asset_name: str,
        resource_spec: Dict[str, Any],
        discovery_spec: Optional[Dict[str, Any]] = None,
        dry_run: bool = True,
    ) -> ExecutionResult:
        """
        Create a GCP Dataplex asset.

        Args:
            project: GCP project ID
            location: Dataplex location
            lake: Lake ID
            zone: Zone ID
            asset_name: Asset name
            resource_spec: Resource specification
            discovery_spec: Discovery configuration (optional)
            dry_run: If True, only show the command

        Returns:
            ExecutionResult
        """
        start_time = datetime.now()

        # Build gcloud command
        cmd = [
            "gcloud", "dataplex", "assets", "create", asset_name,
            "--project", project,
            "--location", location,
            "--lake", lake,
            "--zone", zone,
            "--resource-type", resource_spec.get("type", "STORAGE_BUCKET"),
            "--resource-name", resource_spec.get("name", ""),
        ]

        if discovery_spec:
            if discovery_spec.get("enabled", True):
                cmd.append("--discovery-enabled")
            if discovery_spec.get("include_patterns"):
                cmd.extend(["--discovery-include-patterns",
                           ",".join(discovery_spec["include_patterns"])])
            if discovery_spec.get("exclude_patterns"):
                cmd.extend(["--discovery-exclude-patterns",
                           ",".join(discovery_spec["exclude_patterns"])])

        command_str = " ".join(cmd)

        if dry_run:
            self._log(f"\n[DRY RUN] Would execute:\n  {command_str}")
            return ExecutionResult(
                suggestion_id=f"dataplex-asset-{asset_name}",
                cloud="gcp",
                resource_type="dataplex_asset",
                status=ExecutionStatus.SKIPPED,
                command=command_str,
                output="Dry run - command not executed",
            )

        self._log(f"\n Creating Dataplex Asset: {asset_name}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            duration = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                self._log(f"  ✓ Asset created successfully")
                return ExecutionResult(
                    suggestion_id=f"dataplex-asset-{asset_name}",
                    cloud="gcp",
                    resource_type="dataplex_asset",
                    status=ExecutionStatus.SUCCESS,
                    command=command_str,
                    output=result.stdout,
                    created_resource_id=asset_name,
                    duration_seconds=duration,
                )
            else:
                return ExecutionResult(
                    suggestion_id=f"dataplex-asset-{asset_name}",
                    cloud="gcp",
                    resource_type="dataplex_asset",
                    status=ExecutionStatus.FAILED,
                    command=command_str,
                    error=result.stderr,
                    duration_seconds=duration,
                )

        except Exception as e:
            return ExecutionResult(
                suggestion_id=f"dataplex-asset-{asset_name}",
                cloud="gcp",
                resource_type="dataplex_asset",
                status=ExecutionStatus.FAILED,
                command=command_str,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    # =========================================================================
    # Batch Execution from Discovery Results
    # =========================================================================

    async def execute_aws_suggestions(
        self,
        enhancements: "AWSDiscoveryEnhancements",
        role_arn: str,
        dry_run: bool = True,
        start_crawlers: bool = False,
    ) -> List[ExecutionResult]:
        """
        Execute all AWS crawler suggestions from discovery.

        Args:
            enhancements: AWS discovery enhancements result
            role_arn: IAM role ARN for crawlers
            dry_run: If True, only show commands
            start_crawlers: If True, also start the crawlers after creation

        Returns:
            List of execution results
        """
        results = []

        self._log("\n" + "=" * 70)
        self._log("  EXECUTING AWS CRAWLER SUGGESTIONS")
        self._log("=" * 70)

        for suggestion in enhancements.crawler_suggestions:
            # Create the crawler
            result = await self.create_glue_crawler(
                name=suggestion.crawler_name,
                s3_path=suggestion.s3_path,
                database=suggestion.suggested_database,
                role=role_arn,
                region=enhancements.region if hasattr(enhancements, 'region') else "us-east-1",
                table_prefix=suggestion.table_prefix,
                dry_run=dry_run,
            )
            results.append(result)

            # Optionally start the crawler
            if start_crawlers and result.status == ExecutionStatus.SUCCESS:
                start_result = await self.start_glue_crawler(
                    name=suggestion.crawler_name,
                    region=enhancements.region if hasattr(enhancements, 'region') else "us-east-1",
                    dry_run=dry_run,
                )
                results.append(start_result)

        self._log(f"\n Executed {len(results)} operations")
        return results

    async def execute_azure_suggestions(
        self,
        enhancements: "AzureDiscoveryEnhancements",
        resource_group: str,
        dry_run: bool = True,
        run_scans: bool = False,
    ) -> List[ExecutionResult]:
        """
        Execute all Azure Purview scan suggestions from discovery.

        Args:
            enhancements: Azure discovery enhancements result
            resource_group: Azure resource group
            dry_run: If True, only show commands
            run_scans: If True, also run the scans after creation

        Returns:
            List of execution results
        """
        results = []

        self._log("\n" + "=" * 70)
        self._log("  EXECUTING AZURE PURVIEW SCAN SUGGESTIONS")
        self._log("=" * 70)

        for suggestion in enhancements.scan_suggestions:
            # Create the scan
            result = await self.create_purview_scan(
                account_name=suggestion.purview_account,
                data_source_name=suggestion.data_source_name,
                scan_name=suggestion.scan_name,
                scan_config=suggestion.scan_config,
                resource_group=resource_group,
                dry_run=dry_run,
            )
            results.append(result)

            # Optionally run the scan
            if run_scans and result.status == ExecutionStatus.SUCCESS:
                run_result = await self.run_purview_scan(
                    account_name=suggestion.purview_account,
                    data_source_name=suggestion.data_source_name,
                    scan_name=suggestion.scan_name,
                    resource_group=resource_group,
                    dry_run=dry_run,
                )
                results.append(run_result)

        self._log(f"\n Executed {len(results)} operations")
        return results

    async def execute_gcp_suggestions(
        self,
        enhancements: "GCPDiscoveryEnhancements",
        dry_run: bool = True,
    ) -> List[ExecutionResult]:
        """
        Execute all GCP Dataplex asset suggestions from discovery.

        Args:
            enhancements: GCP discovery enhancements result
            dry_run: If True, only show commands

        Returns:
            List of execution results
        """
        results = []

        self._log("\n" + "=" * 70)
        self._log("  EXECUTING GCP DATAPLEX ASSET SUGGESTIONS")
        self._log("=" * 70)

        for suggestion in enhancements.asset_suggestions:
            result = await self.create_dataplex_asset(
                project=suggestion.project,
                location=suggestion.location,
                lake=suggestion.suggested_lake,
                zone=suggestion.suggested_zone,
                asset_name=suggestion.asset_name,
                resource_spec={
                    "type": suggestion.resource_type,
                    "name": suggestion.resource_name,
                },
                discovery_spec=suggestion.discovery_config,
                dry_run=dry_run,
            )
            results.append(result)

        self._log(f"\n Executed {len(results)} operations")
        return results

    # =========================================================================
    # Summary and Reporting
    # =========================================================================

    def print_execution_summary(self, results: List[ExecutionResult]) -> str:
        """Print a summary of execution results."""
        lines = [
            "",
            "=" * 70,
            "  EXECUTION SUMMARY",
            "=" * 70,
        ]

        # Count by status
        status_counts = {}
        for result in results:
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        lines.append(f"\n  Total operations: {len(results)}")
        for status, count in status_counts.items():
            icon = "✓" if status == "success" else "○" if status == "skipped" else "✗"
            lines.append(f"    {icon} {status}: {count}")

        # Group by cloud
        by_cloud = {}
        for result in results:
            by_cloud.setdefault(result.cloud, []).append(result)

        for cloud, cloud_results in by_cloud.items():
            lines.append(f"\n  {cloud.upper()}:")
            for result in cloud_results:
                icon = "✓" if result.status == ExecutionStatus.SUCCESS else "○" if result.status == ExecutionStatus.SKIPPED else "✗"
                lines.append(f"    {icon} {result.resource_type}: {result.suggestion_id}")
                if result.error:
                    lines.append(f"      Error: {result.error[:60]}...")

        lines.append("")
        return "\n".join(lines)


def print_execution_results(results: List[ExecutionResult]) -> str:
    """Convenience function to print execution results."""
    cli = DiscoveryCLI(verbose=False)
    return cli.print_execution_summary(results)


# =============================================================================
# Main CLI Entry Point
# =============================================================================

async def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DATAMESH.AI Discovery CLI - Execute discovery suggestions"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # AWS subcommand
    aws_parser = subparsers.add_parser("aws", help="Execute AWS suggestions")
    aws_parser.add_argument("--role", required=True, help="IAM role ARN for crawlers")
    aws_parser.add_argument("--region", default="us-east-1", help="AWS region")
    aws_parser.add_argument("--execute", action="store_true", help="Actually execute (not dry run)")
    aws_parser.add_argument("--start", action="store_true", help="Start crawlers after creation")

    # Azure subcommand
    azure_parser = subparsers.add_parser("azure", help="Execute Azure suggestions")
    azure_parser.add_argument("--resource-group", required=True, help="Azure resource group")
    azure_parser.add_argument("--execute", action="store_true", help="Actually execute")
    azure_parser.add_argument("--run-scans", action="store_true", help="Run scans after creation")

    # GCP subcommand
    gcp_parser = subparsers.add_parser("gcp", help="Execute GCP suggestions")
    gcp_parser.add_argument("--project", required=True, help="GCP project ID")
    gcp_parser.add_argument("--execute", action="store_true", help="Actually execute")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    cli = DiscoveryCLI(verbose=True)

    if args.command == "aws":
        # Run AWS discovery first
        from .aws_enhancements import AWSEnhancedDiscovery

        print(f"\n Running AWS discovery in {args.region}...")
        discovery = AWSEnhancedDiscovery(region=args.region)
        enhancements = await discovery.discover()

        if not enhancements.crawler_suggestions:
            print(" No crawler suggestions found.")
            return 0

        print(f" Found {len(enhancements.crawler_suggestions)} crawler suggestions")

        results = await cli.execute_aws_suggestions(
            enhancements=enhancements,
            role_arn=args.role,
            dry_run=not args.execute,
            start_crawlers=args.start,
        )

        print(cli.print_execution_summary(results))

    elif args.command == "azure":
        from .azure_enhancements import AzureEnhancedDiscovery

        print("\n Running Azure discovery...")
        discovery = AzureEnhancedDiscovery()
        enhancements = await discovery.discover()

        if not enhancements.scan_suggestions:
            print(" No scan suggestions found.")
            return 0

        print(f" Found {len(enhancements.scan_suggestions)} scan suggestions")

        results = await cli.execute_azure_suggestions(
            enhancements=enhancements,
            resource_group=args.resource_group,
            dry_run=not args.execute,
            run_scans=args.run_scans,
        )

        print(cli.print_execution_summary(results))

    elif args.command == "gcp":
        from .gcp_enhancements import GCPEnhancedDiscovery

        print("\n Running GCP discovery...")
        discovery = GCPEnhancedDiscovery(project=args.project)
        enhancements = await discovery.discover()

        if not enhancements.asset_suggestions:
            print(" No asset suggestions found.")
            return 0

        print(f" Found {len(enhancements.asset_suggestions)} asset suggestions")

        results = await cli.execute_gcp_suggestions(
            enhancements=enhancements,
            dry_run=not args.execute,
        )

        print(cli.print_execution_summary(results))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
