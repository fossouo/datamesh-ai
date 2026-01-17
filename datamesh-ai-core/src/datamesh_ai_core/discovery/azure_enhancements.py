"""
Azure Enhancements - Additional Azure-specific discovery features.

Includes:
- Azure Storage Account discovery and analysis
- Purview collection and asset scanning
- Data Factory pipeline discovery
- Synapse workspace and pool detection
- Azure SQL/Cosmos DB discovery
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredStorageAccount:
    """Discovered Azure Storage Account."""
    name: str
    resource_group: str
    location: str
    kind: str  # StorageV2, BlobStorage, etc.
    containers: list[str] = field(default_factory=list)
    is_catalogued: bool = False
    catalogued_paths: list[str] = field(default_factory=list)
    uncatalogued_paths: list[str] = field(default_factory=list)
    data_formats: list[str] = field(default_factory=list)
    has_hierarchical_namespace: bool = False  # ADLS Gen2
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveredPurviewAsset:
    """Discovered Purview asset."""
    name: str
    qualified_name: str
    asset_type: str
    collection: str
    classifications: list[str] = field(default_factory=list)
    glossary_terms: list[str] = field(default_factory=list)
    has_lineage: bool = False


@dataclass
class DiscoveredDataFactory:
    """Discovered Azure Data Factory."""
    name: str
    resource_group: str
    location: str
    pipelines: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    linked_services: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)


@dataclass
class DiscoveredSynapseWorkspace:
    """Discovered Synapse Analytics workspace."""
    name: str
    resource_group: str
    location: str
    sql_pools: list[str] = field(default_factory=list)
    spark_pools: list[str] = field(default_factory=list)
    databases: list[str] = field(default_factory=list)
    linked_services: list[str] = field(default_factory=list)
    sql_endpoint: Optional[str] = None


@dataclass
class PurviewScanSuggestion:
    """Suggested Purview scan configuration."""
    name: str
    source_type: str
    target: str
    collection: str
    description: str
    reason: str
    cli_command: str = ""
    arm_template_snippet: str = ""


@dataclass
class AzureDiscoveryEnhancements:
    """Enhanced Azure discovery results."""
    storage_accounts: list[DiscoveredStorageAccount] = field(default_factory=list)
    purview_assets: list[DiscoveredPurviewAsset] = field(default_factory=list)
    data_factories: list[DiscoveredDataFactory] = field(default_factory=list)
    synapse_workspaces: list[DiscoveredSynapseWorkspace] = field(default_factory=list)
    scan_suggestions: list[PurviewScanSuggestion] = field(default_factory=list)
    purview_account: Optional[str] = None
    purview_collections: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class AzureEnhancedDiscovery:
    """
    Enhanced Azure discovery with Storage, Purview, Data Factory, and Synapse.
    """

    def __init__(
        self,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self._credential = None
        self._storage_client = None
        self._purview_client = None
        self._datafactory_client = None
        self._synapse_client = None

    async def discover(self) -> AzureDiscoveryEnhancements:
        """Run enhanced Azure discovery."""
        result = AzureDiscoveryEnhancements()

        try:
            from azure.identity import DefaultAzureCredential

            self._credential = DefaultAzureCredential()

            # Get subscription ID if not provided
            if not self.subscription_id:
                self.subscription_id = await self._get_subscription_id()

            if not self.subscription_id:
                logger.warning("No Azure subscription ID found")
                return result

            # Discover storage accounts
            result.storage_accounts = await self._discover_storage_accounts()

            # Discover Purview
            purview_result = await self._discover_purview()
            result.purview_account = purview_result.get("account")
            result.purview_collections = purview_result.get("collections", [])
            result.purview_assets = purview_result.get("assets", [])

            # Discover Data Factory
            result.data_factories = await self._discover_data_factories()

            # Discover Synapse
            result.synapse_workspaces = await self._discover_synapse_workspaces()

            # Generate scan suggestions
            result.scan_suggestions = self._generate_scan_suggestions(
                result.storage_accounts,
                result.purview_assets,
                result.purview_account,
            )

            # Generate recommendations
            result.recommendations = self._generate_recommendations(result)

        except ImportError as e:
            logger.warning(f"Azure SDK not fully installed: {e}")
            result.recommendations.append(
                "Install Azure SDK: pip install azure-identity azure-mgmt-storage "
                "azure-mgmt-datafactory azure-synapse"
            )
        except Exception as e:
            logger.error(f"Error in Azure discovery: {e}")
            result.recommendations.append(f"Azure discovery error: {str(e)}")

        return result

    async def _get_subscription_id(self) -> Optional[str]:
        """Get subscription ID from environment or Azure CLI."""
        import os

        # Check environment
        sub_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
        if sub_id:
            return sub_id

        # Try to get from Azure CLI config
        try:
            import json
            import subprocess

            result = subprocess.run(
                ["az", "account", "show"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get("id")
        except Exception:
            pass

        return None

    async def _discover_storage_accounts(self) -> list[DiscoveredStorageAccount]:
        """Discover Azure Storage Accounts."""
        accounts = []

        try:
            from azure.mgmt.storage import StorageManagementClient

            self._storage_client = StorageManagementClient(
                credential=self._credential,
                subscription_id=self.subscription_id,
            )

            # List storage accounts
            for account in self._storage_client.storage_accounts.list():
                # Filter by resource group if specified
                if self.resource_group:
                    rg = account.id.split("/")[4]
                    if rg.lower() != self.resource_group.lower():
                        continue

                storage_account = DiscoveredStorageAccount(
                    name=account.name,
                    resource_group=account.id.split("/")[4],
                    location=account.location,
                    kind=account.kind.value if account.kind else "Unknown",
                    has_hierarchical_namespace=account.is_hns_enabled or False,
                )

                # List containers
                try:
                    rg = account.id.split("/")[4]
                    containers = self._storage_client.blob_containers.list(
                        resource_group_name=rg,
                        account_name=account.name,
                    )
                    storage_account.containers = [c.name for c in containers]

                    # Detect data formats (sample approach)
                    storage_account.data_formats = await self._detect_storage_formats(
                        account.name, storage_account.containers[:5]
                    )

                except Exception as e:
                    logger.debug(f"Could not list containers for {account.name}: {e}")

                accounts.append(storage_account)

        except ImportError:
            logger.warning("azure-mgmt-storage not installed")
        except Exception as e:
            logger.error(f"Error discovering storage accounts: {e}")

        return accounts

    async def _detect_storage_formats(
        self, account_name: str, containers: list[str]
    ) -> list[str]:
        """Detect data formats in storage containers."""
        formats = set()

        try:
            from azure.storage.blob import BlobServiceClient

            # Use account URL with credential
            account_url = f"https://{account_name}.blob.core.windows.net"
            blob_service = BlobServiceClient(
                account_url=account_url,
                credential=self._credential,
            )

            for container in containers[:3]:  # Limit for performance
                try:
                    container_client = blob_service.get_container_client(container)
                    blobs = container_client.list_blobs(max_results=50)

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
                        elif name.endswith(".delta") or "/_delta_log/" in name:
                            formats.add("delta")

                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Error detecting formats: {e}")

        return list(formats)

    async def _discover_purview(self) -> dict:
        """Discover Microsoft Purview account and assets."""
        result = {"account": None, "collections": [], "assets": []}

        try:
            from azure.mgmt.purview import PurviewManagementClient

            purview_mgmt = PurviewManagementClient(
                credential=self._credential,
                subscription_id=self.subscription_id,
            )

            # Find Purview accounts
            for account in purview_mgmt.accounts.list_by_subscription():
                result["account"] = account.name

                # Try to get collections and assets via Purview Data Catalog API
                try:
                    from azure.purview.catalog import PurviewCatalogClient

                    endpoint = f"https://{account.name}.purview.azure.com"
                    catalog_client = PurviewCatalogClient(
                        endpoint=endpoint,
                        credential=self._credential,
                    )

                    # Search for assets
                    search_request = {"keywords": "*", "limit": 100}
                    search_result = catalog_client.discovery.query(search_request)

                    collections = set()
                    for entity in search_result.get("value", []):
                        collections.add(entity.get("collectionId", "root"))

                        asset = DiscoveredPurviewAsset(
                            name=entity.get("name", ""),
                            qualified_name=entity.get("qualifiedName", ""),
                            asset_type=entity.get("entityType", ""),
                            collection=entity.get("collectionId", "root"),
                            classifications=entity.get("classification", []),
                        )
                        result["assets"].append(asset)

                    result["collections"] = list(collections)

                except ImportError:
                    logger.debug("azure-purview-catalog not installed")
                except Exception as e:
                    logger.debug(f"Could not query Purview catalog: {e}")

                break  # Use first Purview account

        except ImportError:
            logger.debug("azure-mgmt-purview not installed")
        except Exception as e:
            logger.debug(f"Error discovering Purview: {e}")

        return result

    async def _discover_data_factories(self) -> list[DiscoveredDataFactory]:
        """Discover Azure Data Factory instances."""
        factories = []

        try:
            from azure.mgmt.datafactory import DataFactoryManagementClient

            self._datafactory_client = DataFactoryManagementClient(
                credential=self._credential,
                subscription_id=self.subscription_id,
            )

            for factory in self._datafactory_client.factories.list():
                rg = factory.id.split("/")[4]

                # Filter by resource group if specified
                if self.resource_group and rg.lower() != self.resource_group.lower():
                    continue

                adf = DiscoveredDataFactory(
                    name=factory.name,
                    resource_group=rg,
                    location=factory.location,
                )

                # List pipelines
                try:
                    pipelines = self._datafactory_client.pipelines.list_by_factory(
                        resource_group_name=rg,
                        factory_name=factory.name,
                    )
                    adf.pipelines = [p.name for p in pipelines]
                except Exception:
                    pass

                # List datasets
                try:
                    datasets = self._datafactory_client.datasets.list_by_factory(
                        resource_group_name=rg,
                        factory_name=factory.name,
                    )
                    adf.datasets = [d.name for d in datasets]
                except Exception:
                    pass

                # List linked services
                try:
                    linked = self._datafactory_client.linked_services.list_by_factory(
                        resource_group_name=rg,
                        factory_name=factory.name,
                    )
                    adf.linked_services = [ls.name for ls in linked]
                except Exception:
                    pass

                factories.append(adf)

        except ImportError:
            logger.debug("azure-mgmt-datafactory not installed")
        except Exception as e:
            logger.debug(f"Error discovering Data Factory: {e}")

        return factories

    async def _discover_synapse_workspaces(self) -> list[DiscoveredSynapseWorkspace]:
        """Discover Azure Synapse Analytics workspaces."""
        workspaces = []

        try:
            from azure.mgmt.synapse import SynapseManagementClient

            self._synapse_client = SynapseManagementClient(
                credential=self._credential,
                subscription_id=self.subscription_id,
            )

            for workspace in self._synapse_client.workspaces.list():
                rg = workspace.id.split("/")[4]

                # Filter by resource group if specified
                if self.resource_group and rg.lower() != self.resource_group.lower():
                    continue

                synapse = DiscoveredSynapseWorkspace(
                    name=workspace.name,
                    resource_group=rg,
                    location=workspace.location,
                    sql_endpoint=workspace.connectivity_endpoints.get("sql") if workspace.connectivity_endpoints else None,
                )

                # List SQL pools
                try:
                    sql_pools = self._synapse_client.sql_pools.list_by_workspace(
                        resource_group_name=rg,
                        workspace_name=workspace.name,
                    )
                    synapse.sql_pools = [p.name for p in sql_pools]
                except Exception:
                    pass

                # List Spark pools
                try:
                    spark_pools = self._synapse_client.big_data_pools.list_by_workspace(
                        resource_group_name=rg,
                        workspace_name=workspace.name,
                    )
                    synapse.spark_pools = [p.name for p in spark_pools]
                except Exception:
                    pass

                workspaces.append(synapse)

        except ImportError:
            logger.debug("azure-mgmt-synapse not installed")
        except Exception as e:
            logger.debug(f"Error discovering Synapse: {e}")

        return workspaces

    def _generate_scan_suggestions(
        self,
        storage_accounts: list[DiscoveredStorageAccount],
        purview_assets: list[DiscoveredPurviewAsset],
        purview_account: Optional[str],
    ) -> list[PurviewScanSuggestion]:
        """Generate Purview scan suggestions for uncatalogued storage."""
        suggestions = []

        if not purview_account:
            return suggestions

        # Get already scanned paths
        scanned_paths = set()
        for asset in purview_assets:
            scanned_paths.add(asset.qualified_name.lower())

        for account in storage_accounts:
            # Skip if already fully catalogued
            if account.is_catalogued:
                continue

            for container in account.containers[:5]:  # Limit suggestions
                path = f"https://{account.name}.blob.core.windows.net/{container}"

                # Skip if already scanned
                if any(path.lower() in p for p in scanned_paths):
                    continue

                # Skip system containers
                if container.startswith("$") or container in ["insights-logs", "azure-webjobs"]:
                    continue

                scan_name = f"scan-{account.name}-{container}"
                source_type = "AdlsGen2" if account.has_hierarchical_namespace else "AzureBlob"

                cli_command = self._generate_purview_scan_cli(
                    purview_account, scan_name, source_type, path
                )

                suggestions.append(PurviewScanSuggestion(
                    name=scan_name,
                    source_type=source_type,
                    target=path,
                    collection="root",
                    description=f"Scan {container} container in {account.name}",
                    reason=f"Found {', '.join(account.data_formats) or 'data'} not yet in Purview",
                    cli_command=cli_command,
                ))

        return suggestions

    def _generate_purview_scan_cli(
        self,
        purview_account: str,
        scan_name: str,
        source_type: str,
        target: str,
    ) -> str:
        """Generate Azure CLI command for Purview scan."""
        return f"""# Register data source (if not already registered)
az purview data-source create \\
    --account-name {purview_account} \\
    --name {scan_name.replace('scan-', 'source-')} \\
    --kind {source_type}

# Create and run scan
az purview scan create \\
    --account-name {purview_account} \\
    --data-source-name {scan_name.replace('scan-', 'source-')} \\
    --name {scan_name} \\
    --scan-kind {source_type}

az purview scan run \\
    --account-name {purview_account} \\
    --data-source-name {scan_name.replace('scan-', 'source-')} \\
    --scan-name {scan_name}"""

    def _generate_recommendations(
        self, result: AzureDiscoveryEnhancements
    ) -> list[str]:
        """Generate Azure-specific recommendations."""
        recommendations = []

        # Purview recommendations
        if not result.purview_account:
            recommendations.append(
                "No Microsoft Purview account found. Consider setting up Purview "
                "for centralized data governance, lineage, and classification."
            )
        elif result.scan_suggestions:
            recommendations.append(
                f"Found {len(result.scan_suggestions)} storage locations not in Purview. "
                "Consider running the suggested scans to catalog this data."
            )

        # Storage recommendations
        adls_gen2_accounts = [a for a in result.storage_accounts if a.has_hierarchical_namespace]
        blob_accounts = [a for a in result.storage_accounts if not a.has_hierarchical_namespace]

        if blob_accounts and not adls_gen2_accounts:
            recommendations.append(
                "Consider using ADLS Gen2 (hierarchical namespace) for analytics workloads. "
                "It provides better performance and finer access control."
            )

        # Data Factory recommendations
        if result.data_factories:
            for adf in result.data_factories:
                if not adf.pipelines:
                    recommendations.append(
                        f"Data Factory '{adf.name}' has no pipelines. "
                        "Consider setting up data ingestion pipelines."
                    )

        # Synapse recommendations
        if not result.synapse_workspaces:
            if result.storage_accounts:
                recommendations.append(
                    "Consider Azure Synapse Analytics for unified analytics. "
                    "It provides SQL pools, Spark pools, and integrated data exploration."
                )

        return recommendations


def print_azure_enhancements(result: AzureDiscoveryEnhancements) -> str:
    """Generate printable output for Azure enhancements."""
    lines = []

    lines.append("")
    lines.append("=" * 70)
    lines.append("  AZURE ENHANCED DISCOVERY")
    lines.append("=" * 70)

    # Storage Accounts
    lines.append("")
    lines.append("  STORAGE ACCOUNTS")
    lines.append("  " + "-" * 40)

    if result.storage_accounts:
        adls = [a for a in result.storage_accounts if a.has_hierarchical_namespace]
        blob = [a for a in result.storage_accounts if not a.has_hierarchical_namespace]

        lines.append(f"    ADLS Gen2: {len(adls)}")
        lines.append(f"    Blob Storage: {len(blob)}")
        lines.append("")

        for account in result.storage_accounts[:5]:
            account_type = "ADLS Gen2" if account.has_hierarchical_namespace else "Blob"
            formats = ", ".join(account.data_formats) if account.data_formats else "unknown"
            lines.append(f"    {account.name} ({account_type})")
            lines.append(f"      Location: {account.location}")
            lines.append(f"      Containers: {len(account.containers)}")
            lines.append(f"      Data formats: {formats}")
    else:
        lines.append("    No storage accounts found")

    # Purview
    lines.append("")
    lines.append("  MICROSOFT PURVIEW")
    lines.append("  " + "-" * 40)

    if result.purview_account:
        lines.append(f"    Account: {result.purview_account}")
        lines.append(f"    Collections: {len(result.purview_collections)}")
        lines.append(f"    Catalogued assets: {len(result.purview_assets)}")

        if result.purview_assets:
            lines.append("")
            lines.append("    Asset types:")
            type_counts = {}
            for asset in result.purview_assets:
                type_counts[asset.asset_type] = type_counts.get(asset.asset_type, 0) + 1
            for asset_type, count in sorted(type_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"      - {asset_type}: {count}")
    else:
        lines.append("    No Purview account found")

    # Data Factory
    lines.append("")
    lines.append("  DATA FACTORY")
    lines.append("  " + "-" * 40)

    if result.data_factories:
        for adf in result.data_factories:
            lines.append(f"    {adf.name}")
            lines.append(f"      Pipelines: {len(adf.pipelines)}")
            lines.append(f"      Datasets: {len(adf.datasets)}")
            lines.append(f"      Linked Services: {len(adf.linked_services)}")
    else:
        lines.append("    No Data Factory found")

    # Synapse
    lines.append("")
    lines.append("  SYNAPSE ANALYTICS")
    lines.append("  " + "-" * 40)

    if result.synapse_workspaces:
        for synapse in result.synapse_workspaces:
            lines.append(f"    {synapse.name}")
            lines.append(f"      SQL Pools: {len(synapse.sql_pools)}")
            lines.append(f"      Spark Pools: {len(synapse.spark_pools)}")
            if synapse.sql_endpoint:
                lines.append(f"      SQL Endpoint: {synapse.sql_endpoint}")
    else:
        lines.append("    No Synapse workspaces found")

    # Scan Suggestions
    if result.scan_suggestions:
        lines.append("")
        lines.append("  PURVIEW SCAN SUGGESTIONS")
        lines.append("  " + "-" * 40)

        for suggestion in result.scan_suggestions[:5]:
            lines.append(f"    {suggestion.name}")
            lines.append(f"      Target: {suggestion.target}")
            lines.append(f"      Reason: {suggestion.reason}")
            lines.append("")

    # Recommendations
    if result.recommendations:
        lines.append("")
        lines.append("  AZURE RECOMMENDATIONS")
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
