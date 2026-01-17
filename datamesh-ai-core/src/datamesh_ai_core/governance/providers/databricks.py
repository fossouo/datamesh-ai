"""
Databricks Governance Provider - Unity Catalog and Workspace ACLs integration.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from ..interfaces import (
    GovernanceProvider,
    UserContext,
    Resource,
    Permission,
    Classification,
    MaskingRule,
    RowFilter,
    AccessDecision,
    ClassificationLevel,
)
from ..config import DatabricksConfig

logger = logging.getLogger(__name__)


class DatabricksGovernanceProvider(GovernanceProvider):
    """
    Databricks Governance Provider using Unity Catalog and Workspace ACLs.

    This provider integrates with:
    - Unity Catalog: Centralized data governance
    - Workspace ACLs: Access control for notebooks and clusters
    - Column masking and row filters
    - Audit logging via system tables
    """

    def __init__(self, config: Optional[DatabricksConfig] = None):
        self.config = config or DatabricksConfig()
        self._workspace_client = None
        self._sql_client = None
        self._connected = False

    @property
    def provider_name(self) -> str:
        return "databricks"

    async def connect(self, config: dict) -> None:
        """
        Establish connection to Databricks workspace.

        Args:
            config: Databricks configuration including workspace_url and token
        """
        try:
            from databricks.sdk import WorkspaceClient
            from databricks.sdk.service.sql import StatementExecutionAPI

            workspace_url = config.get("workspace_url", self.config.workspace_url)
            token = config.get("token", self.config.token)

            if not workspace_url:
                raise ValueError("workspace_url is required for Databricks connection")

            self._workspace_client = WorkspaceClient(
                host=workspace_url,
                token=token,
            )

            self._connected = True
            logger.info(f"Connected to Databricks workspace: {workspace_url}")

        except ImportError:
            raise ImportError(
                "databricks-sdk is required for Databricks governance. "
                "Install with: pip install databricks-sdk"
            )

    async def disconnect(self) -> None:
        """Close Databricks connection."""
        self._workspace_client = None
        self._sql_client = None
        self._connected = False

    async def health_check(self) -> bool:
        """Check if Databricks workspace is accessible."""
        if not self._connected or not self._workspace_client:
            return False

        try:
            # Try to get current user
            self._workspace_client.current_user.me()
            return True
        except Exception as e:
            logger.warning(f"Databricks health check failed: {e}")
            return False

    async def get_user_context(self, credentials: dict) -> UserContext:
        """
        Get user context from Databricks workspace.
        """
        if not self._workspace_client:
            raise RuntimeError("Not connected to Databricks")

        try:
            # Get current user info
            user_info = self._workspace_client.current_user.me()

            # Get group memberships
            groups = []
            try:
                for group in self._workspace_client.groups.list():
                    # Check if user is in this group
                    try:
                        group_details = self._workspace_client.groups.get(group.id)
                        for member in group_details.members or []:
                            if member.value == user_info.id:
                                groups.append(group.display_name)
                                break
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Could not fetch group memberships: {e}")

            return UserContext(
                user_id=user_info.id,
                email=user_info.user_name,
                display_name=user_info.display_name or user_info.user_name,
                groups=groups,
                roles=[],
                attributes={
                    "workspace_url": self.config.workspace_url,
                    "active": user_info.active,
                },
                provider="databricks",
            )

        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return UserContext(
                user_id="databricks-user",
                provider="databricks",
            )

    async def get_accessible_resources(
        self,
        user: UserContext,
        resource_type: Optional[str] = None,
        catalog: Optional[str] = None,
        database: Optional[str] = None,
    ) -> list[Resource]:
        """
        Get resources accessible to the user via Unity Catalog.
        """
        resources = []

        if not self._workspace_client:
            return resources

        try:
            # List catalogs
            catalogs_to_check = []
            if catalog:
                catalogs_to_check = [catalog]
            else:
                for cat in self._workspace_client.catalogs.list():
                    catalogs_to_check.append(cat.name)

            for cat_name in catalogs_to_check:
                # List schemas (databases)
                schemas_to_check = []
                if database:
                    schemas_to_check = [database]
                else:
                    try:
                        for schema in self._workspace_client.schemas.list(catalog_name=cat_name):
                            schemas_to_check.append(schema.name)
                    except Exception:
                        continue

                for schema_name in schemas_to_check:
                    if not resource_type or resource_type == "database":
                        resources.append(Resource(
                            resource_type="database",
                            catalog=cat_name,
                            schema=schema_name,
                            provider="databricks",
                        ))

                    if not resource_type or resource_type == "table":
                        try:
                            for table in self._workspace_client.tables.list(
                                catalog_name=cat_name,
                                schema_name=schema_name,
                            ):
                                resources.append(Resource(
                                    resource_type="table",
                                    catalog=cat_name,
                                    schema=schema_name,
                                    table=table.name,
                                    provider="databricks",
                                    metadata={
                                        "table_type": table.table_type.value if table.table_type else None,
                                        "data_source_format": table.data_source_format.value if table.data_source_format else None,
                                    },
                                ))
                        except Exception as e:
                            logger.warning(f"Error listing tables in {cat_name}.{schema_name}: {e}")

        except Exception as e:
            logger.error(f"Error fetching accessible resources: {e}")

        return resources

    async def check_permission(
        self,
        user: UserContext,
        resource: Resource,
        action: str,
    ) -> Permission:
        """
        Check if user has permission via Unity Catalog grants.
        """
        if not self._workspace_client:
            raise RuntimeError("Not connected to Databricks")

        try:
            # Map action to Unity Catalog privilege
            privilege_map = {
                "SELECT": "SELECT",
                "INSERT": "MODIFY",
                "DELETE": "MODIFY",
                "UPDATE": "MODIFY",
                "CREATE": "CREATE TABLE",
                "DROP": "OWNERSHIP",
                "DESCRIBE": "SELECT",  # SELECT implies DESCRIBE
            }

            required_privilege = privilege_map.get(action.upper(), action.upper())

            # Get grants for the securable
            securable_type = "TABLE"
            full_name = f"{resource.catalog}.{resource.schema}.{resource.table}"

            if resource.resource_type == "database":
                securable_type = "SCHEMA"
                full_name = f"{resource.catalog}.{resource.schema}"
            elif resource.resource_type == "catalog":
                securable_type = "CATALOG"
                full_name = resource.catalog

            grants = self._workspace_client.grants.get(
                securable_type=securable_type,
                full_name=full_name,
            )

            # Check if user or their groups have the required privilege
            user_principals = [user.email] + user.groups

            for assignment in grants.privilege_assignments or []:
                if assignment.principal in user_principals:
                    privileges = [p.privilege.value for p in assignment.privileges or []]
                    if required_privilege in privileges or "ALL PRIVILEGES" in privileges:
                        return Permission(
                            resource=resource,
                            action=action,
                            decision=AccessDecision.ALLOW,
                            granted_by="unity_catalog",
                        )

            return Permission(
                resource=resource,
                action=action,
                decision=AccessDecision.DENY,
                reason="No Unity Catalog grant found",
            )

        except Exception as e:
            logger.error(f"Error checking permission: {e}")
            return Permission(
                resource=resource,
                action=action,
                decision=AccessDecision.DENY,
                reason=f"Error checking permission: {str(e)}",
            )

    async def get_classifications(
        self,
        resource: Resource,
        user: Optional[UserContext] = None,
    ) -> list[Classification]:
        """
        Get data classifications from Unity Catalog tags.
        """
        classifications = []

        if not self._workspace_client:
            return classifications

        try:
            if resource.table and resource.schema and resource.catalog:
                # Get table info which includes tags
                table_info = self._workspace_client.tables.get(
                    full_name=f"{resource.catalog}.{resource.schema}.{resource.table}"
                )

                # Check table-level tags
                if table_info.properties:
                    for key, value in table_info.properties.items():
                        if "classification" in key.lower() or "sensitivity" in key.lower():
                            classifications.append(Classification(
                                resource=resource,
                                classification=value,
                                level=self._map_databricks_classification(value),
                                source="unity_catalog",
                                tags=[key],
                            ))

                # Check column-level tags/comments
                for col in table_info.columns or []:
                    if col.comment and any(
                        x in col.comment.lower()
                        for x in ["pii", "sensitive", "confidential", "phi", "pci"]
                    ):
                        col_resource = Resource(
                            resource_type="column",
                            catalog=resource.catalog,
                            schema=resource.schema,
                            table=resource.table,
                            column=col.name,
                            provider="databricks",
                        )
                        classifications.append(Classification(
                            resource=col_resource,
                            classification=col.comment,
                            level=self._infer_classification_from_text(col.comment),
                            source="unity_catalog",
                        ))

        except Exception as e:
            logger.warning(f"Error fetching classifications: {e}")

        return classifications

    async def get_masking_rules(
        self,
        resource: Resource,
        user: UserContext,
    ) -> list[MaskingRule]:
        """
        Get column masking rules from Unity Catalog.
        """
        masking_rules = []

        if not self._workspace_client:
            return masking_rules

        try:
            if resource.table and resource.schema and resource.catalog:
                # Get table info
                table_info = self._workspace_client.tables.get(
                    full_name=f"{resource.catalog}.{resource.schema}.{resource.table}"
                )

                # Check for column masks
                for col in table_info.columns or []:
                    if col.mask:
                        masking_rules.append(MaskingRule(
                            resource=resource,
                            column=col.name,
                            mask_type="unity_catalog_mask",
                            mask_function=col.mask.function_name,
                            parameters={
                                "using_column_names": col.mask.using_column_names,
                            },
                        ))

        except Exception as e:
            logger.warning(f"Error fetching masking rules: {e}")

        return masking_rules

    async def get_row_filters(
        self,
        resource: Resource,
        user: UserContext,
    ) -> list[RowFilter]:
        """
        Get row-level security filters from Unity Catalog.
        """
        row_filters = []

        if not self._workspace_client:
            return row_filters

        try:
            if resource.table and resource.schema and resource.catalog:
                # Get table info
                table_info = self._workspace_client.tables.get(
                    full_name=f"{resource.catalog}.{resource.schema}.{resource.table}"
                )

                # Check for row filter
                if table_info.row_filter:
                    row_filters.append(RowFilter(
                        resource=resource,
                        filter_expression=table_info.row_filter.function_name,
                        filter_type="unity_catalog_function",
                    ))

        except Exception as e:
            logger.warning(f"Error fetching row filters: {e}")

        return row_filters

    async def audit_access(
        self,
        user: UserContext,
        resource: Resource,
        action: str,
        decision: AccessDecision,
        details: Optional[dict] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Log audit entry.

        Note: Databricks has built-in audit logging via system tables.
        This logs additional application-level auditing.
        """
        try:
            audit_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user.user_id,
                "user_email": user.email,
                "resource": resource.fully_qualified_name,
                "resource_type": resource.resource_type,
                "action": action,
                "decision": decision.value,
                "trace_id": trace_id,
                "details": details or {},
            }

            # Log locally - Databricks system tables capture the actual access
            logger.info(f"Audit: {audit_record}")

        except Exception as e:
            logger.error(f"Error logging audit entry: {e}")

    def _map_databricks_classification(self, classification: str) -> ClassificationLevel:
        """Map Databricks classification to unified level."""
        classification_upper = classification.upper()

        databricks_mappings = {
            "PERSONALLY_IDENTIFIABLE_INFORMATION": ClassificationLevel.PII,
            "PII": ClassificationLevel.PII,
            "PAYMENT_CARD_DATA": ClassificationLevel.PCI,
            "PCI": ClassificationLevel.PCI,
            "HEALTH_INFORMATION": ClassificationLevel.PHI,
            "PHI": ClassificationLevel.PHI,
            "HIPAA": ClassificationLevel.PHI,
            "SENSITIVE": ClassificationLevel.CONFIDENTIAL,
            "CONFIDENTIAL": ClassificationLevel.CONFIDENTIAL,
            "RESTRICTED": ClassificationLevel.RESTRICTED,
            "SECRET": ClassificationLevel.RESTRICTED,
            "INTERNAL": ClassificationLevel.INTERNAL,
            "PUBLIC": ClassificationLevel.PUBLIC,
        }

        if classification_upper in databricks_mappings:
            return databricks_mappings[classification_upper]

        return self._infer_classification_from_text(classification)

    def _infer_classification_from_text(self, text: str) -> ClassificationLevel:
        """Infer classification level from descriptive text."""
        text_lower = text.lower()

        if any(x in text_lower for x in ["pii", "personal", "ssn", "social security"]):
            return ClassificationLevel.PII
        elif any(x in text_lower for x in ["phi", "health", "medical", "hipaa"]):
            return ClassificationLevel.PHI
        elif any(x in text_lower for x in ["pci", "credit card", "payment"]):
            return ClassificationLevel.PCI
        elif any(x in text_lower for x in ["restricted", "secret", "credential"]):
            return ClassificationLevel.RESTRICTED
        elif any(x in text_lower for x in ["confidential", "sensitive"]):
            return ClassificationLevel.CONFIDENTIAL
        elif any(x in text_lower for x in ["internal"]):
            return ClassificationLevel.INTERNAL

        return ClassificationLevel.INTERNAL
