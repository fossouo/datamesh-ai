"""
Azure Governance Provider - Purview, RBAC, and Defender for Cloud integration.
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
from ..config import AzureConfig

logger = logging.getLogger(__name__)


class AzureGovernanceProvider(GovernanceProvider):
    """
    Azure Governance Provider using Purview, RBAC, and Defender for Cloud.

    This provider integrates with:
    - Microsoft Purview: Data governance, classification, lineage
    - Azure RBAC: Identity and access management
    - Synapse Analytics: Row-level and column-level security
    - Defender for Cloud: Security classifications (optional)
    """

    def __init__(self, config: Optional[AzureConfig] = None):
        self.config = config or AzureConfig()
        self._credential = None
        self._purview_client = None
        self._graph_client = None
        self._connected = False

    @property
    def provider_name(self) -> str:
        return "azure"

    async def connect(self, config: dict) -> None:
        """
        Establish connection to Azure services.

        Args:
            config: Azure configuration including tenant_id, subscription_id
        """
        try:
            from azure.identity import (
                DefaultAzureCredential,
                ManagedIdentityCredential,
                ClientSecretCredential,
            )

            # Determine credential type
            if config.get("use_managed_identity") or self.config.use_managed_identity:
                self._credential = ManagedIdentityCredential()
            elif config.get("client_id") and config.get("client_secret"):
                self._credential = ClientSecretCredential(
                    tenant_id=config.get("tenant_id", self.config.tenant_id),
                    client_id=config["client_id"],
                    client_secret=config["client_secret"],
                )
            else:
                self._credential = DefaultAzureCredential()

            # Initialize Purview client if configured
            purview_account = config.get("purview_account", self.config.purview_account)
            if purview_account:
                try:
                    from azure.purview.catalog import PurviewCatalogClient
                    from azure.purview.scanning import PurviewScanningClient

                    endpoint = (
                        config.get("purview_endpoint", self.config.purview_endpoint)
                        or f"https://{purview_account}.purview.azure.com"
                    )

                    self._purview_client = PurviewCatalogClient(
                        endpoint=endpoint,
                        credential=self._credential,
                    )
                except ImportError:
                    logger.warning(
                        "azure-purview-catalog not installed. "
                        "Install with: pip install azure-purview-catalog"
                    )

            self._connected = True
            logger.info("Connected to Azure services")

        except ImportError:
            raise ImportError(
                "azure-identity is required for Azure governance. "
                "Install with: pip install azure-identity"
            )

    async def disconnect(self) -> None:
        """Close Azure connections."""
        self._credential = None
        self._purview_client = None
        self._graph_client = None
        self._connected = False

    async def health_check(self) -> bool:
        """Check if Azure services are accessible."""
        if not self._connected:
            return False

        try:
            # Try to get a token to verify credentials
            if self._credential:
                self._credential.get_token("https://management.azure.com/.default")
                return True
            return False
        except Exception as e:
            logger.warning(f"Azure health check failed: {e}")
            return False

    async def get_user_context(self, credentials: dict) -> UserContext:
        """
        Get user context from Azure AD.

        Extracts identity from Azure AD / Entra ID.
        """
        if not self._credential:
            raise RuntimeError("Not connected to Azure")

        try:
            from azure.identity import DefaultAzureCredential
            import jwt

            # Get access token for Graph API
            token = self._credential.get_token("https://graph.microsoft.com/.default")

            # Decode token to get user info (without verification for claims extraction)
            claims = jwt.decode(token.token, options={"verify_signature": False})

            user_id = claims.get("oid", claims.get("sub", ""))
            email = claims.get("preferred_username", claims.get("email", ""))
            name = claims.get("name", "")
            roles = claims.get("roles", [])
            groups = claims.get("groups", [])

            # If groups claim is not present, we may need to call Graph API
            if not groups and credentials.get("fetch_groups", True):
                groups = await self._fetch_user_groups(user_id)

            return UserContext(
                user_id=user_id,
                email=email,
                display_name=name,
                groups=groups,
                roles=roles,
                attributes={
                    "tenant_id": claims.get("tid"),
                    "app_id": claims.get("appid"),
                    "scp": claims.get("scp", ""),
                },
                provider="azure",
            )

        except ImportError:
            logger.warning("PyJWT not installed, returning minimal user context")
            return UserContext(
                user_id="azure-user",
                provider="azure",
            )

    async def _fetch_user_groups(self, user_id: str) -> list[str]:
        """Fetch user groups from Microsoft Graph API."""
        try:
            from azure.identity import DefaultAzureCredential
            import httpx

            token = self._credential.get_token("https://graph.microsoft.com/.default")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://graph.microsoft.com/v1.0/users/{user_id}/memberOf",
                    headers={"Authorization": f"Bearer {token.token}"},
                )

                if response.status_code == 200:
                    data = response.json()
                    return [
                        g.get("displayName", g.get("id"))
                        for g in data.get("value", [])
                        if g.get("@odata.type") == "#microsoft.graph.group"
                    ]

        except Exception as e:
            logger.warning(f"Could not fetch user groups: {e}")

        return []

    async def get_accessible_resources(
        self,
        user: UserContext,
        resource_type: Optional[str] = None,
        catalog: Optional[str] = None,
        database: Optional[str] = None,
    ) -> list[Resource]:
        """
        Get resources accessible to the user via Purview.
        """
        resources = []

        if not self._purview_client:
            logger.warning("Purview client not configured")
            return resources

        try:
            # Search Purview catalog for accessible assets
            search_request = {
                "keywords": "*",
                "limit": 100,
            }

            if database:
                search_request["filter"] = {
                    "and": [{"entityType": "azure_datalake_gen2_path"}]
                }

            response = self._purview_client.discovery.query(search_request)

            for entity in response.get("value", []):
                entity_type = entity.get("entityType", "")
                qualified_name = entity.get("qualifiedName", "")

                # Map Purview entity types to our resource types
                if entity_type in ["azure_sql_table", "azure_synapse_sql_table"]:
                    parts = qualified_name.split("/")
                    resources.append(Resource(
                        resource_type="table",
                        database=parts[-2] if len(parts) > 1 else None,
                        table=parts[-1] if parts else qualified_name,
                        uri=qualified_name,
                        provider="azure",
                        metadata={
                            "purview_id": entity.get("id"),
                            "entity_type": entity_type,
                        },
                    ))

                elif entity_type == "azure_sql_db":
                    if not resource_type or resource_type == "database":
                        resources.append(Resource(
                            resource_type="database",
                            database=entity.get("name", qualified_name),
                            uri=qualified_name,
                            provider="azure",
                        ))

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
        Check if user has permission via Azure RBAC.

        Note: Full RBAC check requires Azure Resource Manager integration.
        """
        try:
            from azure.mgmt.authorization import AuthorizationManagementClient

            if not self.config.subscription_id:
                return Permission(
                    resource=resource,
                    action=action,
                    decision=AccessDecision.ALLOW,
                    reason="No subscription configured, allowing by default",
                )

            auth_client = AuthorizationManagementClient(
                credential=self._credential,
                subscription_id=self.config.subscription_id,
            )

            # Check role assignments for the user
            # This is a simplified check - full implementation would need resource scope
            assignments = list(auth_client.role_assignments.list_for_scope(
                scope=f"/subscriptions/{self.config.subscription_id}",
                filter=f"principalId eq '{user.user_id}'",
            ))

            if assignments:
                return Permission(
                    resource=resource,
                    action=action,
                    decision=AccessDecision.ALLOW,
                    granted_by="azure_rbac",
                )

            return Permission(
                resource=resource,
                action=action,
                decision=AccessDecision.DENY,
                reason="No RBAC role assignment found",
            )

        except ImportError:
            logger.warning("azure-mgmt-authorization not installed")
            return Permission(
                resource=resource,
                action=action,
                decision=AccessDecision.ALLOW,
                reason="RBAC check skipped - SDK not installed",
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
        Get data classifications from Purview.
        """
        classifications = []

        if not self._purview_client:
            return classifications

        try:
            # Get entity by qualified name
            if resource.uri:
                entity = self._purview_client.entity.get_by_unique_attributes(
                    type_name="azure_sql_table",
                    attr_qualified_name=resource.uri,
                )

                # Extract classifications
                for classification in entity.get("entity", {}).get("classifications", []):
                    type_name = classification.get("typeName", "")
                    classifications.append(Classification(
                        resource=resource,
                        classification=type_name,
                        level=self._map_purview_classification(type_name),
                        source="purview",
                        tags=classification.get("attributes", {}).get("tags", []),
                    ))

        except Exception as e:
            logger.warning(f"Error fetching Purview classifications: {e}")

        return classifications

    async def get_masking_rules(
        self,
        resource: Resource,
        user: UserContext,
    ) -> list[MaskingRule]:
        """
        Get column masking rules from Synapse/SQL dynamic data masking.
        """
        # Note: This would require Synapse SQL connection to query
        # sys.masked_columns or similar
        return []

    async def get_row_filters(
        self,
        resource: Resource,
        user: UserContext,
    ) -> list[RowFilter]:
        """
        Get row-level security filters from Synapse/SQL.
        """
        # Note: This would require Synapse SQL connection to query
        # security policies
        return []

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
        Log audit entry to Azure Monitor / Log Analytics.
        """
        try:
            # Prepare audit record
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

            # Note: Full implementation would send to Azure Monitor
            # using azure-monitor-ingestion client
            logger.info(f"Audit: {audit_record}")

        except Exception as e:
            logger.error(f"Error logging audit entry: {e}")

    def _map_purview_classification(self, classification: str) -> ClassificationLevel:
        """Map Purview classification to unified level."""
        classification_lower = classification.lower()

        purview_mappings = {
            "microsoft.personal": ClassificationLevel.PII,
            "microsoft.financial": ClassificationLevel.CONFIDENTIAL,
            "microsoft.government": ClassificationLevel.PII,
            "microsoft.security": ClassificationLevel.RESTRICTED,
            "microsoft.health": ClassificationLevel.PHI,
        }

        for prefix, level in purview_mappings.items():
            if classification_lower.startswith(prefix):
                return level

        if any(x in classification_lower for x in ["pii", "personal", "ssn", "email"]):
            return ClassificationLevel.PII
        elif any(x in classification_lower for x in ["phi", "health"]):
            return ClassificationLevel.PHI
        elif any(x in classification_lower for x in ["pci", "card", "payment"]):
            return ClassificationLevel.PCI
        elif any(x in classification_lower for x in ["confidential"]):
            return ClassificationLevel.CONFIDENTIAL

        return ClassificationLevel.INTERNAL
