"""
GCP Governance Provider - Dataplex, Data Catalog, DLP, and IAM integration.
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
from ..config import GCPConfig

logger = logging.getLogger(__name__)


class GCPGovernanceProvider(GovernanceProvider):
    """
    GCP Governance Provider using Dataplex, Data Catalog, DLP, and IAM.

    This provider integrates with:
    - Google Cloud Dataplex: Data governance and management
    - Google Cloud Data Catalog: Metadata and classification
    - Cloud DLP: Sensitive data discovery (optional)
    - Google Cloud IAM: Identity and access management
    """

    def __init__(self, config: Optional[GCPConfig] = None):
        self.config = config or GCPConfig()
        self._credentials = None
        self._dataplex_client = None
        self._datacatalog_client = None
        self._dlp_client = None
        self._iam_client = None
        self._connected = False

    @property
    def provider_name(self) -> str:
        return "gcp"

    async def connect(self, config: dict) -> None:
        """
        Establish connection to GCP services.

        Args:
            config: GCP configuration including project_id, location
        """
        try:
            from google.auth import default
            from google.auth.credentials import Credentials
            from google.oauth2 import service_account

            project_id = config.get("project_id", self.config.project_id)

            # Get credentials
            service_account_file = config.get(
                "service_account_file",
                self.config.service_account_file,
            )

            if service_account_file:
                self._credentials = service_account.Credentials.from_service_account_file(
                    service_account_file,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            else:
                self._credentials, detected_project = default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                if not project_id:
                    project_id = detected_project

            self.config.project_id = project_id

            # Initialize Data Catalog client
            if config.get("datacatalog_enabled", self.config.datacatalog_enabled):
                try:
                    from google.cloud import datacatalog_v1

                    self._datacatalog_client = datacatalog_v1.DataCatalogClient(
                        credentials=self._credentials
                    )
                except ImportError:
                    logger.warning(
                        "google-cloud-datacatalog not installed. "
                        "Install with: pip install google-cloud-datacatalog"
                    )

            # Initialize Dataplex client
            dataplex_lake = config.get("dataplex_lake", self.config.dataplex_lake)
            if dataplex_lake:
                try:
                    from google.cloud import dataplex_v1

                    self._dataplex_client = dataplex_v1.DataplexServiceClient(
                        credentials=self._credentials
                    )
                except ImportError:
                    logger.warning(
                        "google-cloud-dataplex not installed. "
                        "Install with: pip install google-cloud-dataplex"
                    )

            # Initialize DLP client (optional)
            if config.get("dlp_enabled", self.config.dlp_enabled):
                try:
                    from google.cloud import dlp_v2

                    self._dlp_client = dlp_v2.DlpServiceClient(
                        credentials=self._credentials
                    )
                except ImportError:
                    logger.warning(
                        "google-cloud-dlp not installed. "
                        "Install with: pip install google-cloud-dlp"
                    )

            self._connected = True
            logger.info(f"Connected to GCP project {project_id}")

        except ImportError:
            raise ImportError(
                "google-auth is required for GCP governance. "
                "Install with: pip install google-auth"
            )

    async def disconnect(self) -> None:
        """Close GCP connections."""
        self._credentials = None
        self._dataplex_client = None
        self._datacatalog_client = None
        self._dlp_client = None
        self._iam_client = None
        self._connected = False

    async def health_check(self) -> bool:
        """Check if GCP services are accessible."""
        if not self._connected:
            return False

        try:
            # Try to refresh credentials
            if self._credentials:
                from google.auth.transport import requests

                self._credentials.refresh(requests.Request())
                return True
            return False
        except Exception as e:
            logger.warning(f"GCP health check failed: {e}")
            return False

    async def get_user_context(self, credentials: dict) -> UserContext:
        """
        Get user context from GCP credentials.

        Extracts identity from service account or user credentials.
        """
        if not self._credentials:
            raise RuntimeError("Not connected to GCP")

        try:
            # Get service account or user email
            if hasattr(self._credentials, "service_account_email"):
                email = self._credentials.service_account_email
                user_id = email
                display_name = email.split("@")[0]
            else:
                # For user credentials, need to call OAuth2 userinfo
                email = credentials.get("email", "unknown")
                user_id = credentials.get("sub", email)
                display_name = credentials.get("name", email)

            # Get IAM roles for the identity
            roles = await self._fetch_iam_roles(email)

            return UserContext(
                user_id=user_id,
                email=email,
                display_name=display_name,
                groups=[],
                roles=roles,
                attributes={
                    "project_id": self.config.project_id,
                },
                provider="gcp",
            )

        except Exception as e:
            logger.warning(f"Error getting user context: {e}")
            return UserContext(
                user_id="gcp-user",
                provider="gcp",
            )

    async def _fetch_iam_roles(self, member: str) -> list[str]:
        """Fetch IAM roles for a member."""
        roles = []

        try:
            from google.cloud import resourcemanager_v3

            client = resourcemanager_v3.ProjectsClient(credentials=self._credentials)

            # Get IAM policy for the project
            policy = client.get_iam_policy(
                resource=f"projects/{self.config.project_id}"
            )

            member_id = f"serviceAccount:{member}" if "@" in member and ".iam.gserviceaccount.com" in member else f"user:{member}"

            for binding in policy.bindings:
                if member_id in binding.members:
                    roles.append(binding.role)

        except ImportError:
            logger.warning("google-cloud-resource-manager not installed")
        except Exception as e:
            logger.warning(f"Could not fetch IAM roles: {e}")

        return roles

    async def get_accessible_resources(
        self,
        user: UserContext,
        resource_type: Optional[str] = None,
        catalog: Optional[str] = None,
        database: Optional[str] = None,
    ) -> list[Resource]:
        """
        Get resources accessible to the user via Data Catalog.
        """
        resources = []

        if not self._datacatalog_client:
            logger.warning("Data Catalog client not configured")
            return resources

        try:
            from google.cloud import datacatalog_v1

            # Search for entries in Data Catalog
            scope = datacatalog_v1.SearchCatalogRequest.Scope(
                include_project_ids=[self.config.project_id],
            )

            query = "*"
            if database:
                query = f'parent:{database}'

            request = datacatalog_v1.SearchCatalogRequest(
                scope=scope,
                query=query,
                page_size=100,
            )

            for result in self._datacatalog_client.search_catalog(request=request):
                entry_type = result.search_result_type.name

                if entry_type == "ENTRY" and result.linked_resource:
                    # Parse BigQuery resource path
                    linked_resource = result.linked_resource

                    if "bigquery.googleapis.com" in linked_resource:
                        parts = linked_resource.split("/")
                        # Format: //bigquery.googleapis.com/projects/PROJECT/datasets/DATASET/tables/TABLE
                        if "tables" in parts:
                            table_idx = parts.index("tables")
                            dataset_idx = parts.index("datasets")
                            resources.append(Resource(
                                resource_type="table",
                                database=parts[dataset_idx + 1],
                                table=parts[table_idx + 1],
                                uri=linked_resource,
                                provider="gcp",
                                metadata={
                                    "datacatalog_entry": result.relative_resource_name,
                                },
                            ))
                        elif "datasets" in parts:
                            if not resource_type or resource_type == "database":
                                dataset_idx = parts.index("datasets")
                                resources.append(Resource(
                                    resource_type="database",
                                    database=parts[dataset_idx + 1],
                                    uri=linked_resource,
                                    provider="gcp",
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
        Check if user has permission via GCP IAM.
        """
        try:
            from google.cloud import resourcemanager_v3

            # Map action to IAM permission
            permission_map = {
                "SELECT": "bigquery.tables.getData",
                "INSERT": "bigquery.tables.updateData",
                "DELETE": "bigquery.tables.delete",
                "UPDATE": "bigquery.tables.updateData",
                "DESCRIBE": "bigquery.tables.get",
                "CREATE": "bigquery.tables.create",
            }

            iam_permission = permission_map.get(
                action.upper(),
                f"bigquery.tables.{action.lower()}",
            )

            # Check if user has the permission
            # This requires testIamPermissions API
            client = resourcemanager_v3.ProjectsClient(credentials=self._credentials)

            request = resourcemanager_v3.TestIamPermissionsRequest(
                resource=f"projects/{self.config.project_id}",
                permissions=[iam_permission],
            )

            response = client.test_iam_permissions(request=request)

            if iam_permission in response.permissions:
                return Permission(
                    resource=resource,
                    action=action,
                    decision=AccessDecision.ALLOW,
                    granted_by="gcp_iam",
                )

            return Permission(
                resource=resource,
                action=action,
                decision=AccessDecision.DENY,
                reason="IAM permission not granted",
            )

        except ImportError:
            logger.warning("google-cloud-resource-manager not installed")
            return Permission(
                resource=resource,
                action=action,
                decision=AccessDecision.ALLOW,
                reason="IAM check skipped - SDK not installed",
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
        Get data classifications from Data Catalog and optionally DLP.
        """
        classifications = []

        # Get classifications from Data Catalog tags
        if self._datacatalog_client and resource.metadata.get("datacatalog_entry"):
            try:
                entry_name = resource.metadata["datacatalog_entry"]
                entry = self._datacatalog_client.get_entry(name=entry_name)

                # Get tags for the entry
                tags = self._datacatalog_client.list_tags(parent=entry.name)

                for tag in tags:
                    for field_name, field_value in tag.fields.items():
                        if "classification" in field_name.lower():
                            value = (
                                field_value.string_value
                                or field_value.enum_value.display_name
                            )
                            classifications.append(Classification(
                                resource=resource,
                                classification=value,
                                level=self._map_gcp_classification(value),
                                source="datacatalog",
                                tags=[tag.template],
                            ))

            except Exception as e:
                logger.warning(f"Error fetching Data Catalog classifications: {e}")

        # Get classifications from DLP if enabled
        if self._dlp_client and self.config.dlp_enabled:
            try:
                # Note: DLP inspection would require job results
                # This is a placeholder for DLP integration
                pass
            except Exception as e:
                logger.warning(f"Error fetching DLP classifications: {e}")

        return classifications

    async def get_masking_rules(
        self,
        resource: Resource,
        user: UserContext,
    ) -> list[MaskingRule]:
        """
        Get masking rules from BigQuery column-level access control.
        """
        # BigQuery column-level security requires direct SQL queries
        # to information_schema
        return []

    async def get_row_filters(
        self,
        resource: Resource,
        user: UserContext,
    ) -> list[RowFilter]:
        """
        Get row-level security from BigQuery row-level access policies.
        """
        # BigQuery row-level security requires direct SQL queries
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
        Log audit entry to Cloud Logging.
        """
        try:
            from google.cloud import logging as cloud_logging

            client = cloud_logging.Client(
                credentials=self._credentials,
                project=self.config.project_id,
            )

            logger_name = "datamesh-ai-governance"
            cloud_logger = client.logger(logger_name)

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

            cloud_logger.log_struct(
                audit_record,
                severity="INFO",
            )

        except ImportError:
            logger.warning("google-cloud-logging not installed")
        except Exception as e:
            logger.error(f"Error logging audit entry: {e}")

    def _map_gcp_classification(self, classification: str) -> ClassificationLevel:
        """Map GCP classification to unified level."""
        classification_upper = classification.upper()

        # DLP InfoType mappings
        dlp_mappings = {
            "PERSON_NAME": ClassificationLevel.PII,
            "PHONE_NUMBER": ClassificationLevel.PII,
            "EMAIL_ADDRESS": ClassificationLevel.PII,
            "STREET_ADDRESS": ClassificationLevel.PII,
            "CREDIT_CARD_NUMBER": ClassificationLevel.PCI,
            "US_SOCIAL_SECURITY_NUMBER": ClassificationLevel.PII,
            "US_PASSPORT": ClassificationLevel.PII,
            "DATE_OF_BIRTH": ClassificationLevel.PII,
            "MEDICAL_RECORD_NUMBER": ClassificationLevel.PHI,
            "HEALTH_INSURANCE_ID": ClassificationLevel.PHI,
            "AWS_CREDENTIALS": ClassificationLevel.RESTRICTED,
            "GCP_CREDENTIALS": ClassificationLevel.RESTRICTED,
            "PASSWORD": ClassificationLevel.RESTRICTED,
        }

        if classification_upper in dlp_mappings:
            return dlp_mappings[classification_upper]

        if any(x in classification_upper for x in ["PII", "PERSONAL"]):
            return ClassificationLevel.PII
        elif any(x in classification_upper for x in ["PHI", "HEALTH"]):
            return ClassificationLevel.PHI
        elif any(x in classification_upper for x in ["PCI", "CARD", "PAYMENT"]):
            return ClassificationLevel.PCI
        elif any(x in classification_upper for x in ["RESTRICTED", "SECRET"]):
            return ClassificationLevel.RESTRICTED
        elif any(x in classification_upper for x in ["CONFIDENTIAL"]):
            return ClassificationLevel.CONFIDENTIAL

        return ClassificationLevel.INTERNAL
