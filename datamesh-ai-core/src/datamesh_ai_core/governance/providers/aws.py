"""
AWS Governance Provider - Lake Formation, IAM, Glue, and Macie integration.
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
from ..config import AWSConfig

logger = logging.getLogger(__name__)


class AWSGovernanceProvider(GovernanceProvider):
    """
    AWS Governance Provider using Lake Formation, Glue, and optionally Macie.

    This provider integrates with:
    - AWS Lake Formation: Fine-grained access control, column masking, row filtering
    - AWS Glue Data Catalog: Schema and metadata
    - AWS IAM: Identity and roles
    - AWS Macie (optional): Data classification and sensitive data discovery
    """

    def __init__(self, config: Optional[AWSConfig] = None):
        self.config = config or AWSConfig()
        self._lf_client = None
        self._glue_client = None
        self._sts_client = None
        self._macie_client = None
        self._session = None

    @property
    def provider_name(self) -> str:
        return "aws"

    async def connect(self, config: dict) -> None:
        """
        Establish connection to AWS services.

        Args:
            config: AWS configuration including region, profile, or role
        """
        try:
            import boto3
            from botocore.config import Config as BotoConfig

            boto_config = BotoConfig(
                region_name=config.get("region", self.config.region),
                retries={"max_attempts": 3, "mode": "adaptive"},
            )

            session_kwargs = {}
            if config.get("profile") or self.config.profile:
                session_kwargs["profile_name"] = config.get("profile", self.config.profile)

            self._session = boto3.Session(**session_kwargs)

            # Assume role if specified
            role_arn = config.get("role_arn", self.config.role_arn)
            if role_arn:
                sts = self._session.client("sts", config=boto_config)
                assumed = sts.assume_role(
                    RoleArn=role_arn,
                    RoleSessionName="datamesh-ai-governance",
                )
                creds = assumed["Credentials"]
                self._session = boto3.Session(
                    aws_access_key_id=creds["AccessKeyId"],
                    aws_secret_access_key=creds["SecretAccessKey"],
                    aws_session_token=creds["SessionToken"],
                )

            # Initialize clients
            self._lf_client = self._session.client("lakeformation", config=boto_config)
            self._glue_client = self._session.client("glue", config=boto_config)
            self._sts_client = self._session.client("sts", config=boto_config)

            if self.config.macie_enabled:
                self._macie_client = self._session.client("macie2", config=boto_config)

            logger.info(f"Connected to AWS in region {self.config.region}")

        except ImportError:
            raise ImportError(
                "boto3 is required for AWS governance. "
                "Install with: pip install boto3"
            )

    async def disconnect(self) -> None:
        """Close AWS connections."""
        self._lf_client = None
        self._glue_client = None
        self._sts_client = None
        self._macie_client = None
        self._session = None

    async def health_check(self) -> bool:
        """Check if AWS services are accessible."""
        try:
            if self._sts_client:
                self._sts_client.get_caller_identity()
                return True
            return False
        except Exception as e:
            logger.warning(f"AWS health check failed: {e}")
            return False

    async def get_user_context(self, credentials: dict) -> UserContext:
        """
        Get user context from AWS credentials.

        Extracts identity from STS and IAM.
        """
        if not self._sts_client:
            raise RuntimeError("Not connected to AWS")

        identity = self._sts_client.get_caller_identity()

        user_id = identity["UserId"]
        arn = identity["Arn"]
        account = identity["Account"]

        # Parse ARN to extract user/role info
        arn_parts = arn.split(":")
        resource = arn_parts[-1] if len(arn_parts) > 5 else ""

        # Determine if this is a user, role, or federated identity
        roles = []
        groups = []

        if "/assumed-role/" in arn:
            # Assumed role
            role_name = resource.split("/")[1] if "/" in resource else resource
            roles.append(role_name)
        elif "/user/" in arn:
            # IAM user - fetch groups
            try:
                iam = self._session.client("iam")
                user_name = resource.split("/")[-1]
                response = iam.list_groups_for_user(UserName=user_name)
                groups = [g["GroupName"] for g in response.get("Groups", [])]
            except Exception as e:
                logger.warning(f"Could not fetch IAM groups: {e}")

        return UserContext(
            user_id=user_id,
            email=credentials.get("email"),
            display_name=resource.split("/")[-1] if "/" in resource else user_id,
            groups=groups,
            roles=roles,
            attributes={
                "arn": arn,
                "account": account,
            },
            provider="aws",
        )

    async def get_accessible_resources(
        self,
        user: UserContext,
        resource_type: Optional[str] = None,
        catalog: Optional[str] = None,
        database: Optional[str] = None,
    ) -> list[Resource]:
        """
        Get resources accessible to the user via Lake Formation.
        """
        if not self._lf_client:
            raise RuntimeError("Not connected to AWS")

        resources = []

        try:
            # List permissions for the principal
            paginator = self._lf_client.get_paginator("list_permissions")

            principal = {"DataLakePrincipalIdentifier": user.attributes.get("arn", user.user_id)}

            for page in paginator.paginate(Principal=principal):
                for perm in page.get("PrincipalResourcePermissions", []):
                    resource_info = perm.get("Resource", {})

                    # Handle different resource types
                    if "Table" in resource_info:
                        table_info = resource_info["Table"]
                        if database and table_info.get("DatabaseName") != database:
                            continue
                        resources.append(Resource(
                            resource_type="table",
                            catalog=table_info.get("CatalogId"),
                            database=table_info.get("DatabaseName"),
                            table=table_info.get("Name"),
                            provider="aws",
                        ))

                    elif "Database" in resource_info:
                        db_info = resource_info["Database"]
                        if database and db_info.get("Name") != database:
                            continue
                        if resource_type and resource_type != "database":
                            continue
                        resources.append(Resource(
                            resource_type="database",
                            catalog=db_info.get("CatalogId"),
                            database=db_info.get("Name"),
                            provider="aws",
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
        Check if user has permission via Lake Formation.
        """
        if not self._lf_client:
            raise RuntimeError("Not connected to AWS")

        # Map action to Lake Formation permissions
        lf_permission_map = {
            "SELECT": "SELECT",
            "INSERT": "INSERT",
            "DELETE": "DELETE",
            "UPDATE": "ALTER",
            "DESCRIBE": "DESCRIBE",
            "DROP": "DROP",
            "ALTER": "ALTER",
            "CREATE": "CREATE_TABLE",
        }

        lf_action = lf_permission_map.get(action.upper(), action.upper())

        try:
            principal = {"DataLakePrincipalIdentifier": user.attributes.get("arn", user.user_id)}

            # Build resource specification
            if resource.table:
                lf_resource = {
                    "Table": {
                        "DatabaseName": resource.database,
                        "Name": resource.table,
                    }
                }
                if resource.catalog:
                    lf_resource["Table"]["CatalogId"] = resource.catalog

            elif resource.database:
                lf_resource = {
                    "Database": {
                        "Name": resource.database,
                    }
                }
                if resource.catalog:
                    lf_resource["Database"]["CatalogId"] = resource.catalog
            else:
                # Default to catalog-level
                lf_resource = {"Catalog": {}}

            # Check permissions
            response = self._lf_client.list_permissions(
                Principal=principal,
                Resource=lf_resource,
            )

            # Check if the requested permission is granted
            for perm in response.get("PrincipalResourcePermissions", []):
                permissions = perm.get("Permissions", [])
                if lf_action in permissions or "ALL" in permissions:
                    return Permission(
                        resource=resource,
                        action=action,
                        decision=AccessDecision.ALLOW,
                        granted_by="lake_formation",
                    )

            return Permission(
                resource=resource,
                action=action,
                decision=AccessDecision.DENY,
                reason="No Lake Formation grant found",
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
        Get data classifications from Glue and optionally Macie.
        """
        classifications = []

        if not self._glue_client:
            return classifications

        try:
            # Get table metadata from Glue
            if resource.table and resource.database:
                response = self._glue_client.get_table(
                    DatabaseName=resource.database,
                    Name=resource.table,
                )

                table = response.get("Table", {})
                params = table.get("Parameters", {})

                # Check for Glue classification tags
                if "classification" in params:
                    classifications.append(Classification(
                        resource=resource,
                        classification=params["classification"],
                        level=self._map_classification_level(params["classification"]),
                        source="glue",
                    ))

                # Check column-level classifications
                for col in table.get("StorageDescriptor", {}).get("Columns", []):
                    col_params = col.get("Parameters", {})
                    if "classification" in col_params:
                        col_resource = Resource(
                            resource_type="column",
                            database=resource.database,
                            table=resource.table,
                            column=col["Name"],
                            provider="aws",
                        )
                        classifications.append(Classification(
                            resource=col_resource,
                            classification=col_params["classification"],
                            level=self._map_classification_level(col_params["classification"]),
                            source="glue",
                        ))

        except Exception as e:
            logger.warning(f"Error fetching Glue classifications: {e}")

        # Optionally get Macie findings
        if self._macie_client and self.config.macie_enabled:
            try:
                # Note: Macie integration would require S3 bucket path mapping
                pass
            except Exception as e:
                logger.warning(f"Error fetching Macie classifications: {e}")

        return classifications

    async def get_masking_rules(
        self,
        resource: Resource,
        user: UserContext,
    ) -> list[MaskingRule]:
        """
        Get column masking rules from Lake Formation.
        """
        if not self._lf_client:
            return []

        masking_rules = []

        try:
            # Lake Formation Data Cell Filters
            if resource.table and resource.database:
                response = self._lf_client.list_data_cells_filter(
                    Table={
                        "DatabaseName": resource.database,
                        "Name": resource.table,
                    }
                )

                for filter_item in response.get("DataCellsFilters", []):
                    # Check if user's roles match the filter
                    column_names = filter_item.get("ColumnNames", [])
                    for col in column_names:
                        masking_rules.append(MaskingRule(
                            resource=resource,
                            column=col,
                            mask_type="lf_cell_filter",
                            parameters={
                                "filter_name": filter_item.get("Name"),
                                "row_filter": filter_item.get("RowFilter", {}),
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
        Get row-level security filters from Lake Formation.
        """
        if not self._lf_client:
            return []

        row_filters = []

        try:
            if resource.table and resource.database:
                response = self._lf_client.list_data_cells_filter(
                    Table={
                        "DatabaseName": resource.database,
                        "Name": resource.table,
                    }
                )

                for filter_item in response.get("DataCellsFilters", []):
                    row_filter_expr = filter_item.get("RowFilter", {})
                    filter_expression = row_filter_expr.get("FilterExpression")

                    if filter_expression:
                        row_filters.append(RowFilter(
                            resource=resource,
                            filter_expression=filter_expression,
                            filter_type="sql",
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
        Log audit entry to CloudWatch or CloudTrail.
        """
        try:
            import json

            logs = self._session.client("logs")

            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user.user_id,
                "user_arn": user.attributes.get("arn"),
                "resource": resource.fully_qualified_name,
                "resource_type": resource.resource_type,
                "action": action,
                "decision": decision.value,
                "trace_id": trace_id,
                "details": details or {},
            }

            # Log to CloudWatch (datamesh-ai-governance log group)
            log_group = "/datamesh-ai/governance"
            log_stream = datetime.utcnow().strftime("%Y/%m/%d")

            try:
                logs.create_log_group(logGroupName=log_group)
            except logs.exceptions.ResourceAlreadyExistsException:
                pass

            try:
                logs.create_log_stream(logGroupName=log_group, logStreamName=log_stream)
            except logs.exceptions.ResourceAlreadyExistsException:
                pass

            logs.put_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                logEvents=[{
                    "timestamp": int(datetime.utcnow().timestamp() * 1000),
                    "message": json.dumps(log_entry),
                }],
            )

        except Exception as e:
            logger.error(f"Error logging audit entry: {e}")

    def _map_classification_level(self, classification: str) -> ClassificationLevel:
        """Map AWS classification to unified level."""
        classification_lower = classification.lower()

        if any(x in classification_lower for x in ["pii", "personal", "ssn", "email"]):
            return ClassificationLevel.PII
        elif any(x in classification_lower for x in ["phi", "health", "medical"]):
            return ClassificationLevel.PHI
        elif any(x in classification_lower for x in ["pci", "card", "payment"]):
            return ClassificationLevel.PCI
        elif any(x in classification_lower for x in ["restricted", "secret", "credential"]):
            return ClassificationLevel.RESTRICTED
        elif any(x in classification_lower for x in ["confidential", "financial"]):
            return ClassificationLevel.CONFIDENTIAL
        elif any(x in classification_lower for x in ["internal"]):
            return ClassificationLevel.INTERNAL
        else:
            return ClassificationLevel.PUBLIC
