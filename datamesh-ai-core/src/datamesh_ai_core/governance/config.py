"""
Governance Configuration - Configuration models for governance providers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DATABRICKS = "databricks"
    MULTI_CLOUD = "multi-cloud"
    LOCAL = "local"  # For testing/development


class CredentialType(Enum):
    """Credential types."""
    ENVIRONMENT = "environment"
    FILE = "file"
    IAM_ROLE = "iam_role"
    SERVICE_ACCOUNT = "service_account"
    MANAGED_IDENTITY = "managed_identity"
    TOKEN = "token"


@dataclass
class AWSConfig:
    """AWS-specific configuration."""
    region: str = "us-east-1"
    profile: Optional[str] = None
    role_arn: Optional[str] = None
    lake_formation_enabled: bool = True
    glue_catalog_id: Optional[str] = None
    macie_enabled: bool = False
    s3_access_point: Optional[str] = None


@dataclass
class AzureConfig:
    """Azure-specific configuration."""
    tenant_id: Optional[str] = None
    subscription_id: Optional[str] = None
    purview_account: Optional[str] = None
    purview_endpoint: Optional[str] = None
    synapse_workspace: Optional[str] = None
    storage_account: Optional[str] = None
    use_managed_identity: bool = False


@dataclass
class GCPConfig:
    """GCP-specific configuration."""
    project_id: Optional[str] = None
    location: str = "us"
    dataplex_lake: Optional[str] = None
    datacatalog_enabled: bool = True
    dlp_enabled: bool = False
    service_account_file: Optional[str] = None


@dataclass
class DatabricksConfig:
    """Databricks-specific configuration."""
    workspace_url: Optional[str] = None
    token: Optional[str] = None
    unity_catalog_enabled: bool = True
    metastore_id: Optional[str] = None
    default_catalog: Optional[str] = None
    use_workspace_acls: bool = True


@dataclass
class GovernanceConfig:
    """
    Main governance configuration.

    This configuration determines which cloud provider(s) to use
    and how to connect to them.
    """
    provider: CloudProvider = CloudProvider.AWS
    credential_type: CredentialType = CredentialType.ENVIRONMENT
    auto_discover: bool = True

    # Provider-specific configs
    aws: Optional[AWSConfig] = None
    azure: Optional[AzureConfig] = None
    gcp: Optional[GCPConfig] = None
    databricks: Optional[DatabricksConfig] = None

    # Multi-cloud settings
    primary_provider: Optional[CloudProvider] = None
    fallback_providers: list[CloudProvider] = field(default_factory=list)

    # Classification mapping
    classification_taxonomy: str = "default"  # default, custom, provider-native
    custom_classifications: dict[str, str] = field(default_factory=dict)

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    # Audit
    audit_enabled: bool = True
    audit_sink: str = "native"  # native, s3, cloudwatch, etc.

    @classmethod
    def from_dict(cls, data: dict) -> "GovernanceConfig":
        """Create config from dictionary."""
        provider_str = data.get("provider", "aws")
        provider = CloudProvider(provider_str) if isinstance(provider_str, str) else provider_str

        cred_type_str = data.get("credential_type", data.get("credentialType", "environment"))
        cred_type = CredentialType(cred_type_str) if isinstance(cred_type_str, str) else cred_type_str

        config = cls(
            provider=provider,
            credential_type=cred_type,
            auto_discover=data.get("auto_discover", data.get("autoDiscover", True)),
            classification_taxonomy=data.get("classification_taxonomy", data.get("classificationTaxonomy", "default")),
            custom_classifications=data.get("custom_classifications", data.get("customClassifications", {})),
            cache_enabled=data.get("cache_enabled", data.get("cacheEnabled", True)),
            cache_ttl_seconds=data.get("cache_ttl_seconds", data.get("cacheTtlSeconds", 300)),
            audit_enabled=data.get("audit_enabled", data.get("auditEnabled", True)),
            audit_sink=data.get("audit_sink", data.get("auditSink", "native")),
        )

        # Parse provider-specific configs
        if "aws" in data:
            config.aws = AWSConfig(**data["aws"])
        if "azure" in data:
            config.azure = AzureConfig(**data["azure"])
        if "gcp" in data:
            config.gcp = GCPConfig(**data["gcp"])
        if "databricks" in data:
            config.databricks = DatabricksConfig(**data["databricks"])

        # Multi-cloud settings
        if "primary_provider" in data or "primaryProvider" in data:
            primary = data.get("primary_provider", data.get("primaryProvider"))
            config.primary_provider = CloudProvider(primary) if primary else None

        fallback = data.get("fallback_providers", data.get("fallbackProviders", []))
        config.fallback_providers = [CloudProvider(p) for p in fallback]

        return config

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        result = {
            "provider": self.provider.value,
            "credentialType": self.credential_type.value,
            "autoDiscover": self.auto_discover,
            "classificationTaxonomy": self.classification_taxonomy,
            "customClassifications": self.custom_classifications,
            "cacheEnabled": self.cache_enabled,
            "cacheTtlSeconds": self.cache_ttl_seconds,
            "auditEnabled": self.audit_enabled,
            "auditSink": self.audit_sink,
        }

        if self.aws:
            result["aws"] = {
                "region": self.aws.region,
                "profile": self.aws.profile,
                "roleArn": self.aws.role_arn,
                "lakeFormationEnabled": self.aws.lake_formation_enabled,
                "glueCatalogId": self.aws.glue_catalog_id,
                "macieEnabled": self.aws.macie_enabled,
            }

        if self.azure:
            result["azure"] = {
                "tenantId": self.azure.tenant_id,
                "subscriptionId": self.azure.subscription_id,
                "purviewAccount": self.azure.purview_account,
                "purviewEndpoint": self.azure.purview_endpoint,
                "useManagedIdentity": self.azure.use_managed_identity,
            }

        if self.gcp:
            result["gcp"] = {
                "projectId": self.gcp.project_id,
                "location": self.gcp.location,
                "dataplexLake": self.gcp.dataplex_lake,
                "datacatalogEnabled": self.gcp.datacatalog_enabled,
                "dlpEnabled": self.gcp.dlp_enabled,
            }

        if self.databricks:
            result["databricks"] = {
                "workspaceUrl": self.databricks.workspace_url,
                "unityCatalogEnabled": self.databricks.unity_catalog_enabled,
                "metastoreId": self.databricks.metastore_id,
                "defaultCatalog": self.databricks.default_catalog,
            }

        return result


# Default classification taxonomy mapping
DEFAULT_CLASSIFICATION_MAPPING = {
    # AWS Macie to unified
    "aws:CREDENTIALS": "RESTRICTED",
    "aws:FINANCIAL": "CONFIDENTIAL",
    "aws:PERSONAL_INFORMATION": "PII",

    # Azure Purview to unified
    "azure:Government ID Number": "PII",
    "azure:Financial": "CONFIDENTIAL",
    "azure:Confidential": "CONFIDENTIAL",

    # GCP DLP to unified
    "gcp:PERSON_NAME": "PII",
    "gcp:PHONE_NUMBER": "PII",
    "gcp:EMAIL_ADDRESS": "PII",
    "gcp:CREDIT_CARD_NUMBER": "PCI",
    "gcp:US_SOCIAL_SECURITY_NUMBER": "PII",

    # Databricks Unity Catalog to unified
    "databricks:PERSONALLY_IDENTIFIABLE_INFORMATION": "PII",
    "databricks:PAYMENT_CARD_DATA": "PCI",
    "databricks:HEALTH_INFORMATION": "PHI",
}
