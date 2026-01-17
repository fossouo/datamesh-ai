"""
Governance Factory - Auto-discovery and creation of governance providers.
"""

import logging
import os
from typing import Optional

from .interfaces import GovernanceProvider
from .config import GovernanceConfig, CloudProvider
from .providers.aws import AWSGovernanceProvider
from .providers.azure import AzureGovernanceProvider
from .providers.gcp import GCPGovernanceProvider
from .providers.databricks import DatabricksGovernanceProvider
from .providers.local import LocalGovernanceProvider

logger = logging.getLogger(__name__)


class GovernanceFactory:
    """
    Factory for creating governance providers.

    Supports auto-discovery of cloud provider based on environment
    variables and available credentials.
    """

    # Provider registry
    _providers = {
        CloudProvider.AWS: AWSGovernanceProvider,
        CloudProvider.AZURE: AzureGovernanceProvider,
        CloudProvider.GCP: GCPGovernanceProvider,
        CloudProvider.DATABRICKS: DatabricksGovernanceProvider,
        CloudProvider.LOCAL: LocalGovernanceProvider,
    }

    @classmethod
    def create(
        cls,
        config: Optional[GovernanceConfig] = None,
        provider: Optional[CloudProvider] = None,
    ) -> GovernanceProvider:
        """
        Create a governance provider.

        Args:
            config: Governance configuration
            provider: Explicit provider to use (overrides config)

        Returns:
            Configured governance provider
        """
        if config is None:
            config = GovernanceConfig()

        target_provider = provider or config.provider

        if target_provider == CloudProvider.MULTI_CLOUD:
            # For multi-cloud, use the primary provider
            target_provider = config.primary_provider or cls.auto_discover()

        provider_class = cls._providers.get(target_provider)

        if not provider_class:
            raise ValueError(f"Unknown provider: {target_provider}")

        # Create provider with appropriate config
        if target_provider == CloudProvider.AWS:
            return provider_class(config.aws)
        elif target_provider == CloudProvider.AZURE:
            return provider_class(config.azure)
        elif target_provider == CloudProvider.GCP:
            return provider_class(config.gcp)
        elif target_provider == CloudProvider.DATABRICKS:
            return provider_class(config.databricks)
        else:
            return provider_class()

    @classmethod
    def auto_discover(cls) -> CloudProvider:
        """
        Auto-discover the cloud provider from environment.

        Detection order:
        1. Explicit DATAMESH_CLOUD_PROVIDER env var
        2. Databricks (DATABRICKS_HOST)
        3. AWS (AWS_ACCESS_KEY_ID, AWS_PROFILE, or IAM role)
        4. Azure (AZURE_TENANT_ID, AZURE_CLIENT_ID)
        5. GCP (GOOGLE_APPLICATION_CREDENTIALS, GOOGLE_CLOUD_PROJECT)
        6. Fall back to local

        Returns:
            Detected cloud provider
        """
        # Check explicit configuration
        explicit = os.environ.get("DATAMESH_CLOUD_PROVIDER", "").lower()
        if explicit:
            try:
                return CloudProvider(explicit)
            except ValueError:
                logger.warning(f"Unknown DATAMESH_CLOUD_PROVIDER: {explicit}")

        # Check Databricks first (most specific)
        if cls._detect_databricks():
            logger.info("Auto-discovered Databricks environment")
            return CloudProvider.DATABRICKS

        # Check AWS
        if cls._detect_aws():
            logger.info("Auto-discovered AWS environment")
            return CloudProvider.AWS

        # Check Azure
        if cls._detect_azure():
            logger.info("Auto-discovered Azure environment")
            return CloudProvider.AZURE

        # Check GCP
        if cls._detect_gcp():
            logger.info("Auto-discovered GCP environment")
            return CloudProvider.GCP

        # Fall back to local
        logger.info("No cloud provider detected, using local provider")
        return CloudProvider.LOCAL

    @classmethod
    def _detect_databricks(cls) -> bool:
        """Detect Databricks environment."""
        indicators = [
            "DATABRICKS_HOST",
            "DATABRICKS_TOKEN",
            "DATABRICKS_RUNTIME_VERSION",
        ]
        return any(os.environ.get(var) for var in indicators)

    @classmethod
    def _detect_aws(cls) -> bool:
        """Detect AWS environment."""
        # Direct credentials
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            return True

        # AWS Profile
        if os.environ.get("AWS_PROFILE"):
            return True

        # Check for IAM role (ECS, EC2, Lambda)
        if os.environ.get("AWS_EXECUTION_ENV"):
            return True

        # Check for AWS credentials file
        creds_file = os.path.expanduser("~/.aws/credentials")
        if os.path.exists(creds_file):
            return True

        # Check for EC2 metadata service
        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/",
                method="GET",
            )
            req.add_header("Connection", "close")

            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass

        return False

    @classmethod
    def _detect_azure(cls) -> bool:
        """Detect Azure environment."""
        indicators = [
            "AZURE_TENANT_ID",
            "AZURE_CLIENT_ID",
            "AZURE_SUBSCRIPTION_ID",
            "MSI_ENDPOINT",  # Managed Identity
            "IDENTITY_ENDPOINT",  # Managed Identity (newer)
        ]
        return any(os.environ.get(var) for var in indicators)

    @classmethod
    def _detect_gcp(cls) -> bool:
        """Detect GCP environment."""
        # Application Default Credentials file
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            return True

        # GCP project
        if os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT"):
            return True

        # Check for default credentials file
        default_creds = os.path.expanduser(
            "~/.config/gcloud/application_default_credentials.json"
        )
        if os.path.exists(default_creds):
            return True

        # Check for GCE metadata service
        try:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/",
                method="GET",
            )
            req.add_header("Metadata-Flavor", "Google")
            req.add_header("Connection", "close")

            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass

        return False

    @classmethod
    async def create_and_connect(
        cls,
        config: Optional[GovernanceConfig] = None,
        provider: Optional[CloudProvider] = None,
        connection_config: Optional[dict] = None,
    ) -> GovernanceProvider:
        """
        Create and connect a governance provider.

        Args:
            config: Governance configuration
            provider: Explicit provider to use
            connection_config: Additional connection configuration

        Returns:
            Connected governance provider
        """
        governance = cls.create(config, provider)
        await governance.connect(connection_config or {})
        return governance

    @classmethod
    def register_provider(
        cls,
        provider_type: CloudProvider,
        provider_class: type,
    ) -> None:
        """
        Register a custom governance provider.

        Args:
            provider_type: The provider type
            provider_class: The provider class (must implement GovernanceProvider)
        """
        if not issubclass(provider_class, GovernanceProvider):
            raise TypeError(
                f"Provider class must implement GovernanceProvider interface"
            )
        cls._providers[provider_type] = provider_class
        logger.info(f"Registered custom provider: {provider_type.value}")
