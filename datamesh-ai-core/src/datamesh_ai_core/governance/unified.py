"""
Unified Governance - High-level API for governance operations.
"""

import logging
from typing import Optional

from .interfaces import (
    GovernanceProvider,
    UserContext,
    Resource,
    Permission,
    Classification,
    MaskingRule,
    RowFilter,
    AccessDecision,
)
from .config import GovernanceConfig, CloudProvider, DEFAULT_CLASSIFICATION_MAPPING
from .factory import GovernanceFactory

logger = logging.getLogger(__name__)


class UnifiedGovernance:
    """
    Unified Governance API.

    Provides a high-level interface for governance operations that
    abstracts away the underlying cloud provider. Supports multi-cloud
    scenarios with automatic fallback.
    """

    def __init__(
        self,
        config: Optional[GovernanceConfig] = None,
        auto_discover: bool = True,
    ):
        """
        Initialize unified governance.

        Args:
            config: Governance configuration
            auto_discover: Whether to auto-discover cloud provider
        """
        self.config = config or GovernanceConfig()
        self._primary_provider: Optional[GovernanceProvider] = None
        self._fallback_providers: list[GovernanceProvider] = []
        self._user_context: Optional[UserContext] = None
        self._classification_map = DEFAULT_CLASSIFICATION_MAPPING.copy()

        if self.config.custom_classifications:
            self._classification_map.update(self.config.custom_classifications)

        if auto_discover and self.config.auto_discover:
            detected = GovernanceFactory.auto_discover()
            if self.config.provider == CloudProvider.MULTI_CLOUD:
                self.config.primary_provider = detected
            else:
                self.config.provider = detected

    async def connect(self, credentials: Optional[dict] = None) -> None:
        """
        Connect to governance services.

        Args:
            credentials: Optional credentials for connection
        """
        # Create and connect primary provider
        self._primary_provider = GovernanceFactory.create(self.config)
        await self._primary_provider.connect(credentials or {})

        # Create and connect fallback providers
        for fallback_type in self.config.fallback_providers:
            try:
                fallback = GovernanceFactory.create(self.config, fallback_type)
                await fallback.connect(credentials or {})
                self._fallback_providers.append(fallback)
            except Exception as e:
                logger.warning(f"Could not connect fallback provider {fallback_type}: {e}")

        # Get user context
        if credentials:
            self._user_context = await self._primary_provider.get_user_context(credentials)

        logger.info(f"Connected to {self._primary_provider.provider_name} governance")

    async def disconnect(self) -> None:
        """Disconnect from all governance services."""
        if self._primary_provider:
            await self._primary_provider.disconnect()
            self._primary_provider = None

        for fallback in self._fallback_providers:
            await fallback.disconnect()
        self._fallback_providers.clear()

        self._user_context = None

    @property
    def provider(self) -> Optional[GovernanceProvider]:
        """Get the current governance provider."""
        return self._primary_provider

    @property
    def user(self) -> Optional[UserContext]:
        """Get the current user context."""
        return self._user_context

    async def set_user(self, credentials: dict) -> UserContext:
        """
        Set/update the user context.

        Args:
            credentials: User credentials

        Returns:
            Updated user context
        """
        if not self._primary_provider:
            raise RuntimeError("Not connected to governance services")

        self._user_context = await self._primary_provider.get_user_context(credentials)
        return self._user_context

    async def can_access(
        self,
        resource: Resource,
        action: str = "SELECT",
        user: Optional[UserContext] = None,
    ) -> bool:
        """
        Check if user can access a resource.

        Args:
            resource: The target resource
            action: The action to perform
            user: Optional user context (uses current user if not provided)

        Returns:
            True if access is allowed
        """
        permission = await self.check_permission(resource, action, user)
        return permission.decision == AccessDecision.ALLOW

    async def check_permission(
        self,
        resource: Resource,
        action: str,
        user: Optional[UserContext] = None,
    ) -> Permission:
        """
        Check permission for an action on a resource.

        Args:
            resource: The target resource
            action: The action to check
            user: Optional user context

        Returns:
            Permission object with decision and details
        """
        if not self._primary_provider:
            raise RuntimeError("Not connected to governance services")

        target_user = user or self._user_context
        if not target_user:
            raise ValueError("No user context available")

        try:
            permission = await self._primary_provider.check_permission(
                target_user, resource, action
            )

            # Audit the access check
            if self.config.audit_enabled:
                await self._primary_provider.audit_access(
                    target_user,
                    resource,
                    action,
                    permission.decision,
                    {"source": "unified_governance"},
                )

            return permission

        except Exception as e:
            logger.error(f"Error checking permission: {e}")

            # Try fallback providers
            for fallback in self._fallback_providers:
                try:
                    return await fallback.check_permission(target_user, resource, action)
                except Exception as fe:
                    logger.warning(f"Fallback provider error: {fe}")

            # Return deny on error
            return Permission(
                resource=resource,
                action=action,
                decision=AccessDecision.DENY,
                reason=f"Error checking permission: {str(e)}",
            )

    async def get_accessible_tables(
        self,
        database: Optional[str] = None,
        catalog: Optional[str] = None,
        user: Optional[UserContext] = None,
    ) -> list[Resource]:
        """
        Get list of tables the user can access.

        Args:
            database: Filter by database
            catalog: Filter by catalog
            user: Optional user context

        Returns:
            List of accessible table resources
        """
        if not self._primary_provider:
            raise RuntimeError("Not connected to governance services")

        target_user = user or self._user_context
        if not target_user:
            raise ValueError("No user context available")

        return await self._primary_provider.get_accessible_resources(
            target_user,
            resource_type="table",
            catalog=catalog,
            database=database,
        )

    async def get_classifications(
        self,
        resource: Resource,
        user: Optional[UserContext] = None,
    ) -> list[Classification]:
        """
        Get data classifications for a resource.

        Args:
            resource: The resource to get classifications for
            user: Optional user context

        Returns:
            List of classifications
        """
        if not self._primary_provider:
            raise RuntimeError("Not connected to governance services")

        target_user = user or self._user_context
        classifications = await self._primary_provider.get_classifications(
            resource, target_user
        )

        # Normalize classifications using mapping
        for classification in classifications:
            provider_key = f"{self._primary_provider.provider_name}:{classification.classification}"
            if provider_key in self._classification_map:
                classification.classification = self._classification_map[provider_key]

        return classifications

    async def get_sensitive_columns(
        self,
        resource: Resource,
        user: Optional[UserContext] = None,
    ) -> list[tuple[str, Classification]]:
        """
        Get list of sensitive columns for a table.

        Args:
            resource: The table resource
            user: Optional user context

        Returns:
            List of (column_name, classification) tuples
        """
        classifications = await self.get_classifications(resource, user)

        sensitive_columns = []
        for classification in classifications:
            if classification.resource.column:
                sensitive_columns.append((
                    classification.resource.column,
                    classification,
                ))

        return sensitive_columns

    async def get_masking_rules(
        self,
        resource: Resource,
        user: Optional[UserContext] = None,
    ) -> list[MaskingRule]:
        """
        Get masking rules for a resource.

        Args:
            resource: The resource
            user: Optional user context

        Returns:
            List of masking rules
        """
        if not self._primary_provider:
            raise RuntimeError("Not connected to governance services")

        target_user = user or self._user_context
        if not target_user:
            raise ValueError("No user context available")

        return await self._primary_provider.get_masking_rules(resource, target_user)

    async def get_row_filters(
        self,
        resource: Resource,
        user: Optional[UserContext] = None,
    ) -> list[RowFilter]:
        """
        Get row-level security filters for a resource.

        Args:
            resource: The resource
            user: Optional user context

        Returns:
            List of row filters
        """
        if not self._primary_provider:
            raise RuntimeError("Not connected to governance services")

        target_user = user or self._user_context
        if not target_user:
            raise ValueError("No user context available")

        return await self._primary_provider.get_row_filters(resource, target_user)

    async def prepare_governed_query(
        self,
        query: str,
        resources: list[Resource],
        user: Optional[UserContext] = None,
    ) -> tuple[str, list[str], list[str]]:
        """
        Prepare a query with governance applied.

        This method checks permissions, applies masking rules, and
        applies row filters to transform a query for safe execution.

        Args:
            query: The original SQL query
            resources: Resources referenced in the query
            user: Optional user context

        Returns:
            Tuple of (modified_query, warnings, errors)
        """
        if not self._primary_provider:
            raise RuntimeError("Not connected to governance services")

        target_user = user or self._user_context
        if not target_user:
            raise ValueError("No user context available")

        warnings = []
        errors = []

        for resource in resources:
            # Check access
            permission = await self.check_permission(resource, "SELECT", target_user)
            if permission.decision == AccessDecision.DENY:
                errors.append(
                    f"Access denied to {resource.fully_qualified_name}: {permission.reason}"
                )
                continue

            # Get masking rules
            masking_rules = await self.get_masking_rules(resource, target_user)
            for rule in masking_rules:
                warnings.append(
                    f"Column {resource.fully_qualified_name}.{rule.column} "
                    f"will be masked ({rule.mask_type})"
                )

            # Get row filters
            row_filters = await self.get_row_filters(resource, target_user)
            for rfilter in row_filters:
                warnings.append(
                    f"Row filter applied to {resource.fully_qualified_name}: "
                    f"{rfilter.filter_expression}"
                )

        # Note: Actual query transformation would happen here
        # For now, we return the original query with warnings/errors
        modified_query = query

        return modified_query, warnings, errors

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all connected providers.

        Returns:
            Dictionary of provider_name -> health_status
        """
        status = {}

        if self._primary_provider:
            status[self._primary_provider.provider_name] = (
                await self._primary_provider.health_check()
            )

        for fallback in self._fallback_providers:
            status[fallback.provider_name] = await fallback.health_check()

        return status


# Convenience function for quick setup
async def create_governance(
    config: Optional[GovernanceConfig] = None,
    credentials: Optional[dict] = None,
) -> UnifiedGovernance:
    """
    Create and connect a unified governance instance.

    Args:
        config: Governance configuration
        credentials: Connection credentials

    Returns:
        Connected UnifiedGovernance instance
    """
    governance = UnifiedGovernance(config)
    await governance.connect(credentials)
    return governance
