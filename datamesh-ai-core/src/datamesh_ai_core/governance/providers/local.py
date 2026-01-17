"""
Local Governance Provider - For testing and development.
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

logger = logging.getLogger(__name__)


class LocalGovernanceProvider(GovernanceProvider):
    """
    Local Governance Provider for testing and development.

    This provider uses in-memory storage for permissions, classifications,
    and other governance data. Useful for local development without
    cloud dependencies.
    """

    def __init__(self):
        self._users: dict[str, UserContext] = {}
        self._permissions: dict[str, list[Permission]] = {}
        self._classifications: dict[str, list[Classification]] = {}
        self._masking_rules: dict[str, list[MaskingRule]] = {}
        self._row_filters: dict[str, list[RowFilter]] = {}
        self._audit_log: list[dict] = []
        self._connected = False

    @property
    def provider_name(self) -> str:
        return "local"

    async def connect(self, config: dict) -> None:
        """Initialize local governance provider."""
        self._connected = True
        logger.info("Connected to local governance provider")

    async def disconnect(self) -> None:
        """Disconnect from local provider."""
        self._connected = False

    async def health_check(self) -> bool:
        """Check if provider is ready."""
        return self._connected

    async def get_user_context(self, credentials: dict) -> UserContext:
        """
        Get or create user context from credentials.
        """
        user_id = credentials.get("user_id", credentials.get("email", "local-user"))

        if user_id in self._users:
            return self._users[user_id]

        user = UserContext(
            user_id=user_id,
            email=credentials.get("email", f"{user_id}@local"),
            display_name=credentials.get("name", user_id),
            groups=credentials.get("groups", ["developers"]),
            roles=credentials.get("roles", ["reader"]),
            attributes=credentials,
            provider="local",
        )

        self._users[user_id] = user
        return user

    async def get_accessible_resources(
        self,
        user: UserContext,
        resource_type: Optional[str] = None,
        catalog: Optional[str] = None,
        database: Optional[str] = None,
    ) -> list[Resource]:
        """
        Get accessible resources for the user.
        """
        resources = []

        # Return all resources that have permissions for this user
        user_key = user.user_id

        for resource_key, permissions in self._permissions.items():
            for perm in permissions:
                if perm.decision == AccessDecision.ALLOW:
                    resource = perm.resource

                    if resource_type and resource.resource_type != resource_type:
                        continue
                    if catalog and resource.catalog != catalog:
                        continue
                    if database and resource.database != database:
                        continue

                    resources.append(resource)

        return resources

    async def check_permission(
        self,
        user: UserContext,
        resource: Resource,
        action: str,
    ) -> Permission:
        """
        Check permission for the user on a resource.
        """
        resource_key = resource.fully_qualified_name

        # Check explicit permissions
        if resource_key in self._permissions:
            for perm in self._permissions[resource_key]:
                if perm.action == action:
                    return perm

        # Default: allow for local testing
        return Permission(
            resource=resource,
            action=action,
            decision=AccessDecision.ALLOW,
            reason="Local provider default allow",
            granted_by="local_default",
        )

    async def get_classifications(
        self,
        resource: Resource,
        user: Optional[UserContext] = None,
    ) -> list[Classification]:
        """
        Get classifications for a resource.
        """
        resource_key = resource.fully_qualified_name
        return self._classifications.get(resource_key, [])

    async def get_masking_rules(
        self,
        resource: Resource,
        user: UserContext,
    ) -> list[MaskingRule]:
        """
        Get masking rules for a resource.
        """
        resource_key = resource.fully_qualified_name
        return self._masking_rules.get(resource_key, [])

    async def get_row_filters(
        self,
        resource: Resource,
        user: UserContext,
    ) -> list[RowFilter]:
        """
        Get row filters for a resource.
        """
        resource_key = resource.fully_qualified_name
        return self._row_filters.get(resource_key, [])

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
        Log audit entry to in-memory store.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user.user_id,
            "resource": resource.fully_qualified_name,
            "action": action,
            "decision": decision.value,
            "trace_id": trace_id,
            "details": details or {},
        }
        self._audit_log.append(entry)
        logger.debug(f"Audit: {entry}")

    # Helper methods for setting up test data

    def add_permission(
        self,
        resource: Resource,
        action: str,
        decision: AccessDecision = AccessDecision.ALLOW,
        reason: Optional[str] = None,
    ) -> None:
        """Add a permission rule."""
        resource_key = resource.fully_qualified_name

        if resource_key not in self._permissions:
            self._permissions[resource_key] = []

        self._permissions[resource_key].append(Permission(
            resource=resource,
            action=action,
            decision=decision,
            reason=reason,
            granted_by="local_admin",
        ))

    def add_classification(
        self,
        resource: Resource,
        classification: str,
        level: ClassificationLevel,
    ) -> None:
        """Add a classification."""
        resource_key = resource.fully_qualified_name

        if resource_key not in self._classifications:
            self._classifications[resource_key] = []

        self._classifications[resource_key].append(Classification(
            resource=resource,
            classification=classification,
            level=level,
            source="local_admin",
        ))

    def add_masking_rule(
        self,
        resource: Resource,
        column: str,
        mask_type: str,
    ) -> None:
        """Add a masking rule."""
        resource_key = resource.fully_qualified_name

        if resource_key not in self._masking_rules:
            self._masking_rules[resource_key] = []

        self._masking_rules[resource_key].append(MaskingRule(
            resource=resource,
            column=column,
            mask_type=mask_type,
        ))

    def add_row_filter(
        self,
        resource: Resource,
        filter_expression: str,
    ) -> None:
        """Add a row filter."""
        resource_key = resource.fully_qualified_name

        if resource_key not in self._row_filters:
            self._row_filters[resource_key] = []

        self._row_filters[resource_key].append(RowFilter(
            resource=resource,
            filter_expression=filter_expression,
            filter_type="sql",
        ))

    def get_audit_log(self) -> list[dict]:
        """Get the audit log."""
        return self._audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()

    def reset(self) -> None:
        """Reset all governance data."""
        self._users.clear()
        self._permissions.clear()
        self._classifications.clear()
        self._masking_rules.clear()
        self._row_filters.clear()
        self._audit_log.clear()
