"""
Governance Interfaces - Abstract base classes for governance providers.

These interfaces define the contract that all cloud-specific governance
adapters must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class AccessDecision(Enum):
    """Result of an access permission check."""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"
    MASKED = "masked"


class ClassificationLevel(Enum):
    """Data classification sensitivity levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    PHI = "phi"
    PCI = "pci"


@dataclass
class UserContext:
    """
    User context representing the authenticated user and their attributes.

    This is populated from the cloud provider's identity system.
    """
    user_id: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    groups: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    raw_token: Optional[dict] = None

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def in_group(self, group: str) -> bool:
        """Check if user is in a specific group."""
        return group in self.groups


@dataclass
class Resource:
    """
    Represents a data resource (table, column, database, etc.).
    """
    resource_type: str  # database, schema, table, column, file
    catalog: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = None
    table: Optional[str] = None
    column: Optional[str] = None
    path: Optional[str] = None
    uri: Optional[str] = None
    provider: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fully_qualified_name(self) -> str:
        """Get the fully qualified name of the resource."""
        parts = []
        if self.catalog:
            parts.append(self.catalog)
        if self.database:
            parts.append(self.database)
        if self.schema:
            parts.append(self.schema)
        if self.table:
            parts.append(self.table)
        if self.column:
            parts.append(self.column)
        return ".".join(parts) if parts else self.path or self.uri or ""


@dataclass
class Permission:
    """
    Permission details for a resource access check.
    """
    resource: Resource
    action: str  # SELECT, INSERT, UPDATE, DELETE, DESCRIBE, etc.
    decision: AccessDecision
    reason: Optional[str] = None
    conditions: list[str] = field(default_factory=list)
    granted_by: Optional[str] = None  # policy, role, or direct grant
    expires_at: Optional[datetime] = None


@dataclass
class Classification:
    """
    Data classification for a resource.
    """
    resource: Resource
    classification: str
    level: ClassificationLevel
    source: str  # auto, manual, policy
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskingRule:
    """
    Data masking rule for sensitive data.
    """
    resource: Resource
    column: str
    mask_type: str  # hash, redact, partial, null, custom
    mask_function: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)
    applies_to_roles: list[str] = field(default_factory=list)
    excludes_roles: list[str] = field(default_factory=list)


@dataclass
class RowFilter:
    """
    Row-level security filter.
    """
    resource: Resource
    filter_expression: str
    filter_type: str  # sql, predicate, function
    applies_to_roles: list[str] = field(default_factory=list)
    excludes_roles: list[str] = field(default_factory=list)


@dataclass
class AuditEntry:
    """
    Audit log entry for governance actions.
    """
    timestamp: datetime
    user: UserContext
    resource: Resource
    action: str
    decision: AccessDecision
    details: dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class GovernanceProvider(ABC):
    """
    Abstract base class for cloud-specific governance providers.

    Each cloud provider (AWS, Azure, GCP, Databricks) must implement
    this interface to provide governance capabilities.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (aws, azure, gcp, databricks)."""
        pass

    @abstractmethod
    async def connect(self, config: dict) -> None:
        """
        Establish connection to the governance service.

        Args:
            config: Provider-specific configuration
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the governance service."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the governance service is accessible."""
        pass

    @abstractmethod
    async def get_user_context(self, credentials: dict) -> UserContext:
        """
        Get user context from credentials.

        Args:
            credentials: Authentication credentials (token, keys, etc.)

        Returns:
            UserContext with user identity and attributes
        """
        pass

    @abstractmethod
    async def get_accessible_resources(
        self,
        user: UserContext,
        resource_type: Optional[str] = None,
        catalog: Optional[str] = None,
        database: Optional[str] = None,
    ) -> list[Resource]:
        """
        Get list of resources accessible to the user.

        Args:
            user: The user context
            resource_type: Filter by resource type
            catalog: Filter by catalog
            database: Filter by database

        Returns:
            List of accessible resources
        """
        pass

    @abstractmethod
    async def check_permission(
        self,
        user: UserContext,
        resource: Resource,
        action: str,
    ) -> Permission:
        """
        Check if user has permission for an action on a resource.

        Args:
            user: The user context
            resource: The target resource
            action: The action to check (SELECT, INSERT, etc.)

        Returns:
            Permission object with decision and details
        """
        pass

    @abstractmethod
    async def get_classifications(
        self,
        resource: Resource,
        user: Optional[UserContext] = None,
    ) -> list[Classification]:
        """
        Get data classifications for a resource.

        Args:
            resource: The resource to get classifications for
            user: Optional user context to filter visible classifications

        Returns:
            List of classifications
        """
        pass

    @abstractmethod
    async def get_masking_rules(
        self,
        resource: Resource,
        user: UserContext,
    ) -> list[MaskingRule]:
        """
        Get applicable masking rules for a resource and user.

        Args:
            resource: The resource
            user: The user context

        Returns:
            List of masking rules to apply
        """
        pass

    @abstractmethod
    async def get_row_filters(
        self,
        resource: Resource,
        user: UserContext,
    ) -> list[RowFilter]:
        """
        Get row-level security filters for a resource and user.

        Args:
            resource: The resource
            user: The user context

        Returns:
            List of row filters to apply
        """
        pass

    @abstractmethod
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
        Log an access audit entry.

        Args:
            user: The user context
            resource: The accessed resource
            action: The action performed
            decision: The access decision
            details: Additional details
            trace_id: Optional trace ID for correlation
        """
        pass

    async def apply_governance_to_query(
        self,
        user: UserContext,
        query: str,
        resources: list[Resource],
    ) -> tuple[str, list[str]]:
        """
        Apply governance rules to a SQL query.

        This is a convenience method that combines permission checks,
        masking rules, and row filters into query modifications.

        Args:
            user: The user context
            query: The original SQL query
            resources: Resources referenced in the query

        Returns:
            Tuple of (modified_query, list_of_warnings)
        """
        warnings = []
        modified_query = query

        for resource in resources:
            # Check permissions
            permission = await self.check_permission(user, resource, "SELECT")
            if permission.decision == AccessDecision.DENY:
                raise PermissionError(
                    f"Access denied to {resource.fully_qualified_name}: {permission.reason}"
                )

            # Get and apply masking rules
            masking_rules = await self.get_masking_rules(resource, user)
            for rule in masking_rules:
                warnings.append(
                    f"Column {rule.column} will be masked using {rule.mask_type}"
                )

            # Get and apply row filters
            row_filters = await self.get_row_filters(resource, user)
            for rfilter in row_filters:
                warnings.append(
                    f"Row filter applied: {rfilter.filter_expression}"
                )

        return modified_query, warnings


class PermissionError(Exception):
    """Raised when a permission check fails."""
    pass
