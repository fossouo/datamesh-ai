"""
Authorizer - Authorization decision engine.

Evaluates authorization requests against policies and user context
to make access control decisions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuthorizationResult:
    """Result of an authorization check."""
    authorized: bool
    reason: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    policy_refs: list[str] = field(default_factory=list)


class Authorizer:
    """
    Authorization decision engine.

    Evaluates authorization requests based on:
    - User roles and attributes
    - Resource classifications
    - Applied policies
    - Contextual constraints
    """

    def __init__(self):
        # In production, these would be loaded from policy store
        self._role_permissions = self._load_role_permissions()
        self._resource_policies = self._load_resource_policies()

    def _load_role_permissions(self) -> dict[str, dict]:
        """Load role-based permissions."""
        return {
            "admin": {
                "actions": ["read", "write", "delete", "admin"],
                "resources": ["*"],
            },
            "analyst": {
                "actions": ["read"],
                "resources": ["catalog://*", "finance.revenue", "finance.customer_transactions"],
                "constraints": {
                    "maxRows": 100000,
                    "excludeClassifications": ["SECRET"],
                },
            },
            "engineer": {
                "actions": ["read", "write"],
                "resources": ["catalog://*"],
                "constraints": {
                    "excludeClassifications": ["PII", "SECRET"],
                },
            },
            "viewer": {
                "actions": ["read"],
                "resources": ["catalog://*"],
                "constraints": {
                    "maxRows": 1000,
                    "excludeClassifications": ["PII", "SENSITIVE", "SECRET"],
                },
            },
        }

    def _load_resource_policies(self) -> dict[str, dict]:
        """Load resource-specific policies."""
        return {
            "catalog://finance.customer_transactions": {
                "classifications": ["PII", "SENSITIVE"],
                "requiredRoles": ["analyst", "admin"],
                "constraints": {
                    "rowFilters": ["country IN (${user.allowed_countries})"],
                    "columnMasking": [
                        {"column": "card_number", "method": "mask_last4"},
                        {"column": "ssn", "method": "full_mask"},
                    ],
                },
            },
            "catalog://finance.revenue": {
                "classifications": ["INTERNAL"],
                "requiredRoles": ["analyst", "engineer", "admin"],
            },
            "catalog://hr.salaries": {
                "classifications": ["PII", "SECRET"],
                "requiredRoles": ["admin"],
            },
        }

    async def authorize(
        self,
        user: dict,
        action: str,
        resources: list[str],
        policies: list[str],
        context: dict,
    ) -> AuthorizationResult:
        """
        Evaluate authorization request.

        Args:
            user: User information (id, roles)
            action: Requested action (read, write, etc.)
            resources: List of resource URIs
            policies: List of policy references
            context: Additional context

        Returns:
            AuthorizationResult with decision and constraints
        """
        user_id = user.get("id", "anonymous")
        user_roles = user.get("roles", [])

        logger.info(
            f"Authorizing user={user_id} action={action} "
            f"resources={resources} roles={user_roles}"
        )

        # Check each resource
        all_constraints = {}
        applied_policies = []

        for resource in resources:
            result = self._check_resource_access(
                user_id=user_id,
                user_roles=user_roles,
                action=action,
                resource=resource,
                policies=policies,
            )

            if not result.authorized:
                return result

            # Merge constraints
            all_constraints.update(result.constraints)
            applied_policies.extend(result.policy_refs)

        # Check policy-specific rules
        for policy in policies:
            policy_result = self._evaluate_policy(
                policy=policy,
                user=user,
                action=action,
                resources=resources,
            )
            if not policy_result.authorized:
                return policy_result
            applied_policies.extend(policy_result.policy_refs)

        return AuthorizationResult(
            authorized=True,
            constraints=all_constraints,
            policy_refs=list(set(applied_policies)),
        )

    def _check_resource_access(
        self,
        user_id: str,
        user_roles: list[str],
        action: str,
        resource: str,
        policies: list[str],
    ) -> AuthorizationResult:
        """Check access to a single resource."""

        # Get resource policy
        resource_policy = self._get_resource_policy(resource)

        # Check required roles
        if resource_policy:
            required_roles = resource_policy.get("requiredRoles", [])
            if required_roles and not any(r in user_roles for r in required_roles):
                return AuthorizationResult(
                    authorized=False,
                    reason=f"User lacks required role for {resource}",
                    policy_refs=[resource],
                )

        # Check role permissions
        for role in user_roles:
            role_perms = self._role_permissions.get(role, {})

            # Check action allowed
            allowed_actions = role_perms.get("actions", [])
            if action not in allowed_actions and "*" not in allowed_actions:
                continue

            # Check resource allowed
            allowed_resources = role_perms.get("resources", [])
            if self._matches_resource(resource, allowed_resources):
                constraints = role_perms.get("constraints", {}).copy()

                # Add resource-specific constraints
                if resource_policy:
                    resource_constraints = resource_policy.get("constraints", {})
                    constraints.update(resource_constraints)

                return AuthorizationResult(
                    authorized=True,
                    constraints=constraints,
                    policy_refs=[role],
                )

        return AuthorizationResult(
            authorized=False,
            reason=f"No role grants {action} access to {resource}",
        )

    def _get_resource_policy(self, resource: str) -> Optional[dict]:
        """Get policy for a specific resource."""
        # Direct match
        if resource in self._resource_policies:
            return self._resource_policies[resource]

        # Check for pattern matches
        for policy_resource, policy in self._resource_policies.items():
            if self._matches_resource(resource, [policy_resource]):
                return policy

        return None

    def _matches_resource(self, resource: str, patterns: list[str]) -> bool:
        """Check if resource matches any pattern."""
        for pattern in patterns:
            if pattern == "*":
                return True
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if resource.startswith(prefix):
                    return True
            if resource == pattern:
                return True
            # Handle catalog:// URIs
            if resource.replace("catalog://", "") == pattern:
                return True

        return False

    def _evaluate_policy(
        self,
        policy: str,
        user: dict,
        action: str,
        resources: list[str],
    ) -> AuthorizationResult:
        """Evaluate a specific policy."""
        # RGPD policy
        if "rgpd" in policy.lower():
            return self._evaluate_rgpd_policy(user, action, resources)

        # Data retention policy
        if "retention" in policy.lower():
            return self._evaluate_retention_policy(user, action, resources)

        # Default: allow
        return AuthorizationResult(authorized=True, policy_refs=[policy])

    def _evaluate_rgpd_policy(
        self,
        user: dict,
        action: str,
        resources: list[str],
    ) -> AuthorizationResult:
        """Evaluate RGPD compliance policy."""
        # RGPD requires purpose limitation and data minimization
        constraints = {
            "dataPurpose": "analytics",
            "dataMinimization": True,
            "auditRequired": True,
        }

        return AuthorizationResult(
            authorized=True,
            constraints=constraints,
            policy_refs=["policies/rgpd.yaml"],
        )

    def _evaluate_retention_policy(
        self,
        user: dict,
        action: str,
        resources: list[str],
    ) -> AuthorizationResult:
        """Evaluate data retention policy."""
        constraints = {
            "maxRetentionDays": 365,
            "deleteAfterUse": False,
        }

        return AuthorizationResult(
            authorized=True,
            constraints=constraints,
            policy_refs=["policies/data-retention.yaml"],
        )
