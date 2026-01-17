"""
DATAMESH.AI Orchestrator - Contract and Policy Validator
=========================================================

This module provides validation for A2A requests against agent contracts
and governance policies. It ensures that:
- Callers are authorized to call callees
- Requested capabilities exist and are allowed
- Policies are properly applied
- Trace context requirements are met
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ValidationErrorCode(str, Enum):
    """Error codes for validation failures."""
    FORBIDDEN_CAPABILITY = "FORBIDDEN_CAPABILITY"
    UNKNOWN_CAPABILITY = "UNKNOWN_CAPABILITY"
    UNKNOWN_AGENT = "UNKNOWN_AGENT"
    MAX_DEPTH_EXCEEDED = "MAX_DEPTH_EXCEEDED"
    MISSING_TRACE_PARENT = "MISSING_TRACE_PARENT"
    INVALID_DELEGATION = "INVALID_DELEGATION"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    MISSING_REQUIRED_POLICY = "MISSING_REQUIRED_POLICY"
    INVALID_REQUEST_FORMAT = "INVALID_REQUEST_FORMAT"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    DEADLINE_REJECTED = "DEADLINE_REJECTED"
    AGENT_DISABLED = "AGENT_DISABLED"
    CLASSIFICATION_BLOCKED = "CLASSIFICATION_BLOCKED"


@dataclass
class ValidationError:
    """Represents a validation error with context."""
    code: ValidationErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    policy_refs: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    applied_policies: list[str] = field(default_factory=list)

    def add_error(
        self,
        code: ValidationErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
        policy_refs: list[str] | None = None
    ) -> None:
        """Add a validation error."""
        self.valid = False
        self.errors.append(ValidationError(
            code=code,
            message=message,
            details=details or {},
            policy_refs=policy_refs or []
        ))

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.applied_policies.extend(other.applied_policies)


@dataclass
class AgentInfo:
    """Information about a registered agent."""
    name: str
    display_name: str
    endpoint: str
    health_endpoint: str
    capabilities: list[str]
    can_call: list[str]
    contract_ref: str
    enabled: bool


@dataclass
class A2ARequest:
    """Parsed A2A request for validation."""
    protocol_version: str
    request_id: str
    deadline_ms: int
    trace_id: Optional[str]
    parent_span_id: Optional[str]
    span_id: Optional[str]
    caller_agent: str
    caller_capability: str
    callee_agent: str
    callee_capability: str
    on_behalf_of: Optional[dict[str, Any]]
    policies_applied: list[str]
    current_depth: int
    max_depth: int
    auth_context_ref: Optional[str]
    payload_data: dict[str, Any]


class ContractValidator:
    """
    Validates A2A requests against agent contracts.

    Ensures that:
    - Both caller and callee agents exist and are enabled
    - Caller is authorized to call callee
    - Requested capability is exposed by callee
    - Depth limits are respected
    - Trace context is provided when required
    """

    def __init__(
        self,
        agents: dict[str, AgentInfo],
        require_trace_parent: bool = True,
        max_depth: int = 5,
        allow_delegation: bool = True
    ):
        """
        Initialize the contract validator.

        Args:
            agents: Dictionary mapping agent names to AgentInfo
            require_trace_parent: Whether trace context is required
            max_depth: Maximum allowed call depth
            allow_delegation: Whether delegation (onBehalfOf) is allowed
        """
        self.agents = agents
        self.require_trace_parent = require_trace_parent
        self.max_depth = max_depth
        self.allow_delegation = allow_delegation

    def validate(self, request: A2ARequest) -> ValidationResult:
        """
        Validate an A2A request against contracts.

        Args:
            request: The A2A request to validate

        Returns:
            ValidationResult with any errors or warnings
        """
        result = ValidationResult(valid=True)

        # Validate protocol version
        self._validate_protocol_version(request, result)

        # Validate caller agent
        self._validate_caller(request, result)

        # Validate callee agent
        self._validate_callee(request, result)

        # Validate caller can call callee
        if result.valid:
            self._validate_call_authorization(request, result)

        # Validate capability exists
        if result.valid:
            self._validate_capability(request, result)

        # Validate depth
        self._validate_depth(request, result)

        # Validate trace context
        self._validate_trace_context(request, result)

        # Validate delegation
        self._validate_delegation(request, result)

        # Validate deadline
        self._validate_deadline(request, result)

        return result

    def _validate_protocol_version(
        self,
        request: A2ARequest,
        result: ValidationResult
    ) -> None:
        """Validate the protocol version."""
        supported_versions = ["datamesh.ai/a2a/v1"]
        if request.protocol_version not in supported_versions:
            result.add_error(
                ValidationErrorCode.INVALID_REQUEST_FORMAT,
                f"Unsupported protocol version: {request.protocol_version}",
                {"supported_versions": supported_versions}
            )

    def _validate_caller(
        self,
        request: A2ARequest,
        result: ValidationResult
    ) -> None:
        """Validate the caller agent exists and is enabled."""
        if request.caller_agent not in self.agents:
            result.add_error(
                ValidationErrorCode.UNKNOWN_AGENT,
                f"Unknown caller agent: {request.caller_agent}",
                {"agent": request.caller_agent, "role": "caller"}
            )
            return

        caller = self.agents[request.caller_agent]
        if not caller.enabled:
            result.add_error(
                ValidationErrorCode.AGENT_DISABLED,
                f"Caller agent is disabled: {request.caller_agent}",
                {"agent": request.caller_agent}
            )

    def _validate_callee(
        self,
        request: A2ARequest,
        result: ValidationResult
    ) -> None:
        """Validate the callee agent exists and is enabled."""
        if request.callee_agent not in self.agents:
            result.add_error(
                ValidationErrorCode.UNKNOWN_AGENT,
                f"Unknown callee agent: {request.callee_agent}",
                {"agent": request.callee_agent, "role": "callee"}
            )
            return

        callee = self.agents[request.callee_agent]
        if not callee.enabled:
            result.add_error(
                ValidationErrorCode.AGENT_DISABLED,
                f"Callee agent is disabled: {request.callee_agent}",
                {"agent": request.callee_agent}
            )

    def _validate_call_authorization(
        self,
        request: A2ARequest,
        result: ValidationResult
    ) -> None:
        """Validate that caller is authorized to call callee."""
        caller = self.agents[request.caller_agent]

        if request.callee_agent not in caller.can_call:
            result.add_error(
                ValidationErrorCode.FORBIDDEN_CAPABILITY,
                f"Agent '{request.caller_agent}' is not authorized to call "
                f"agent '{request.callee_agent}'",
                {
                    "caller": request.caller_agent,
                    "callee": request.callee_agent,
                    "allowed_callees": caller.can_call
                }
            )

    def _validate_capability(
        self,
        request: A2ARequest,
        result: ValidationResult
    ) -> None:
        """Validate that the requested capability exists on callee."""
        callee = self.agents[request.callee_agent]

        if request.callee_capability not in callee.capabilities:
            result.add_error(
                ValidationErrorCode.UNKNOWN_CAPABILITY,
                f"Agent '{request.callee_agent}' does not expose capability "
                f"'{request.callee_capability}'",
                {
                    "agent": request.callee_agent,
                    "requested_capability": request.callee_capability,
                    "available_capabilities": callee.capabilities
                }
            )

    def _validate_depth(
        self,
        request: A2ARequest,
        result: ValidationResult
    ) -> None:
        """Validate call depth limits."""
        effective_max_depth = min(request.max_depth, self.max_depth)

        if request.current_depth >= effective_max_depth:
            result.add_error(
                ValidationErrorCode.MAX_DEPTH_EXCEEDED,
                f"Maximum call depth exceeded: {request.current_depth} >= "
                f"{effective_max_depth}",
                {
                    "current_depth": request.current_depth,
                    "max_depth": effective_max_depth,
                    "request_max_depth": request.max_depth,
                    "global_max_depth": self.max_depth
                }
            )

    def _validate_trace_context(
        self,
        request: A2ARequest,
        result: ValidationResult
    ) -> None:
        """Validate trace context is provided when required."""
        if self.require_trace_parent:
            if not request.trace_id:
                result.add_error(
                    ValidationErrorCode.MISSING_TRACE_PARENT,
                    "Trace context is required but traceId is missing",
                    {"require_trace_parent": True}
                )
            elif not request.span_id:
                result.add_warning(
                    "spanId is missing from trace context; a new span will be created"
                )

    def _validate_delegation(
        self,
        request: A2ARequest,
        result: ValidationResult
    ) -> None:
        """Validate delegation (onBehalfOf) is allowed."""
        if request.on_behalf_of:
            if not self.allow_delegation:
                result.add_error(
                    ValidationErrorCode.INVALID_DELEGATION,
                    "Delegation (onBehalfOf) is not allowed by orchestrator policy",
                    {"on_behalf_of": request.on_behalf_of}
                )
            elif "userId" not in request.on_behalf_of:
                result.add_error(
                    ValidationErrorCode.INVALID_DELEGATION,
                    "onBehalfOf must include userId",
                    {"on_behalf_of": request.on_behalf_of}
                )

    def _validate_deadline(
        self,
        request: A2ARequest,
        result: ValidationResult
    ) -> None:
        """Validate deadline is within acceptable bounds."""
        # Note: Actual timeout config should be injected
        min_deadline = 1000  # 1 second
        max_deadline = 300000  # 5 minutes

        if request.deadline_ms < min_deadline:
            result.add_error(
                ValidationErrorCode.DEADLINE_REJECTED,
                f"Deadline too short: {request.deadline_ms}ms < {min_deadline}ms",
                {
                    "deadline_ms": request.deadline_ms,
                    "min_deadline_ms": min_deadline
                }
            )
        elif request.deadline_ms > max_deadline:
            result.add_warning(
                f"Deadline {request.deadline_ms}ms exceeds recommended maximum "
                f"{max_deadline}ms; may be truncated"
            )


class PolicyValidator:
    """
    Validates A2A requests against governance policies.

    Ensures that:
    - Required global policies are applied
    - Classification-specific policies are enforced
    - Policy violations are detected and reported
    """

    def __init__(
        self,
        global_policies: list[dict[str, Any]],
        classification_policies: dict[str, list[dict[str, Any]]],
        blocked_classifications: list[str] | None = None
    ):
        """
        Initialize the policy validator.

        Args:
            global_policies: List of global policies with ref and required flag
            classification_policies: Policies by classification level
            blocked_classifications: Classifications that should be blocked
        """
        self.global_policies = global_policies
        self.classification_policies = classification_policies
        self.blocked_classifications = blocked_classifications or []

    def validate(
        self,
        request: A2ARequest,
        data_classifications: list[str] | None = None
    ) -> ValidationResult:
        """
        Validate an A2A request against policies.

        Args:
            request: The A2A request to validate
            data_classifications: Classifications of data being accessed

        Returns:
            ValidationResult with any policy violations
        """
        result = ValidationResult(valid=True)

        # Validate global policies
        self._validate_global_policies(request, result)

        # Validate classification-specific policies
        if data_classifications:
            self._validate_classification_policies(
                request, data_classifications, result
            )

            # Check blocked classifications
            self._check_blocked_classifications(
                data_classifications, result
            )

        return result

    def _validate_global_policies(
        self,
        request: A2ARequest,
        result: ValidationResult
    ) -> None:
        """Validate that required global policies are applied."""
        applied_set = set(request.policies_applied)

        for policy in self.global_policies:
            policy_ref = policy["ref"]
            required = policy.get("required", False)

            if required and policy_ref not in applied_set:
                result.add_error(
                    ValidationErrorCode.MISSING_REQUIRED_POLICY,
                    f"Required policy not applied: {policy_ref}",
                    {"policy_ref": policy_ref},
                    policy_refs=[policy_ref]
                )
            elif policy_ref in applied_set:
                result.applied_policies.append(policy_ref)

    def _validate_classification_policies(
        self,
        request: A2ARequest,
        classifications: list[str],
        result: ValidationResult
    ) -> None:
        """Validate classification-specific policies are applied."""
        applied_set = set(request.policies_applied)

        for classification in classifications:
            if classification in self.classification_policies:
                for policy in self.classification_policies[classification]:
                    policy_ref = policy["ref"]
                    required = policy.get("required", False)

                    if required and policy_ref not in applied_set:
                        result.add_error(
                            ValidationErrorCode.MISSING_REQUIRED_POLICY,
                            f"Required policy for classification '{classification}' "
                            f"not applied: {policy_ref}",
                            {
                                "policy_ref": policy_ref,
                                "classification": classification
                            },
                            policy_refs=[policy_ref]
                        )
                    elif policy_ref in applied_set:
                        result.applied_policies.append(policy_ref)

    def _check_blocked_classifications(
        self,
        classifications: list[str],
        result: ValidationResult
    ) -> None:
        """Check if any blocked classifications are present."""
        blocked = [c for c in classifications if c in self.blocked_classifications]
        if blocked:
            result.add_error(
                ValidationErrorCode.CLASSIFICATION_BLOCKED,
                f"Access to data with blocked classifications: {blocked}",
                {"blocked_classifications": blocked}
            )


class RequestParser:
    """Parses raw A2A request dictionaries into A2ARequest objects."""

    @staticmethod
    def parse(raw_request: dict[str, Any]) -> A2ARequest:
        """
        Parse a raw A2A request dictionary.

        Args:
            raw_request: Raw request dictionary

        Returns:
            Parsed A2ARequest object

        Raises:
            ValueError: If required fields are missing
        """
        # Extract trace context
        trace = raw_request.get("trace", {})

        # Extract context
        context = raw_request.get("context", {})
        constraints = context.get("constraints", {})
        on_behalf_of = context.get("onBehalfOf")

        # Extract caller/callee
        caller = raw_request.get("caller", {})
        callee = raw_request.get("callee", {})

        # Extract payload
        payload_ref = raw_request.get("payloadRef", {})

        return A2ARequest(
            protocol_version=raw_request.get("protocolVersion", ""),
            request_id=raw_request.get("requestId", ""),
            deadline_ms=raw_request.get("deadlineMs", 30000),
            trace_id=trace.get("traceId"),
            parent_span_id=trace.get("parentSpanId"),
            span_id=trace.get("spanId"),
            caller_agent=caller.get("agent", ""),
            caller_capability=caller.get("capability", ""),
            callee_agent=callee.get("agent", ""),
            callee_capability=callee.get("capability", ""),
            on_behalf_of=on_behalf_of,
            policies_applied=context.get("policiesApplied", []),
            current_depth=constraints.get("currentDepth", 0),
            max_depth=constraints.get("maxDepth", 5),
            auth_context_ref=context.get("authContextRef"),
            payload_data=payload_ref.get("data", {})
        )

    @staticmethod
    def validate_structure(raw_request: dict[str, Any]) -> ValidationResult:
        """
        Validate the basic structure of a raw A2A request.

        Args:
            raw_request: Raw request dictionary

        Returns:
            ValidationResult with any structural errors
        """
        result = ValidationResult(valid=True)

        # Required top-level fields
        required_fields = ["protocolVersion", "requestId", "caller", "callee"]
        for field in required_fields:
            if field not in raw_request:
                result.add_error(
                    ValidationErrorCode.INVALID_REQUEST_FORMAT,
                    f"Missing required field: {field}",
                    {"field": field}
                )

        # Validate caller structure
        caller = raw_request.get("caller", {})
        if not isinstance(caller, dict):
            result.add_error(
                ValidationErrorCode.INVALID_REQUEST_FORMAT,
                "caller must be an object",
                {"field": "caller"}
            )
        elif "agent" not in caller:
            result.add_error(
                ValidationErrorCode.INVALID_REQUEST_FORMAT,
                "caller.agent is required",
                {"field": "caller.agent"}
            )

        # Validate callee structure
        callee = raw_request.get("callee", {})
        if not isinstance(callee, dict):
            result.add_error(
                ValidationErrorCode.INVALID_REQUEST_FORMAT,
                "callee must be an object",
                {"field": "callee"}
            )
        elif "agent" not in callee or "capability" not in callee:
            result.add_error(
                ValidationErrorCode.INVALID_REQUEST_FORMAT,
                "callee.agent and callee.capability are required",
                {"field": "callee"}
            )

        # Validate requestId format (should be non-empty string)
        request_id = raw_request.get("requestId", "")
        if not isinstance(request_id, str) or not request_id.strip():
            result.add_error(
                ValidationErrorCode.INVALID_REQUEST_FORMAT,
                "requestId must be a non-empty string",
                {"field": "requestId"}
            )

        return result


def redact_fields(
    data: dict[str, Any],
    patterns: list[str]
) -> dict[str, Any]:
    """
    Redact sensitive fields from a dictionary based on patterns.

    Args:
        data: Dictionary to redact
        patterns: List of field patterns to redact (supports wildcards)

    Returns:
        Dictionary with sensitive fields redacted
    """
    def should_redact(path: str) -> bool:
        for pattern in patterns:
            # Convert pattern to regex
            regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
            if re.match(f"^{regex_pattern}$", path):
                return True
        return False

    def redact_recursive(
        obj: Any,
        current_path: str = ""
    ) -> Any:
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                new_path = f"{current_path}.{key}" if current_path else key
                if should_redact(new_path):
                    result[key] = "[REDACTED]"
                else:
                    result[key] = redact_recursive(value, new_path)
            return result
        elif isinstance(obj, list):
            return [
                redact_recursive(item, f"{current_path}[]")
                for item in obj
            ]
        else:
            return obj

    return redact_recursive(data)
