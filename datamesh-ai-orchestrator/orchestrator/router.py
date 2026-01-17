"""
DATAMESH.AI Orchestrator - A2A Request Router
==============================================

This module handles routing of A2A requests between agents. It:
- Validates requests against contracts and policies
- Propagates trace context
- Enforces depth limits and timeouts
- Handles SUCCESS/ERROR/IN_PROGRESS responses
- Implements circuit breaker and retry logic
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import httpx

from orchestrator.validator import (
    AgentInfo,
    A2ARequest,
    ContractValidator,
    PolicyValidator,
    RequestParser,
    ValidationResult,
    redact_fields,
)

logger = logging.getLogger(__name__)


class ResponseStatus(str, Enum):
    """Status codes for A2A responses."""
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"


class ErrorCode(str, Enum):
    """Error codes for A2A errors."""
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
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    TIMEOUT = "TIMEOUT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class A2AResponse:
    """Represents an A2A response."""
    request_id: str
    status: ResponseStatus
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    output: Optional[dict[str, Any]] = None
    error: Optional[dict[str, Any]] = None
    audit_ref: Optional[str] = None
    next_poll_after_ms: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert response to dictionary."""
        result: dict[str, Any] = {
            "requestId": self.request_id,
            "status": self.status.value,
        }

        if self.trace_id:
            result["trace"] = {
                "traceId": self.trace_id,
                "spanId": self.span_id,
            }

        if self.status == ResponseStatus.SUCCESS and self.output:
            result["output"] = self.output

        if self.status == ResponseStatus.ERROR and self.error:
            result["error"] = self.error

        if self.status == ResponseStatus.IN_PROGRESS and self.next_poll_after_ms:
            result["nextPollAfterMs"] = self.next_poll_after_ms

        if self.audit_ref:
            result["auditRef"] = self.audit_ref

        return result


@dataclass
class CircuitBreakerState:
    """State for circuit breaker pattern."""
    failures: int = 0
    successes: int = 0
    state: str = "closed"  # closed, open, half-open
    last_failure_time: float = 0.0


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    enabled: bool = True
    max_attempts: int = 3
    backoff_ms: int = 500
    backoff_multiplier: float = 2.0
    max_backoff_ms: int = 5000
    retryable_errors: list[str] = field(default_factory=lambda: [
        "TIMEOUT", "SERVICE_UNAVAILABLE", "CONNECTION_ERROR"
    ])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    enabled: bool = True
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_ms: int = 60000


@dataclass
class RouterConfig:
    """Configuration for the A2A router."""
    default_deadline_ms: int = 30000
    max_deadline_ms: int = 300000
    min_deadline_ms: int = 1000
    connection_timeout_ms: int = 10000
    read_timeout_ms: int = 60000
    max_depth: int = 5
    require_trace_parent: bool = True
    allow_delegation: bool = True
    retry: RetryConfig = field(default_factory=RetryConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    redact_fields: list[str] = field(default_factory=lambda: [
        "user.token", "secrets.*", "credentials.*", "password", "apiKey"
    ])


class TraceContext:
    """Handles trace context propagation."""

    @staticmethod
    def generate_span_id() -> str:
        """Generate a new span ID."""
        return uuid.uuid4().hex[:16]

    @staticmethod
    def generate_trace_id() -> str:
        """Generate a new trace ID."""
        return uuid.uuid4().hex

    @staticmethod
    def propagate(
        parent_trace_id: Optional[str],
        parent_span_id: Optional[str]
    ) -> tuple[str, str, Optional[str]]:
        """
        Propagate trace context by creating a new span.

        Args:
            parent_trace_id: Parent trace ID (or None to create new trace)
            parent_span_id: Parent span ID

        Returns:
            Tuple of (trace_id, new_span_id, parent_span_id)
        """
        trace_id = parent_trace_id or TraceContext.generate_trace_id()
        new_span_id = TraceContext.generate_span_id()
        return trace_id, new_span_id, parent_span_id


class A2ARouter:
    """
    Routes A2A requests between agents with validation and observability.

    This router:
    - Validates requests against contracts and policies
    - Routes requests to appropriate agents
    - Propagates trace context
    - Implements circuit breaker pattern
    - Handles retries for transient failures
    - Logs audit information
    """

    def __init__(
        self,
        agents: dict[str, AgentInfo],
        config: RouterConfig,
        global_policies: list[dict[str, Any]] | None = None,
        classification_policies: dict[str, list[dict[str, Any]]] | None = None,
        audit_callback: Callable[[dict[str, Any]], None] | None = None
    ):
        """
        Initialize the A2A router.

        Args:
            agents: Dictionary mapping agent names to AgentInfo
            config: Router configuration
            global_policies: Global policies for validation
            classification_policies: Policies by classification
            audit_callback: Callback for audit logging
        """
        self.agents = agents
        self.config = config
        self.audit_callback = audit_callback

        # Initialize validators
        self.contract_validator = ContractValidator(
            agents=agents,
            require_trace_parent=config.require_trace_parent,
            max_depth=config.max_depth,
            allow_delegation=config.allow_delegation
        )

        self.policy_validator = PolicyValidator(
            global_policies=global_policies or [],
            classification_policies=classification_policies or {}
        )

        # Circuit breaker state per agent
        self.circuit_breakers: dict[str, CircuitBreakerState] = {}

        # HTTP client for forwarding requests
        self._client: Optional[httpx.AsyncClient] = None

        # In-flight requests for deduplication
        self._in_flight: dict[str, asyncio.Future] = {}

        # Trace store (in-memory, should be replaced with distributed store)
        self._trace_store: dict[str, list[dict[str, Any]]] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.connection_timeout_ms / 1000,
                    read=self.config.read_timeout_ms / 1000,
                    write=30.0,
                    pool=10.0
                )
            )
        return self._client

    async def close(self) -> None:
        """Close the router and release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def route(
        self,
        raw_request: dict[str, Any],
        data_classifications: list[str] | None = None
    ) -> A2AResponse:
        """
        Route an A2A request to the appropriate agent.

        Args:
            raw_request: Raw A2A request dictionary
            data_classifications: Classifications of data being accessed

        Returns:
            A2AResponse with result or error
        """
        start_time = time.time()

        # Validate request structure
        structure_result = RequestParser.validate_structure(raw_request)
        if not structure_result.valid:
            return self._error_response(
                request_id=raw_request.get("requestId", "unknown"),
                code=ErrorCode.INVALID_REQUEST_FORMAT,
                message="Invalid request structure",
                details={"errors": [e.__dict__ for e in structure_result.errors]}
            )

        # Parse request
        try:
            request = RequestParser.parse(raw_request)
        except Exception as e:
            logger.error(f"Failed to parse request: {e}")
            return self._error_response(
                request_id=raw_request.get("requestId", "unknown"),
                code=ErrorCode.INVALID_REQUEST_FORMAT,
                message=f"Failed to parse request: {str(e)}"
            )

        # Generate trace context
        trace_id, span_id, parent_span_id = TraceContext.propagate(
            request.trace_id,
            request.parent_span_id
        )

        # Store trace info
        self._store_trace(trace_id, {
            "requestId": request.request_id,
            "spanId": span_id,
            "parentSpanId": parent_span_id,
            "caller": request.caller_agent,
            "callee": request.callee_agent,
            "capability": request.callee_capability,
            "timestamp": time.time(),
            "status": "started"
        })

        try:
            # Validate against contracts
            contract_result = self.contract_validator.validate(request)
            if not contract_result.valid:
                return self._validation_error_response(
                    request, contract_result, trace_id, span_id
                )

            # Validate against policies
            policy_result = self.policy_validator.validate(
                request, data_classifications
            )
            if not policy_result.valid:
                return self._validation_error_response(
                    request, policy_result, trace_id, span_id
                )

            # Check circuit breaker
            if not self._check_circuit_breaker(request.callee_agent):
                return self._error_response(
                    request_id=request.request_id,
                    code=ErrorCode.CIRCUIT_OPEN,
                    message=f"Circuit breaker is open for agent: {request.callee_agent}",
                    trace_id=trace_id,
                    span_id=span_id
                )

            # Check deadline
            effective_deadline = self._get_effective_deadline(request.deadline_ms)
            elapsed = (time.time() - start_time) * 1000
            remaining_deadline = effective_deadline - elapsed

            if remaining_deadline <= 0:
                return self._error_response(
                    request_id=request.request_id,
                    code=ErrorCode.DEADLINE_EXCEEDED,
                    message="Request deadline exceeded before routing",
                    trace_id=trace_id,
                    span_id=span_id
                )

            # Forward request to agent
            response = await self._forward_request(
                request=request,
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                deadline_ms=int(remaining_deadline)
            )

            # Update circuit breaker
            self._record_success(request.callee_agent)

            # Update trace
            self._store_trace(trace_id, {
                "requestId": request.request_id,
                "spanId": span_id,
                "status": response.status.value,
                "timestamp": time.time(),
                "duration_ms": (time.time() - start_time) * 1000
            })

            # Audit log
            self._audit(request, response, trace_id, span_id)

            return response

        except Exception as e:
            logger.exception(f"Error routing request: {e}")
            self._record_failure(request.callee_agent)

            return self._error_response(
                request_id=request.request_id,
                code=ErrorCode.INTERNAL_ERROR,
                message=f"Internal error: {str(e)}",
                trace_id=trace_id,
                span_id=span_id
            )

    async def _forward_request(
        self,
        request: A2ARequest,
        trace_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        deadline_ms: int
    ) -> A2AResponse:
        """
        Forward a request to the target agent.

        Args:
            request: Parsed A2A request
            trace_id: Trace ID for this request
            span_id: Span ID for this hop
            parent_span_id: Parent span ID
            deadline_ms: Remaining deadline in milliseconds

        Returns:
            A2AResponse from the agent
        """
        agent = self.agents[request.callee_agent]
        client = await self._get_client()

        # Build forwarded request
        forward_request = {
            "protocolVersion": request.protocol_version,
            "requestId": request.request_id,
            "deadlineMs": deadline_ms,
            "trace": {
                "traceId": trace_id,
                "parentSpanId": span_id,  # Our span becomes their parent
                "spanId": TraceContext.generate_span_id()
            },
            "caller": {
                "agent": request.caller_agent,
                "capability": request.caller_capability
            },
            "callee": {
                "agent": request.callee_agent,
                "capability": request.callee_capability
            },
            "context": {
                "constraints": {
                    "maxDepth": request.max_depth,
                    "currentDepth": request.current_depth + 1
                },
                "policiesApplied": request.policies_applied
            },
            "payloadRef": {
                "data": request.payload_data
            }
        }

        if request.on_behalf_of:
            forward_request["context"]["onBehalfOf"] = request.on_behalf_of

        if request.auth_context_ref:
            forward_request["context"]["authContextRef"] = request.auth_context_ref

        # Retry logic
        last_error: Optional[Exception] = None
        attempt = 0
        max_attempts = self.config.retry.max_attempts if self.config.retry.enabled else 1

        while attempt < max_attempts:
            attempt += 1
            try:
                response = await client.post(
                    f"{agent.endpoint}/a2a/request",
                    json=forward_request,
                    timeout=deadline_ms / 1000
                )

                if response.status_code == 200:
                    data = response.json()
                    return A2AResponse(
                        request_id=data.get("requestId", request.request_id),
                        status=ResponseStatus(data.get("status", "ERROR")),
                        trace_id=trace_id,
                        span_id=span_id,
                        output=data.get("output"),
                        error=data.get("error"),
                        audit_ref=data.get("auditRef"),
                        next_poll_after_ms=data.get("nextPollAfterMs")
                    )
                elif response.status_code >= 500:
                    # Server error - may be retryable
                    last_error = Exception(f"Server error: {response.status_code}")
                    if attempt < max_attempts:
                        await self._backoff(attempt)
                        continue
                else:
                    # Client error - not retryable
                    return self._error_response(
                        request_id=request.request_id,
                        code=ErrorCode.AGENT_UNAVAILABLE,
                        message=f"Agent returned error: {response.status_code}",
                        trace_id=trace_id,
                        span_id=span_id
                    )

            except httpx.TimeoutException:
                last_error = Exception("Request timeout")
                if attempt < max_attempts and "TIMEOUT" in self.config.retry.retryable_errors:
                    await self._backoff(attempt)
                    continue
                return self._error_response(
                    request_id=request.request_id,
                    code=ErrorCode.TIMEOUT,
                    message="Request timed out",
                    trace_id=trace_id,
                    span_id=span_id
                )

            except httpx.ConnectError:
                last_error = Exception("Connection error")
                if attempt < max_attempts and "CONNECTION_ERROR" in self.config.retry.retryable_errors:
                    await self._backoff(attempt)
                    continue
                return self._error_response(
                    request_id=request.request_id,
                    code=ErrorCode.CONNECTION_ERROR,
                    message=f"Failed to connect to agent: {agent.endpoint}",
                    trace_id=trace_id,
                    span_id=span_id
                )

        # All retries exhausted
        return self._error_response(
            request_id=request.request_id,
            code=ErrorCode.AGENT_UNAVAILABLE,
            message=f"Agent unavailable after {max_attempts} attempts: {str(last_error)}",
            trace_id=trace_id,
            span_id=span_id
        )

    async def _backoff(self, attempt: int) -> None:
        """Calculate and wait for backoff period."""
        backoff = min(
            self.config.retry.backoff_ms * (self.config.retry.backoff_multiplier ** (attempt - 1)),
            self.config.retry.max_backoff_ms
        )
        await asyncio.sleep(backoff / 1000)

    def _get_effective_deadline(self, requested_deadline: int) -> int:
        """Get effective deadline within configured bounds."""
        if requested_deadline <= 0:
            return self.config.default_deadline_ms

        return max(
            self.config.min_deadline_ms,
            min(requested_deadline, self.config.max_deadline_ms)
        )

    def _check_circuit_breaker(self, agent_name: str) -> bool:
        """Check if circuit breaker allows request."""
        if not self.config.circuit_breaker.enabled:
            return True

        if agent_name not in self.circuit_breakers:
            self.circuit_breakers[agent_name] = CircuitBreakerState()

        state = self.circuit_breakers[agent_name]

        if state.state == "open":
            # Check if timeout has passed
            if time.time() - state.last_failure_time > self.config.circuit_breaker.timeout_ms / 1000:
                state.state = "half-open"
                return True
            return False

        return True

    def _record_success(self, agent_name: str) -> None:
        """Record successful request for circuit breaker."""
        if not self.config.circuit_breaker.enabled:
            return

        if agent_name not in self.circuit_breakers:
            return

        state = self.circuit_breakers[agent_name]
        state.successes += 1
        state.failures = 0

        if state.state == "half-open":
            if state.successes >= self.config.circuit_breaker.success_threshold:
                state.state = "closed"
                logger.info(f"Circuit breaker closed for agent: {agent_name}")

    def _record_failure(self, agent_name: str) -> None:
        """Record failed request for circuit breaker."""
        if not self.config.circuit_breaker.enabled:
            return

        if agent_name not in self.circuit_breakers:
            self.circuit_breakers[agent_name] = CircuitBreakerState()

        state = self.circuit_breakers[agent_name]
        state.failures += 1
        state.successes = 0
        state.last_failure_time = time.time()

        if state.failures >= self.config.circuit_breaker.failure_threshold:
            state.state = "open"
            logger.warning(f"Circuit breaker opened for agent: {agent_name}")

    def _store_trace(self, trace_id: str, span_data: dict[str, Any]) -> None:
        """Store trace span data."""
        if trace_id not in self._trace_store:
            self._trace_store[trace_id] = []
        self._trace_store[trace_id].append(span_data)

    def get_trace(self, trace_id: str) -> list[dict[str, Any]] | None:
        """Get trace data by trace ID."""
        return self._trace_store.get(trace_id)

    def _audit(
        self,
        request: A2ARequest,
        response: A2AResponse,
        trace_id: str,
        span_id: str
    ) -> None:
        """Log audit information."""
        if self.audit_callback:
            audit_data = {
                "requestId": request.request_id,
                "traceId": trace_id,
                "spanId": span_id,
                "caller": request.caller_agent,
                "callee": request.callee_agent,
                "capability": request.callee_capability,
                "status": response.status.value,
                "timestamp": time.time(),
                "policiesApplied": request.policies_applied
            }

            # Redact sensitive fields
            if request.payload_data:
                audit_data["payload"] = redact_fields(
                    request.payload_data,
                    self.config.redact_fields
                )

            self.audit_callback(audit_data)

    def _error_response(
        self,
        request_id: str,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
        policy_refs: list[str] | None = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None
    ) -> A2AResponse:
        """Create an error response."""
        error = {
            "code": code.value,
            "message": message
        }
        if details:
            error["details"] = details
        if policy_refs:
            error["policyRefs"] = policy_refs

        return A2AResponse(
            request_id=request_id,
            status=ResponseStatus.ERROR,
            trace_id=trace_id,
            span_id=span_id,
            error=error,
            audit_ref=f"audits/orchestrator/{request_id}-error.json"
        )

    def _validation_error_response(
        self,
        request: A2ARequest,
        result: ValidationResult,
        trace_id: str,
        span_id: str
    ) -> A2AResponse:
        """Create an error response from validation result."""
        first_error = result.errors[0] if result.errors else None

        return self._error_response(
            request_id=request.request_id,
            code=ErrorCode(first_error.code.value) if first_error else ErrorCode.INTERNAL_ERROR,
            message=first_error.message if first_error else "Validation failed",
            details=first_error.details if first_error else {},
            policy_refs=first_error.policy_refs if first_error else [],
            trace_id=trace_id,
            span_id=span_id
        )
