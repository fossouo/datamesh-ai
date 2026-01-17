"""
Pydantic models for the A2A (Agent-to-Agent) protocol.

This module defines the core data models used for agent communication
following the datamesh.ai/a2a/v1 protocol specification.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class A2AStatus(str, Enum):
    """Status codes for A2A responses."""

    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"
    PENDING = "PENDING"
    TIMEOUT = "TIMEOUT"
    REJECTED = "REJECTED"


class TraceContext(BaseModel):
    """
    W3C Trace Context for distributed tracing.

    Follows the W3C Trace Context specification for propagating
    trace information across agent boundaries.
    """

    trace_id: str = Field(
        ...,
        description="Unique identifier for the entire trace (32 hex characters)",
        min_length=32,
        max_length=32,
    )
    span_id: str = Field(
        ...,
        description="Unique identifier for this span (16 hex characters)",
        min_length=16,
        max_length=16,
    )
    parent_span_id: Optional[str] = Field(
        default=None,
        description="Parent span identifier for hierarchy",
        min_length=16,
        max_length=16,
    )
    trace_flags: int = Field(
        default=1,
        description="Trace flags (1 = sampled)",
        ge=0,
        le=255,
    )
    trace_state: Optional[str] = Field(
        default=None,
        description="Vendor-specific trace information",
    )

    def to_traceparent(self) -> str:
        """
        Convert to W3C traceparent header format.

        Returns:
            W3C traceparent string in format: version-trace_id-span_id-flags
        """
        return f"00-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    @classmethod
    def from_traceparent(cls, traceparent: str) -> "TraceContext":
        """
        Parse W3C traceparent header.

        Args:
            traceparent: W3C traceparent header string

        Returns:
            TraceContext instance

        Raises:
            ValueError: If traceparent format is invalid
        """
        parts = traceparent.split("-")
        if len(parts) != 4:
            raise ValueError(f"Invalid traceparent format: {traceparent}")

        version, trace_id, span_id, flags = parts
        if version != "00":
            raise ValueError(f"Unsupported traceparent version: {version}")

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=int(flags, 16),
        )


class AgentCapability(BaseModel):
    """
    Describes a capability that an agent can provide.

    Capabilities are used for agent discovery and routing.
    """

    name: str = Field(
        ...,
        description="Unique capability name",
        min_length=1,
        max_length=255,
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version of the capability",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description",
    )
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON Schema for capability input",
    )
    output_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON Schema for capability output",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization and discovery",
    )


class AgentMetadata(BaseModel):
    """
    Metadata describing an agent's identity and capabilities.

    Used for agent registration and discovery in the registry.
    """

    agent_id: str = Field(
        ...,
        description="Unique agent identifier",
        min_length=1,
        max_length=255,
    )
    name: str = Field(
        ...,
        description="Human-readable agent name",
        min_length=1,
        max_length=255,
    )
    version: str = Field(
        default="1.0.0",
        description="Agent version (semantic versioning)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description",
    )
    owner: Optional[str] = Field(
        default=None,
        description="Team or individual responsible for the agent",
    )
    capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="List of capabilities this agent provides",
    )
    endpoint: Optional[str] = Field(
        default=None,
        description="HTTP endpoint for A2A communication",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata",
    )


class A2ARequest(BaseModel):
    """
    A2A protocol request message.

    This is the standard message format for agent-to-agent communication
    following the datamesh.ai/a2a/v1 protocol.
    """

    protocol_version: str = Field(
        default="datamesh.ai/a2a/v1",
        description="Protocol version identifier",
        alias="protocolVersion",
    )
    request_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique request identifier",
        alias="requestId",
    )
    source_agent: str = Field(
        ...,
        description="Identifier of the requesting agent",
        alias="sourceAgent",
    )
    target_agent: str = Field(
        ...,
        description="Identifier of the target agent",
        alias="targetAgent",
    )
    capability: str = Field(
        ...,
        description="Name of the capability being invoked",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Request payload/parameters",
    )
    deadline_ms: Optional[int] = Field(
        default=None,
        description="Request deadline in milliseconds from now",
        alias="deadlineMs",
        ge=0,
    )
    trace_context: Optional[TraceContext] = Field(
        default=None,
        description="Distributed tracing context",
        alias="traceContext",
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for request chains",
        alias="correlationId",
    )
    priority: int = Field(
        default=5,
        description="Request priority (1=highest, 10=lowest)",
        ge=1,
        le=10,
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Request timestamp (UTC)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata",
    )

    class Config:
        """Pydantic configuration."""

        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z",
            UUID: str,
        }


class A2AResponse(BaseModel):
    """
    A2A protocol response message.

    This is the standard response format for agent-to-agent communication
    following the datamesh.ai/a2a/v1 protocol.
    """

    protocol_version: str = Field(
        default="datamesh.ai/a2a/v1",
        description="Protocol version identifier",
        alias="protocolVersion",
    )
    request_id: str = Field(
        ...,
        description="Request ID this response corresponds to",
        alias="requestId",
    )
    source_agent: str = Field(
        ...,
        description="Identifier of the responding agent",
        alias="sourceAgent",
    )
    target_agent: str = Field(
        ...,
        description="Identifier of the original requesting agent",
        alias="targetAgent",
    )
    status: A2AStatus = Field(
        ...,
        description="Response status",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response payload/result",
    )
    error: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Error details if status is ERROR",
    )
    trace_context: Optional[TraceContext] = Field(
        default=None,
        description="Distributed tracing context",
        alias="traceContext",
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID from request",
        alias="correlationId",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp (UTC)",
    )
    duration_ms: Optional[int] = Field(
        default=None,
        description="Processing duration in milliseconds",
        alias="durationMs",
        ge=0,
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata",
    )

    class Config:
        """Pydantic configuration."""

        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z",
            UUID: str,
        }

    @classmethod
    def success(
        cls,
        request: A2ARequest,
        payload: Dict[str, Any],
        duration_ms: Optional[int] = None,
    ) -> "A2AResponse":
        """
        Create a success response for a request.

        Args:
            request: The original A2A request
            payload: Response payload
            duration_ms: Processing duration in milliseconds

        Returns:
            A2AResponse with SUCCESS status
        """
        return cls(
            request_id=request.request_id,
            source_agent=request.target_agent,
            target_agent=request.source_agent,
            status=A2AStatus.SUCCESS,
            payload=payload,
            trace_context=request.trace_context,
            correlation_id=request.correlation_id,
            duration_ms=duration_ms,
        )

    @classmethod
    def error(
        cls,
        request: A2ARequest,
        error_code: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None,
    ) -> "A2AResponse":
        """
        Create an error response for a request.

        Args:
            request: The original A2A request
            error_code: Error code identifier
            error_message: Human-readable error message
            error_details: Additional error details
            duration_ms: Processing duration in milliseconds

        Returns:
            A2AResponse with ERROR status
        """
        return cls(
            request_id=request.request_id,
            source_agent=request.target_agent,
            target_agent=request.source_agent,
            status=A2AStatus.ERROR,
            error={
                "code": error_code,
                "message": error_message,
                "details": error_details or {},
            },
            trace_context=request.trace_context,
            correlation_id=request.correlation_id,
            duration_ms=duration_ms,
        )

    @classmethod
    def in_progress(
        cls,
        request: A2ARequest,
        progress_info: Optional[Dict[str, Any]] = None,
    ) -> "A2AResponse":
        """
        Create an in-progress response for long-running operations.

        Args:
            request: The original A2A request
            progress_info: Optional progress information

        Returns:
            A2AResponse with IN_PROGRESS status
        """
        return cls(
            request_id=request.request_id,
            source_agent=request.target_agent,
            target_agent=request.source_agent,
            status=A2AStatus.IN_PROGRESS,
            payload=progress_info or {},
            trace_context=request.trace_context,
            correlation_id=request.correlation_id,
        )
