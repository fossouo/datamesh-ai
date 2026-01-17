"""
A2A Message - DATAMESH.AI A2A Protocol v1 message definitions.

Implements the standard payload structure for Agent-to-Agent communication.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import json
import uuid


PROTOCOL_VERSION = "datamesh.ai/a2a/v1"


class A2AStatus(Enum):
    """A2A response status."""
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    IN_PROGRESS = "IN_PROGRESS"


@dataclass
class TraceContext:
    """W3C/OpenTelemetry trace context."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None

    @classmethod
    def new(cls, parent: Optional["TraceContext"] = None) -> "TraceContext":
        """Create a new trace context, optionally as child of parent."""
        trace_id = parent.trace_id if parent else uuid.uuid4().hex
        return cls(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent.span_id if parent else None,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "TraceContext":
        return cls(
            trace_id=data["traceId"],
            span_id=data["spanId"],
            parent_span_id=data.get("parentSpanId"),
        )

    def to_dict(self) -> dict:
        result = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
        }
        if self.parent_span_id:
            result["parentSpanId"] = self.parent_span_id
        return result


@dataclass
class CallerInfo:
    """Caller agent information."""
    agent: str
    capability: str

    @classmethod
    def from_dict(cls, data: dict) -> "CallerInfo":
        return cls(agent=data["agent"], capability=data["capability"])

    def to_dict(self) -> dict:
        return {"agent": self.agent, "capability": self.capability}


@dataclass
class CalleeInfo:
    """Callee agent information."""
    agent: str
    capability: str

    @classmethod
    def from_dict(cls, data: dict) -> "CalleeInfo":
        return cls(agent=data["agent"], capability=data["capability"])

    def to_dict(self) -> dict:
        return {"agent": self.agent, "capability": self.capability}


@dataclass
class OnBehalfOf:
    """Delegation context."""
    user_id: str
    roles: list[str] = field(default_factory=list)
    delegation_chain: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "OnBehalfOf":
        return cls(
            user_id=data["userId"],
            roles=data.get("roles", []),
            delegation_chain=data.get("delegationChain", []),
        )

    def to_dict(self) -> dict:
        return {
            "userId": self.user_id,
            "roles": self.roles,
            "delegationChain": self.delegation_chain,
        }


@dataclass
class CallConstraints:
    """A2A call constraints."""
    max_depth: int = 3
    current_depth: int = 1

    @classmethod
    def from_dict(cls, data: dict) -> "CallConstraints":
        return cls(
            max_depth=data.get("maxDepth", 3),
            current_depth=data.get("currentDepth", 1),
        )

    def to_dict(self) -> dict:
        return {"maxDepth": self.max_depth, "currentDepth": self.current_depth}


@dataclass
class Signature:
    """Message signature for mutual attestation."""
    alg: str = "ed25519"
    value: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Signature":
        return cls(alg=data.get("alg", "ed25519"), value=data.get("value", ""))

    def to_dict(self) -> dict:
        return {"alg": self.alg, "value": self.value}


@dataclass
class RequestContext:
    """A2A request context."""
    on_behalf_of: Optional[OnBehalfOf] = None
    policies_applied: list[str] = field(default_factory=list)
    constraints: Optional[CallConstraints] = None
    auth_context_ref: Optional[str] = None
    signature: Optional[Signature] = None

    @classmethod
    def from_dict(cls, data: dict) -> "RequestContext":
        return cls(
            on_behalf_of=OnBehalfOf.from_dict(data["onBehalfOf"]) if "onBehalfOf" in data else None,
            policies_applied=data.get("policiesApplied", []),
            constraints=CallConstraints.from_dict(data["constraints"]) if "constraints" in data else None,
            auth_context_ref=data.get("authContextRef"),
            signature=Signature.from_dict(data["signature"]) if "signature" in data else None,
        )

    def to_dict(self) -> dict:
        result = {
            "policiesApplied": self.policies_applied,
        }
        if self.on_behalf_of:
            result["onBehalfOf"] = self.on_behalf_of.to_dict()
        if self.constraints:
            result["constraints"] = self.constraints.to_dict()
        if self.auth_context_ref:
            result["authContextRef"] = self.auth_context_ref
        if self.signature:
            result["signature"] = self.signature.to_dict()
        return result


@dataclass
class PayloadRef:
    """Payload reference with schemas."""
    input_schema: str
    output_schema: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "PayloadRef":
        return cls(
            input_schema=data.get("inputSchema", ""),
            output_schema=data.get("outputSchema", ""),
            data=data.get("data", {}),
        )

    def to_dict(self) -> dict:
        return {
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "data": self.data,
        }


@dataclass
class A2AMessage:
    """
    DATAMESH.AI A2A Protocol v1 Message.

    Standard payload for Agent-to-Agent communication.
    """
    request_id: str
    trace: TraceContext
    caller: CallerInfo
    callee: CalleeInfo
    payload_ref: PayloadRef
    deadline_ms: int = 15000
    context: Optional[RequestContext] = None
    protocol_version: str = PROTOCOL_VERSION
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def new(
        cls,
        caller_agent: str,
        caller_capability: str,
        callee_agent: str,
        callee_capability: str,
        data: dict,
        input_schema: str = "",
        output_schema: str = "",
        deadline_ms: int = 15000,
        parent_trace: Optional[TraceContext] = None,
        on_behalf_of: Optional[OnBehalfOf] = None,
        policies: Optional[list[str]] = None,
        current_depth: int = 1,
        max_depth: int = 3,
    ) -> "A2AMessage":
        """Create a new A2A message."""
        return cls(
            request_id=f"req-{uuid.uuid4().hex[:8]}",
            trace=TraceContext.new(parent_trace),
            caller=CallerInfo(agent=caller_agent, capability=caller_capability),
            callee=CalleeInfo(agent=callee_agent, capability=callee_capability),
            payload_ref=PayloadRef(
                input_schema=input_schema,
                output_schema=output_schema,
                data=data,
            ),
            deadline_ms=deadline_ms,
            context=RequestContext(
                on_behalf_of=on_behalf_of,
                policies_applied=policies or [],
                constraints=CallConstraints(max_depth=max_depth, current_depth=current_depth),
            ),
        )

    @classmethod
    def from_dict(cls, data: dict) -> "A2AMessage":
        """Parse A2A message from dictionary."""
        return cls(
            protocol_version=data.get("protocolVersion", PROTOCOL_VERSION),
            request_id=data["requestId"],
            deadline_ms=data.get("deadlineMs", 15000),
            trace=TraceContext.from_dict(data["trace"]),
            caller=CallerInfo.from_dict(data["caller"]),
            callee=CalleeInfo.from_dict(data["callee"]),
            context=RequestContext.from_dict(data["context"]) if "context" in data else None,
            payload_ref=PayloadRef.from_dict(data["payloadRef"]),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "A2AMessage":
        """Parse A2A message from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> dict:
        """Convert message to dictionary."""
        result = {
            "protocolVersion": self.protocol_version,
            "requestId": self.request_id,
            "deadlineMs": self.deadline_ms,
            "trace": self.trace.to_dict(),
            "caller": self.caller.to_dict(),
            "callee": self.callee.to_dict(),
            "payloadRef": self.payload_ref.to_dict(),
        }
        if self.context:
            result["context"] = self.context.to_dict()
        return result

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def create_child_message(
        self,
        callee_agent: str,
        callee_capability: str,
        data: dict,
        input_schema: str = "",
        output_schema: str = "",
    ) -> "A2AMessage":
        """Create a child message for delegation."""
        current_depth = 1
        max_depth = 3
        if self.context and self.context.constraints:
            current_depth = self.context.constraints.current_depth + 1
            max_depth = self.context.constraints.max_depth

        # Build delegation chain
        delegation_chain = []
        if self.context and self.context.on_behalf_of:
            delegation_chain = self.context.on_behalf_of.delegation_chain.copy()
        delegation_chain.append(self.caller.agent)

        on_behalf_of = None
        if self.context and self.context.on_behalf_of:
            on_behalf_of = OnBehalfOf(
                user_id=self.context.on_behalf_of.user_id,
                roles=self.context.on_behalf_of.roles,
                delegation_chain=delegation_chain,
            )

        policies = self.context.policies_applied if self.context else []

        return A2AMessage.new(
            caller_agent=self.callee.agent,  # Current callee becomes caller
            caller_capability=self.callee.capability,
            callee_agent=callee_agent,
            callee_capability=callee_capability,
            data=data,
            input_schema=input_schema,
            output_schema=output_schema,
            deadline_ms=self.deadline_ms,
            parent_trace=self.trace,
            on_behalf_of=on_behalf_of,
            policies=policies,
            current_depth=current_depth,
            max_depth=max_depth,
        )


@dataclass
class A2AError:
    """A2A error details."""
    code: str
    message: str
    policy_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "A2AError":
        return cls(
            code=data["code"],
            message=data["message"],
            policy_refs=data.get("policyRefs", []),
        )

    def to_dict(self) -> dict:
        result = {"code": self.code, "message": self.message}
        if self.policy_refs:
            result["policyRefs"] = self.policy_refs
        return result


@dataclass
class A2AOutput:
    """A2A response output."""
    schema: str
    data: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict) -> "A2AOutput":
        return cls(schema=data.get("schema", ""), data=data.get("data", {}))

    def to_dict(self) -> dict:
        return {"schema": self.schema, "data": self.data}


@dataclass
class A2AResponse:
    """
    DATAMESH.AI A2A Protocol v1 Response.

    Standard response for Agent-to-Agent communication.
    """
    request_id: str
    status: A2AStatus
    trace: TraceContext
    audit_ref: str
    output: Optional[A2AOutput] = None
    error: Optional[A2AError] = None
    next_poll_after_ms: Optional[int] = None

    @classmethod
    def success(
        cls,
        request_id: str,
        trace: TraceContext,
        data: dict,
        schema: str = "",
        audit_ref: str = "",
    ) -> "A2AResponse":
        """Create a success response."""
        return cls(
            request_id=request_id,
            status=A2AStatus.SUCCESS,
            trace=trace,
            audit_ref=audit_ref or f"audits/{request_id}.json",
            output=A2AOutput(schema=schema, data=data),
        )

    @classmethod
    def error(
        cls,
        request_id: str,
        trace: TraceContext,
        code: str,
        message: str,
        policy_refs: Optional[list[str]] = None,
        audit_ref: str = "",
    ) -> "A2AResponse":
        """Create an error response."""
        return cls(
            request_id=request_id,
            status=A2AStatus.ERROR,
            trace=trace,
            audit_ref=audit_ref or f"audits/{request_id}-error.json",
            error=A2AError(code=code, message=message, policy_refs=policy_refs or []),
        )

    @classmethod
    def in_progress(
        cls,
        request_id: str,
        trace: TraceContext,
        next_poll_after_ms: int = 2000,
        audit_ref: str = "",
    ) -> "A2AResponse":
        """Create an in-progress response for async operations."""
        return cls(
            request_id=request_id,
            status=A2AStatus.IN_PROGRESS,
            trace=trace,
            audit_ref=audit_ref or f"audits/{request_id}-stage.json",
            next_poll_after_ms=next_poll_after_ms,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "A2AResponse":
        """Parse response from dictionary."""
        return cls(
            request_id=data["requestId"],
            status=A2AStatus(data["status"]),
            trace=TraceContext.from_dict(data["trace"]),
            audit_ref=data.get("auditRef", ""),
            output=A2AOutput.from_dict(data["output"]) if "output" in data else None,
            error=A2AError.from_dict(data["error"]) if "error" in data else None,
            next_poll_after_ms=data.get("nextPollAfterMs"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "A2AResponse":
        """Parse response from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> dict:
        """Convert response to dictionary."""
        result = {
            "requestId": self.request_id,
            "status": self.status.value,
            "trace": self.trace.to_dict(),
            "auditRef": self.audit_ref,
        }
        if self.output:
            result["output"] = self.output.to_dict()
        if self.error:
            result["error"] = self.error.to_dict()
        if self.next_poll_after_ms:
            result["nextPollAfterMs"] = self.next_poll_after_ms
        return result

    def to_json(self) -> str:
        """Convert response to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @property
    def is_success(self) -> bool:
        return self.status == A2AStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        return self.status == A2AStatus.ERROR

    @property
    def is_in_progress(self) -> bool:
        return self.status == A2AStatus.IN_PROGRESS
