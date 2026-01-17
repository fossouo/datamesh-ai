"""
Tracing Context - OpenTelemetry context propagation for A2A calls.

Provides W3C Trace Context compatible tracing for distributed agent systems.
"""

import uuid
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Context variable for current trace
_current_trace: ContextVar[Optional["TracingContext"]] = ContextVar(
    "current_trace", default=None
)


@dataclass
class SpanContext:
    """Individual span context within a trace."""
    span_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    status: str = "UNSET"
    status_message: str = ""

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        })

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_status(self, status: str, message: str = "") -> None:
        """Set span status (OK, ERROR, UNSET)."""
        self.status = status
        self.status_message = message

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.utcnow()

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if not self.end_time:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def to_dict(self) -> dict:
        """Convert span to dictionary for export."""
        return {
            "spanId": self.span_id,
            "name": self.name,
            "startTime": self.start_time.isoformat(),
            "endTime": self.end_time.isoformat() if self.end_time else None,
            "durationMs": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "statusMessage": self.status_message,
        }


@dataclass
class TracingContext:
    """
    Tracing context for distributed agent operations.

    Follows W3C Trace Context specification for propagation
    and OpenTelemetry semantics for spans.
    """
    trace_id: str
    spans: list[SpanContext] = field(default_factory=list)
    parent_span_id: Optional[str] = None
    baggage: dict[str, str] = field(default_factory=dict)
    _current_span: Optional[SpanContext] = None

    @classmethod
    def new(cls, parent: Optional["TracingContext"] = None) -> "TracingContext":
        """Create a new tracing context."""
        if parent:
            return cls(
                trace_id=parent.trace_id,
                parent_span_id=parent.current_span_id,
                baggage=parent.baggage.copy(),
            )
        return cls(trace_id=uuid.uuid4().hex)

    @classmethod
    def from_w3c_header(cls, traceparent: str, tracestate: str = "") -> "TracingContext":
        """
        Parse W3C traceparent header.

        Format: 00-{trace_id}-{parent_span_id}-{flags}
        """
        try:
            parts = traceparent.split("-")
            if len(parts) >= 3:
                return cls(
                    trace_id=parts[1],
                    parent_span_id=parts[2] if parts[2] != "0" * 16 else None,
                )
        except Exception as e:
            logger.warning(f"Failed to parse traceparent: {e}")

        return cls.new()

    @property
    def current_span_id(self) -> Optional[str]:
        """Get current span ID."""
        if self._current_span:
            return self._current_span.span_id
        if self.spans:
            return self.spans[-1].span_id
        return None

    def start_span(
        self,
        name: str,
        attributes: Optional[dict] = None,
    ) -> SpanContext:
        """Start a new span."""
        span = SpanContext(
            span_id=uuid.uuid4().hex[:16],
            name=name,
            start_time=datetime.utcnow(),
            attributes=attributes or {},
        )
        self.spans.append(span)
        self._current_span = span
        return span

    def end_span(self, status: str = "OK", message: str = "") -> Optional[SpanContext]:
        """End the current span."""
        if self._current_span:
            self._current_span.set_status(status, message)
            self._current_span.end()
            ended_span = self._current_span
            self._current_span = None
            return ended_span
        return None

    def to_w3c_header(self) -> str:
        """
        Generate W3C traceparent header.

        Format: 00-{trace_id}-{span_id}-01
        """
        span_id = self.current_span_id or "0" * 16
        return f"00-{self.trace_id}-{span_id}-01"

    def to_dict(self) -> dict:
        """Convert context to dictionary for A2A propagation."""
        return {
            "traceId": self.trace_id,
            "spanId": self.current_span_id,
            "parentSpanId": self.parent_span_id,
        }

    def get_export_data(self) -> dict:
        """Get full trace data for export to collector."""
        return {
            "traceId": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "baggage": self.baggage,
        }

    def set_baggage(self, key: str, value: str) -> None:
        """Set baggage item for propagation."""
        self.baggage[key] = value

    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage item."""
        return self.baggage.get(key)


def get_current_trace() -> Optional[TracingContext]:
    """Get the current trace context from context variable."""
    return _current_trace.get()


def set_current_trace(ctx: TracingContext) -> None:
    """Set the current trace context."""
    _current_trace.set(ctx)


class TraceScope:
    """Context manager for trace scopes."""

    def __init__(
        self,
        name: str,
        parent: Optional[TracingContext] = None,
        attributes: Optional[dict] = None,
    ):
        self.name = name
        self.parent = parent
        self.attributes = attributes or {}
        self.context: Optional[TracingContext] = None
        self.span: Optional[SpanContext] = None
        self._token = None

    def __enter__(self) -> "TraceScope":
        self.context = TracingContext.new(self.parent or get_current_trace())
        self.span = self.context.start_span(self.name, self.attributes)
        self._token = _current_trace.set(self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.context.end_span("ERROR", str(exc_val))
        else:
            self.context.end_span("OK")

        if self._token:
            _current_trace.reset(self._token)

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        """Add event to current span."""
        if self.span:
            self.span.add_event(name, attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set attribute on current span."""
        if self.span:
            self.span.set_attribute(key, value)
