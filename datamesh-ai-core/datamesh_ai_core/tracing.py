"""
OpenTelemetry tracing utilities.

This module provides utilities for distributed tracing in the
DATAMESH.AI ecosystem, following W3C Trace Context standards.
"""

import logging
import os
import secrets
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional, Generator

from datamesh_ai_core.models import TraceContext


def generate_trace_id() -> str:
    """
    Generate a random trace ID (32 hex characters).

    Returns:
        Random trace ID
    """
    return secrets.token_hex(16)


def generate_span_id() -> str:
    """
    Generate a random span ID (16 hex characters).

    Returns:
        Random span ID
    """
    return secrets.token_hex(8)


def create_trace_context(
    parent_context: Optional[TraceContext] = None,
) -> TraceContext:
    """
    Create a new trace context.

    If a parent context is provided, the new context will be a child span.
    Otherwise, a new root trace is created.

    Args:
        parent_context: Optional parent trace context

    Returns:
        New TraceContext
    """
    if parent_context:
        return TraceContext(
            trace_id=parent_context.trace_id,
            span_id=generate_span_id(),
            parent_span_id=parent_context.span_id,
            trace_flags=parent_context.trace_flags,
            trace_state=parent_context.trace_state,
        )
    else:
        return TraceContext(
            trace_id=generate_trace_id(),
            span_id=generate_span_id(),
        )


class SpanContext:
    """
    Context for a tracing span.

    This class manages the lifecycle of a single span within a trace.
    """

    def __init__(
        self,
        name: str,
        trace_context: TraceContext,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize span context.

        Args:
            name: Span name/operation
            trace_context: Trace context for this span
            logger: Optional logger for span events
        """
        self.name = name
        self.trace_context = trace_context
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.status = "OK"
        self.error: Optional[Exception] = None
        self.attributes: Dict[str, Any] = {}
        self.events: list = []

        self._logger = logger or logging.getLogger("datamesh.tracing")

    @property
    def duration_ms(self) -> Optional[int]:
        """Get span duration in milliseconds."""
        if self.end_time:
            delta = self.end_time - self.start_time
            return int(delta.total_seconds() * 1000)
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set a span attribute.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """
        Set multiple span attributes.

        Args:
            attributes: Dictionary of attributes
        """
        self.attributes.update(attributes)

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an event to the span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        })

    def set_error(self, error: Exception) -> None:
        """
        Mark the span as errored.

        Args:
            error: The exception that occurred
        """
        self.status = "ERROR"
        self.error = error
        self.set_attribute("error.type", type(error).__name__)
        self.set_attribute("error.message", str(error))

    def end(self) -> None:
        """End the span and record duration."""
        self.end_time = datetime.utcnow()

        self._logger.debug(
            f"Span ended: {self.name} "
            f"(trace_id={self.trace_context.trace_id}, "
            f"span_id={self.trace_context.span_id}, "
            f"duration={self.duration_ms}ms, "
            f"status={self.status})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.trace_context.trace_id,
            "span_id": self.trace_context.span_id,
            "parent_span_id": self.trace_context.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class TracingManager:
    """
    Manager for distributed tracing.

    This class provides:
    - Span creation and management
    - Context propagation
    - OpenTelemetry integration (optional)

    When OpenTelemetry is available, it delegates to the OTel SDK.
    Otherwise, it provides a lightweight local implementation.
    """

    def __init__(
        self,
        service_name: str = "datamesh-agent",
        enable_otel: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the tracing manager.

        Args:
            service_name: Name of the service for tracing
            enable_otel: Whether to enable OpenTelemetry integration
            logger: Optional logger instance
        """
        self._service_name = service_name
        self._enable_otel = enable_otel
        self._logger = logger or logging.getLogger("datamesh.tracing")

        # Active spans stack (for nested spans)
        self._active_spans: list = []

        # OpenTelemetry tracer (lazy initialized)
        self._otel_tracer = None

        if enable_otel:
            self._init_otel()

    def _init_otel(self) -> None:
        """Initialize OpenTelemetry integration."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": self._service_name})
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)

            self._otel_tracer = trace.get_tracer(self._service_name)
            self._logger.info("OpenTelemetry tracing initialized")

        except ImportError:
            self._logger.warning(
                "OpenTelemetry not available, using local tracing"
            )
            self._enable_otel = False

    @property
    def current_span(self) -> Optional[SpanContext]:
        """Get the current active span."""
        return self._active_spans[-1] if self._active_spans else None

    @property
    def current_trace_context(self) -> Optional[TraceContext]:
        """Get the current trace context."""
        span = self.current_span
        return span.trace_context if span else None

    @contextmanager
    def start_span(
        self,
        name: str,
        trace_context: Optional[TraceContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[SpanContext, None, None]:
        """
        Start a new span.

        This is a context manager that automatically ends the span
        when the context exits.

        Args:
            name: Span name/operation
            trace_context: Optional trace context (creates new if None)
            attributes: Initial span attributes

        Yields:
            SpanContext for the new span
        """
        # Create or inherit trace context
        if trace_context:
            span_context = create_trace_context(trace_context)
        elif self.current_span:
            span_context = create_trace_context(self.current_span.trace_context)
        else:
            span_context = create_trace_context()

        # Create span
        span = SpanContext(
            name=name,
            trace_context=span_context,
            logger=self._logger,
        )

        if attributes:
            span.set_attributes(attributes)

        # Push to active spans
        self._active_spans.append(span)

        self._logger.debug(
            f"Span started: {name} "
            f"(trace_id={span_context.trace_id}, "
            f"span_id={span_context.span_id})"
        )

        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            span.end()
            self._active_spans.pop()

    def inject_context(
        self,
        headers: Dict[str, str],
        trace_context: Optional[TraceContext] = None,
    ) -> Dict[str, str]:
        """
        Inject trace context into headers for propagation.

        Args:
            headers: Headers dictionary to inject into
            trace_context: Trace context to inject (uses current if None)

        Returns:
            Updated headers dictionary
        """
        ctx = trace_context or self.current_trace_context
        if ctx:
            headers["traceparent"] = ctx.to_traceparent()
            if ctx.trace_state:
                headers["tracestate"] = ctx.trace_state
        return headers

    def extract_context(
        self,
        headers: Dict[str, str],
    ) -> Optional[TraceContext]:
        """
        Extract trace context from headers.

        Args:
            headers: Headers dictionary to extract from

        Returns:
            TraceContext or None if not found
        """
        traceparent = headers.get("traceparent")
        if not traceparent:
            return None

        try:
            ctx = TraceContext.from_traceparent(traceparent)
            ctx.trace_state = headers.get("tracestate")
            return ctx
        except ValueError:
            self._logger.warning(f"Invalid traceparent: {traceparent}")
            return None

    def create_child_context(
        self,
        parent: Optional[TraceContext] = None,
    ) -> TraceContext:
        """
        Create a child trace context.

        Args:
            parent: Parent context (uses current if None)

        Returns:
            New child TraceContext
        """
        parent = parent or self.current_trace_context
        return create_trace_context(parent)


class NoOpTracingManager(TracingManager):
    """
    No-operation tracing manager for testing.

    All operations are no-ops that don't record any data.
    """

    def __init__(self):
        """Initialize no-op tracing manager."""
        super().__init__(enable_otel=False)

    @contextmanager
    def start_span(
        self,
        name: str,
        trace_context: Optional[TraceContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[SpanContext, None, None]:
        """No-op span that doesn't record anything."""
        ctx = trace_context or create_trace_context()
        span = SpanContext(name=name, trace_context=ctx)
        yield span
