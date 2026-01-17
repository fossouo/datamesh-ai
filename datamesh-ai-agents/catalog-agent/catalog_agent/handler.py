"""
Catalog Agent Handler

Main request handler for the Catalog Agent, implementing the catalog.resolve
and catalog.lineage capabilities.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypedDict

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel, Field, ValidationError

from catalog_agent.resolver import SchemaResolver, SchemaResolutionError
from catalog_agent.lineage import LineageTracker, LineageTrackingError

# Configure structured logging
logger = logging.getLogger(__name__)

# Initialize tracer for observability
tracer = trace.get_tracer("catalog_agent", "1.0.0")


class ResolveRequest(BaseModel):
    """Request model for catalog.resolve capability."""

    dataset_uri: str = Field(
        ...,
        description="The catalog URI of the dataset",
        pattern=r"^catalog://[a-z_]+\.[a-z_]+$"
    )


class LineageRequest(BaseModel):
    """Request model for catalog.lineage capability."""

    dataset_uri: str = Field(
        ...,
        description="The catalog URI of the dataset",
        pattern=r"^catalog://[a-z_]+\.[a-z_]+$"
    )
    depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum depth of lineage traversal"
    )


class AgentRequest(TypedDict):
    """Typed dictionary for incoming agent requests."""

    capability: str
    payload: dict[str, Any]
    trace_context: dict[str, str] | None
    caller: str | None


class AgentResponse(TypedDict):
    """Typed dictionary for agent responses."""

    success: bool
    data: dict[str, Any] | None
    error: str | None
    trace_id: str | None


# Initialize resolver and lineage tracker
_resolver = SchemaResolver()
_lineage_tracker = LineageTracker()


def handle_request(request: AgentRequest) -> AgentResponse:
    """
    Main entry point for handling Catalog Agent requests.

    Routes requests to the appropriate capability handler based on
    the capability field in the request.

    Args:
        request: The incoming agent request containing capability and payload.

    Returns:
        AgentResponse containing the result or error information.
    """
    with tracer.start_as_current_span("handle_request") as span:
        capability = request.get("capability", "")
        payload = request.get("payload", {})
        caller = request.get("caller")

        span.set_attribute("capability", capability)
        span.set_attribute("caller", caller or "unknown")

        logger.info(
            "Processing request",
            extra={
                "capability": capability,
                "caller": caller,
                "payload_keys": list(payload.keys())
            }
        )

        try:
            if capability == "catalog.resolve":
                return _handle_resolve(payload, span)
            elif capability == "catalog.lineage":
                return _handle_lineage(payload, span)
            else:
                span.set_status(Status(StatusCode.ERROR, "Unknown capability"))
                return AgentResponse(
                    success=False,
                    data=None,
                    error=f"Unknown capability: {capability}",
                    trace_id=_get_trace_id(span)
                )

        except Exception as e:
            logger.exception("Unexpected error handling request")
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            return AgentResponse(
                success=False,
                data=None,
                error=f"Internal error: {str(e)}",
                trace_id=_get_trace_id(span)
            )


def _handle_resolve(payload: dict[str, Any], parent_span: Span) -> AgentResponse:
    """
    Handle catalog.resolve capability requests.

    Args:
        payload: Request payload containing dataset_uri.
        parent_span: Parent OpenTelemetry span for tracing.

    Returns:
        AgentResponse with schema metadata or error.
    """
    with tracer.start_as_current_span("resolve_schema", parent=parent_span) as span:
        try:
            # Validate request
            request = ResolveRequest(**payload)
            span.set_attribute("dataset_uri", request.dataset_uri)

            # Resolve schema
            schema = _resolver.resolve(request.dataset_uri)

            response_data = {
                "dataset_uri": request.dataset_uri,
                "schema": schema,
                "resolved_at": datetime.now(timezone.utc).isoformat()
            }

            span.set_status(Status(StatusCode.OK))
            logger.info(
                "Schema resolved successfully",
                extra={
                    "dataset_uri": request.dataset_uri,
                    "field_count": len(schema)
                }
            )

            return AgentResponse(
                success=True,
                data=response_data,
                error=None,
                trace_id=_get_trace_id(span)
            )

        except ValidationError as e:
            span.set_status(Status(StatusCode.ERROR, "Validation error"))
            return AgentResponse(
                success=False,
                data=None,
                error=f"Validation error: {e.errors()}",
                trace_id=_get_trace_id(span)
            )

        except SchemaResolutionError as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            return AgentResponse(
                success=False,
                data=None,
                error=str(e),
                trace_id=_get_trace_id(span)
            )


def _handle_lineage(payload: dict[str, Any], parent_span: Span) -> AgentResponse:
    """
    Handle catalog.lineage capability requests.

    Args:
        payload: Request payload containing dataset_uri and optional depth.
        parent_span: Parent OpenTelemetry span for tracing.

    Returns:
        AgentResponse with lineage information or error.
    """
    with tracer.start_as_current_span("trace_lineage", parent=parent_span) as span:
        try:
            # Validate request
            request = LineageRequest(**payload)
            span.set_attribute("dataset_uri", request.dataset_uri)
            span.set_attribute("depth", request.depth)

            # Get lineage information
            lineage = _lineage_tracker.get_lineage(
                request.dataset_uri,
                depth=request.depth
            )

            response_data = {
                "dataset_uri": request.dataset_uri,
                "upstream": lineage["upstream"],
                "downstream": lineage["downstream"],
                "lineage_depth": request.depth,
                "traced_at": datetime.now(timezone.utc).isoformat()
            }

            span.set_status(Status(StatusCode.OK))
            logger.info(
                "Lineage traced successfully",
                extra={
                    "dataset_uri": request.dataset_uri,
                    "upstream_count": len(lineage["upstream"]),
                    "downstream_count": len(lineage["downstream"])
                }
            )

            return AgentResponse(
                success=True,
                data=response_data,
                error=None,
                trace_id=_get_trace_id(span)
            )

        except ValidationError as e:
            span.set_status(Status(StatusCode.ERROR, "Validation error"))
            return AgentResponse(
                success=False,
                data=None,
                error=f"Validation error: {e.errors()}",
                trace_id=_get_trace_id(span)
            )

        except LineageTrackingError as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            return AgentResponse(
                success=False,
                data=None,
                error=str(e),
                trace_id=_get_trace_id(span)
            )


def _get_trace_id(span: Span) -> str | None:
    """Extract trace ID from span context."""
    context = span.get_span_context()
    if context.is_valid:
        return format(context.trace_id, '032x')
    return None


async def health_check() -> dict[str, Any]:
    """
    Health check endpoint for the Catalog Agent.

    Returns:
        Dictionary containing health status and metadata.
    """
    return {
        "status": "healthy",
        "agent": "catalog-agent",
        "version": "1.0.0",
        "capabilities": ["catalog.resolve", "catalog.lineage"],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
