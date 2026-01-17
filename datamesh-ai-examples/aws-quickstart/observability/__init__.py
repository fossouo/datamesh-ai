"""
DataMesh.AI Observability Module

Provides:
- OpenTelemetry metrics and tracing
- Prometheus-compatible metrics export
- Distributed tracing across A2A calls
"""

from .otel_metrics import (
    MetricsCollector,
    OTelConfig,
    init_metrics,
    get_metrics,
    traced,
    create_traced_call_agent,
    OTEL_AVAILABLE,
)

__all__ = [
    "MetricsCollector",
    "OTelConfig",
    "init_metrics",
    "get_metrics",
    "traced",
    "create_traced_call_agent",
    "OTEL_AVAILABLE",
]
