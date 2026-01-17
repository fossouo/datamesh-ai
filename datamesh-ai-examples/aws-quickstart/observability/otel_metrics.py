"""
OpenTelemetry Metrics for DataMesh.AI A2A Communication

Provides distributed tracing and metrics export for:
- Agent-to-agent call latency
- Request throughput
- Error rates per agent
- Circuit breaker states

Supports exporters:
- Console (development)
- OTLP (production - compatible with Jaeger, Zipkin, etc.)
- Prometheus (for scraping)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Generator

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry packages
try:
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not installed. Run: pip install opentelemetry-sdk opentelemetry-api")


# Try to import OTLP exporter
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False


# Try to import Prometheus exporter
try:
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from prometheus_client import start_http_server as start_prometheus_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class OTelConfig:
    """OpenTelemetry configuration."""
    service_name: str = "datamesh-ai"
    service_version: str = "1.0.0"
    environment: str = "development"

    # Exporter settings
    exporter_type: str = "console"  # console, otlp, prometheus
    otlp_endpoint: str = "http://localhost:4317"
    prometheus_port: int = 9464

    # Metrics collection interval
    export_interval_ms: int = 10000


class MetricsCollector:
    """
    Collects and exports metrics for A2A communication.

    Metrics tracked:
    - a2a_requests_total: Counter of all A2A requests
    - a2a_request_duration_ms: Histogram of request latency
    - a2a_errors_total: Counter of errors by agent and type
    - circuit_breaker_state: Gauge of circuit breaker states
    """

    def __init__(self, config: OTelConfig | None = None):
        self.config = config or OTelConfig()
        self._initialized = False

        # Fallback metrics storage when OTel is not available
        self._fallback_metrics: dict[str, Any] = {
            "requests_total": 0,
            "errors_total": 0,
            "request_durations": [],
            "by_agent": {},
            "by_capability": {},
        }

        if OTEL_AVAILABLE:
            self._initialize_otel()
        else:
            logger.info("Using fallback metrics (OpenTelemetry not available)")

    def _initialize_otel(self) -> None:
        """Initialize OpenTelemetry providers and instruments."""
        # Create resource
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: self.config.service_name,
            ResourceAttributes.SERVICE_VERSION: self.config.service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.environment,
        })

        # Initialize tracer
        tracer_provider = TracerProvider(resource=resource)

        if self.config.exporter_type == "otlp" and OTLP_AVAILABLE:
            span_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
        else:
            span_exporter = ConsoleSpanExporter()

        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        self._tracer = trace.get_tracer(__name__)

        # Initialize meter
        if self.config.exporter_type == "prometheus" and PROMETHEUS_AVAILABLE:
            reader = PrometheusMetricReader()
            start_prometheus_server(port=self.config.prometheus_port)
            logger.info(f"Prometheus metrics available at :{self.config.prometheus_port}")
        elif self.config.exporter_type == "otlp" and OTLP_AVAILABLE:
            exporter = OTLPMetricExporter(endpoint=self.config.otlp_endpoint)
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self.config.export_interval_ms,
            )
        else:
            exporter = ConsoleMetricExporter()
            reader = PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self.config.export_interval_ms,
            )

        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)
        self._meter = metrics.get_meter(__name__)

        # Create instruments
        self._request_counter = self._meter.create_counter(
            name="a2a_requests_total",
            description="Total number of A2A requests",
            unit="1",
        )

        self._error_counter = self._meter.create_counter(
            name="a2a_errors_total",
            description="Total number of A2A errors",
            unit="1",
        )

        self._duration_histogram = self._meter.create_histogram(
            name="a2a_request_duration_ms",
            description="A2A request duration in milliseconds",
            unit="ms",
        )

        self._circuit_state_gauge = self._meter.create_up_down_counter(
            name="circuit_breaker_state",
            description="Circuit breaker state (0=closed, 1=open, 2=half_open)",
            unit="1",
        )

        self._initialized = True
        logger.info(f"OpenTelemetry initialized with {self.config.exporter_type} exporter")

    @contextmanager
    def trace_request(
        self,
        operation: str,
        agent: str,
        capability: str,
        trace_id: str | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Context manager for tracing an A2A request.

        Usage:
            with metrics.trace_request("call_agent", "sql-agent", "sql.execute") as span_ctx:
                result = call_agent(...)
                span_ctx["status"] = "success"
        """
        start_time = time.time()
        span_context: dict[str, Any] = {"status": "unknown", "error": None}

        if OTEL_AVAILABLE and self._initialized:
            with self._tracer.start_as_current_span(
                operation,
                attributes={
                    "a2a.agent": agent,
                    "a2a.capability": capability,
                    "a2a.trace_id": trace_id or "",
                },
            ) as span:
                try:
                    yield span_context

                    if span_context.get("error"):
                        span.set_status(Status(StatusCode.ERROR, span_context["error"]))
                    else:
                        span.set_status(Status(StatusCode.OK))

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span_context["error"] = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    span.set_attribute("duration_ms", duration_ms)
                    self._record_request(agent, capability, duration_ms, span_context)
        else:
            try:
                yield span_context
            except Exception as e:
                span_context["error"] = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                self._record_request(agent, capability, duration_ms, span_context)

    def _record_request(
        self,
        agent: str,
        capability: str,
        duration_ms: float,
        context: dict[str, Any],
    ) -> None:
        """Record request metrics."""
        attributes = {"agent": agent, "capability": capability}

        if OTEL_AVAILABLE and self._initialized:
            self._request_counter.add(1, attributes)
            self._duration_histogram.record(duration_ms, attributes)

            if context.get("error"):
                self._error_counter.add(1, {**attributes, "error_type": context.get("error_type", "unknown")})
        else:
            # Fallback metrics
            self._fallback_metrics["requests_total"] += 1
            self._fallback_metrics["request_durations"].append(duration_ms)

            # Keep only last 1000 durations
            if len(self._fallback_metrics["request_durations"]) > 1000:
                self._fallback_metrics["request_durations"] = self._fallback_metrics["request_durations"][-1000:]

            # Track by agent
            if agent not in self._fallback_metrics["by_agent"]:
                self._fallback_metrics["by_agent"][agent] = {"requests": 0, "errors": 0, "total_duration_ms": 0}
            self._fallback_metrics["by_agent"][agent]["requests"] += 1
            self._fallback_metrics["by_agent"][agent]["total_duration_ms"] += duration_ms

            if context.get("error"):
                self._fallback_metrics["errors_total"] += 1
                self._fallback_metrics["by_agent"][agent]["errors"] += 1

    def record_circuit_state(self, agent: str, state: str) -> None:
        """Record circuit breaker state change."""
        state_values = {"closed": 0, "open": 1, "half_open": 2}
        state_value = state_values.get(state, -1)

        if OTEL_AVAILABLE and self._initialized:
            self._circuit_state_gauge.add(state_value, {"agent": agent, "state": state})
        else:
            if agent not in self._fallback_metrics["by_agent"]:
                self._fallback_metrics["by_agent"][agent] = {"requests": 0, "errors": 0, "total_duration_ms": 0}
            self._fallback_metrics["by_agent"][agent]["circuit_state"] = state

    def get_stats(self) -> dict[str, Any]:
        """Get current metrics statistics."""
        if OTEL_AVAILABLE and self._initialized:
            return {
                "otel_enabled": True,
                "exporter_type": self.config.exporter_type,
                "service_name": self.config.service_name,
            }
        else:
            durations = self._fallback_metrics["request_durations"]
            return {
                "otel_enabled": False,
                "requests_total": self._fallback_metrics["requests_total"],
                "errors_total": self._fallback_metrics["errors_total"],
                "error_rate": f"{(self._fallback_metrics['errors_total'] / max(self._fallback_metrics['requests_total'], 1)) * 100:.2f}%",
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "p50_duration_ms": sorted(durations)[len(durations) // 2] if durations else 0,
                "p99_duration_ms": sorted(durations)[int(len(durations) * 0.99)] if durations else 0,
                "by_agent": self._fallback_metrics["by_agent"],
            }


def traced(operation: str | None = None):
    """
    Decorator for tracing function calls.

    Usage:
        @traced("call_sql_agent")
        def call_sql_agent(payload):
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to extract agent and capability from args/kwargs
            agent = kwargs.get("agent", kwargs.get("agent_name", "unknown"))
            capability = kwargs.get("capability", "unknown")
            trace_id = kwargs.get("trace_id", "")

            with _default_collector.trace_request(op_name, agent, capability, trace_id) as ctx:
                try:
                    result = func(*args, **kwargs)

                    # Try to extract status from result
                    if isinstance(result, dict):
                        ctx["status"] = result.get("status", "success")
                        if result.get("status") == "ERROR":
                            ctx["error"] = result.get("error", "unknown error")
                            ctx["error_type"] = result.get("errorCode", "UNKNOWN")

                    return result
                except Exception as e:
                    ctx["error"] = str(e)
                    ctx["error_type"] = type(e).__name__
                    raise

        return wrapper
    return decorator


# Default collector instance
_default_collector: MetricsCollector | None = None


def init_metrics(config: OTelConfig | None = None) -> MetricsCollector:
    """Initialize the default metrics collector."""
    global _default_collector
    _default_collector = MetricsCollector(config)
    return _default_collector


def get_metrics() -> MetricsCollector:
    """Get the default metrics collector (initializes with defaults if needed)."""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector()
    return _default_collector


# =============================================================================
# Integration helper for orchestrator
# =============================================================================

def create_traced_call_agent(original_call_agent: Callable) -> Callable:
    """
    Wrap a call_agent function with tracing.

    Usage:
        orchestrator.call_agent = create_traced_call_agent(orchestrator.call_agent)
    """
    collector = get_metrics()

    @wraps(original_call_agent)
    def traced_call_agent(
        agent_name: str,
        capability: str,
        payload: dict,
        trace_id: str,
        *args,
        **kwargs,
    ):
        with collector.trace_request("call_agent", agent_name, capability, trace_id) as ctx:
            result = original_call_agent(agent_name, capability, payload, trace_id, *args, **kwargs)

            if isinstance(result, dict):
                ctx["status"] = result.get("status", "unknown")
                if result.get("status") == "ERROR":
                    ctx["error"] = result.get("error", "unknown")
                    ctx["error_type"] = result.get("errorCode", "UNKNOWN")

            return result

    return traced_call_agent
