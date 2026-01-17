"""
Span Manager - Manages span lifecycle and export.

Provides utilities for creating, managing, and exporting spans
to OpenTelemetry collectors.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

from .context import TracingContext, SpanContext, get_current_trace

logger = logging.getLogger(__name__)


@dataclass
class SpanExporterConfig:
    """Configuration for span exporter."""
    endpoint: str = "http://localhost:4318/v1/traces"
    headers: dict[str, str] = field(default_factory=dict)
    batch_size: int = 100
    flush_interval_ms: int = 5000
    timeout_ms: int = 10000


class SpanManager:
    """
    Manages span lifecycle and export to OpenTelemetry collectors.

    Provides batching, retry logic, and async export capabilities.
    """

    def __init__(
        self,
        service_name: str,
        config: Optional[SpanExporterConfig] = None,
    ):
        self.service_name = service_name
        self.config = config or SpanExporterConfig()
        self._pending_spans: list[dict] = []
        self._export_task: Optional[asyncio.Task] = None
        self._running = False
        self._callbacks: list[Callable[[dict], None]] = []

    async def start(self) -> None:
        """Start the span manager background export task."""
        if self._running:
            return

        self._running = True
        self._export_task = asyncio.create_task(self._export_loop())
        logger.info(f"SpanManager started for {self.service_name}")

    async def stop(self) -> None:
        """Stop the span manager and flush pending spans."""
        self._running = False

        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass

        # Final flush
        if self._pending_spans:
            await self._export_batch(self._pending_spans)
            self._pending_spans = []

        logger.info(f"SpanManager stopped for {self.service_name}")

    def record_span(
        self,
        span: SpanContext,
        trace_id: str,
        parent_span_id: Optional[str] = None,
    ) -> None:
        """Record a completed span for export."""
        span_data = self._format_span(span, trace_id, parent_span_id)
        self._pending_spans.append(span_data)

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(span_data)
            except Exception as e:
                logger.warning(f"Span callback error: {e}")

        # Check if we should flush immediately
        if len(self._pending_spans) >= self.config.batch_size:
            asyncio.create_task(self._flush())

    def record_from_context(self, ctx: TracingContext) -> None:
        """Record all spans from a tracing context."""
        for span in ctx.spans:
            self.record_span(span, ctx.trace_id, ctx.parent_span_id)

    def add_callback(self, callback: Callable[[dict], None]) -> None:
        """Add a callback to be notified on span recording."""
        self._callbacks.append(callback)

    def _format_span(
        self,
        span: SpanContext,
        trace_id: str,
        parent_span_id: Optional[str] = None,
    ) -> dict:
        """Format span for OTLP export."""
        return {
            "traceId": trace_id,
            "spanId": span.span_id,
            "parentSpanId": parent_span_id,
            "name": span.name,
            "kind": "SPAN_KIND_INTERNAL",
            "startTimeUnixNano": int(span.start_time.timestamp() * 1e9),
            "endTimeUnixNano": int(span.end_time.timestamp() * 1e9) if span.end_time else None,
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in span.attributes.items()
            ],
            "events": [
                {
                    "name": e["name"],
                    "timeUnixNano": int(datetime.fromisoformat(e["timestamp"]).timestamp() * 1e9),
                    "attributes": [
                        {"key": k, "value": {"stringValue": str(v)}}
                        for k, v in e.get("attributes", {}).items()
                    ],
                }
                for e in span.events
            ],
            "status": {
                "code": self._status_code(span.status),
                "message": span.status_message,
            },
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": self.service_name}},
                ],
            },
        }

    def _status_code(self, status: str) -> int:
        """Convert status string to OTLP code."""
        return {
            "UNSET": 0,
            "OK": 1,
            "ERROR": 2,
        }.get(status, 0)

    async def _export_loop(self) -> None:
        """Background loop for periodic span export."""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval_ms / 1000)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Export loop error: {e}")

    async def _flush(self) -> None:
        """Flush pending spans."""
        if not self._pending_spans:
            return

        batch = self._pending_spans
        self._pending_spans = []

        await self._export_batch(batch)

    async def _export_batch(self, spans: list[dict]) -> bool:
        """Export a batch of spans to the collector."""
        if not spans:
            return True

        payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}},
                        ],
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "datamesh-ai-core"},
                            "spans": spans,
                        },
                    ],
                },
            ],
        }

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        **self.config.headers,
                    },
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_ms / 1000),
                ) as resp:
                    if resp.status >= 400:
                        logger.warning(f"Span export failed: {resp.status}")
                        return False

            logger.debug(f"Exported {len(spans)} spans")
            return True

        except ImportError:
            # aiohttp not available, log locally
            logger.debug(f"Would export {len(spans)} spans (aiohttp not available)")
            return True

        except Exception as e:
            logger.error(f"Span export error: {e}")
            return False


def trace_function(name: Optional[str] = None):
    """
    Decorator to automatically trace a function.

    Usage:
        @trace_function("my_operation")
        async def my_function():
            ...
    """
    def decorator(func: Callable):
        span_name = name or func.__name__

        async def async_wrapper(*args, **kwargs):
            ctx = get_current_trace()
            if not ctx:
                ctx = TracingContext.new()

            span = ctx.start_span(span_name)
            try:
                result = await func(*args, **kwargs)
                ctx.end_span("OK")
                return result
            except Exception as e:
                ctx.end_span("ERROR", str(e))
                raise

        def sync_wrapper(*args, **kwargs):
            ctx = get_current_trace()
            if not ctx:
                ctx = TracingContext.new()

            span = ctx.start_span(span_name)
            try:
                result = func(*args, **kwargs)
                ctx.end_span("OK")
                return result
            except Exception as e:
                ctx.end_span("ERROR", str(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
