"""
Base Agent Framework for DataMesh.AI

Provides shared abstractions to reduce code duplication across agents:
- HTTP handler with consistent request/response patterns
- Configuration management
- Error handling with retry logic
- Health check endpoints
- Metrics and tracing support
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("datamesh.agent", "0.1.0")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BaseAgentConfig:
    """Base configuration for all agents."""
    agent_id: str
    agent_type: str
    host: str = "0.0.0.0"
    port: int = 8080
    aws_region: str = "eu-west-1"

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0

    # Health check
    health_check_timeout_seconds: float = 5.0


# =============================================================================
# Response Builders
# =============================================================================

def build_success_response(
    request_id: str,
    agent_id: str,
    capability: str,
    data: dict[str, Any],
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Build a standardized success response."""
    response = {
        "status": "SUCCESS",
        "requestId": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "agent": agent_id,
        "capability": capability,
        "data": data,
    }
    if trace_id:
        response["traceId"] = trace_id
    return response


def build_error_response(
    request_id: str,
    agent_id: str,
    error: str,
    error_code: str | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Build a standardized error response."""
    response = {
        "status": "ERROR",
        "requestId": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "agent": agent_id,
        "error": error,
    }
    if error_code:
        response["errorCode"] = error_code
    if trace_id:
        response["traceId"] = trace_id
    return response


# =============================================================================
# Retry Decorator
# =============================================================================

def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed: {e}")

            raise last_exception
        return wrapper
    return decorator


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all DataMesh.AI agents.

    Subclasses must implement:
    - handle_capability(): Route requests to handlers
    - get_capabilities(): Return list of supported capabilities
    """

    def __init__(self, config: BaseAgentConfig) -> None:
        self.config = config
        self._metrics: dict[str, list[float]] = {}
        logger.info(f"Initialized {config.agent_type}: {config.agent_id}")

    @abstractmethod
    def handle_capability(
        self,
        capability: str,
        payload: dict[str, Any],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Handle a capability request.

        Args:
            capability: The capability being invoked (e.g., "sql.execute")
            payload: The request payload
            request: The full request object (for context, trace info, etc.)

        Returns:
            Result data to include in the response
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> list[str]:
        """Return list of capabilities this agent supports."""
        pass

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Main request handler with tracing and error handling.

        This method should not be overridden. Override handle_capability instead.
        """
        capability = request.get("capability", "")
        payload = request.get("payload", {})
        request_id = request.get("requestId", str(uuid.uuid4()))
        trace_info = request.get("trace", {})
        trace_id = trace_info.get("traceId")

        # Create span for tracing
        with tracer.start_as_current_span(
            f"{self.config.agent_type}.{capability}",
            attributes={
                "agent.id": self.config.agent_id,
                "agent.type": self.config.agent_type,
                "request.id": request_id,
                "capability": capability,
            },
        ) as span:
            start_time = time.time()

            try:
                if capability not in self.get_capabilities():
                    span.set_attribute("error", True)
                    return build_error_response(
                        request_id,
                        self.config.agent_id,
                        f"Unknown capability: {capability}",
                        error_code="UNKNOWN_CAPABILITY",
                        trace_id=trace_id,
                    )

                result = self.handle_capability(capability, payload, request)

                # Record metrics
                elapsed_ms = (time.time() - start_time) * 1000
                self._record_metric(f"{capability}.latency_ms", elapsed_ms)
                span.set_attribute("latency_ms", elapsed_ms)

                return build_success_response(
                    request_id,
                    self.config.agent_id,
                    capability,
                    result,
                    trace_id=trace_id,
                )

            except ValueError as e:
                # Validation errors - client's fault
                span.set_attribute("error", True)
                span.set_attribute("error.type", "validation")
                logger.warning(f"[{request_id}] Validation error: {e}")
                return build_error_response(
                    request_id,
                    self.config.agent_id,
                    str(e),
                    error_code="VALIDATION_ERROR",
                    trace_id=trace_id,
                )

            except Exception as e:
                # Internal errors
                span.set_attribute("error", True)
                span.set_attribute("error.type", "internal")
                logger.error(f"[{request_id}] Internal error: {e}", exc_info=True)
                return build_error_response(
                    request_id,
                    self.config.agent_id,
                    str(e),
                    error_code="INTERNAL_ERROR",
                    trace_id=trace_id,
                )

    def get_health(self) -> dict[str, Any]:
        """Return health status and metrics."""
        return {
            "status": "healthy",
            "agent": self.config.agent_id,
            "type": self.config.agent_type,
            "capabilities": self.get_capabilities(),
            "metrics": self._get_metrics_summary(),
        }

    def _record_metric(self, name: str, value: float) -> None:
        """Record a metric value (keeps last 100 values per metric)."""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)
        # Keep only last 100 values
        if len(self._metrics[name]) > 100:
            self._metrics[name] = self._metrics[name][-100:]

    def _get_metrics_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for name, values in self._metrics.items():
            if values:
                sorted_vals = sorted(values)
                summary[name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": sorted_vals[0],
                    "max": sorted_vals[-1],
                    "p50": sorted_vals[len(sorted_vals) // 2],
                    "p95": sorted_vals[int(len(sorted_vals) * 0.95)] if len(sorted_vals) >= 20 else sorted_vals[-1],
                }
        return summary


# =============================================================================
# HTTP Handler
# =============================================================================

class AgentHTTPHandler(BaseHTTPRequestHandler):
    """
    Shared HTTP handler for all agents.

    Set the 'agent' class attribute before creating the server:
        AgentHTTPHandler.agent = my_agent
    """

    agent: BaseAgent = None  # Must be set before server creation

    def do_POST(self):
        """Handle POST requests (capability invocations)."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            request = json.loads(body)
            response = self.agent.handle_request(request)
            self._send_json(200, response)
        except json.JSONDecodeError:
            self._send_json(400, {
                "status": "ERROR",
                "error": "Invalid JSON",
                "errorCode": "INVALID_JSON",
            })
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            self._send_json(500, {
                "status": "ERROR",
                "error": str(e),
                "errorCode": "INTERNAL_ERROR",
            })

    def do_GET(self):
        """Handle GET requests (health checks, info)."""
        if self.path == "/health":
            self._send_json(200, self.agent.get_health())
        elif self.path == "/capabilities":
            self._send_json(200, {
                "agent": self.agent.config.agent_id,
                "capabilities": self.agent.get_capabilities(),
            })
        elif self.path == "/metrics":
            self._send_json(200, {
                "agent": self.agent.config.agent_id,
                "metrics": self.agent._get_metrics_summary(),
            })
        else:
            self._send_json(404, {"error": "Not found"})

    def _send_json(self, status: int, data: dict):
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        """Override to use Python logging instead of stderr."""
        logger.debug(f"{self.address_string()} - {format % args}")


def run_agent_server(agent: BaseAgent, config: BaseAgentConfig) -> None:
    """
    Start the HTTP server for an agent.

    Args:
        agent: The agent instance
        config: Agent configuration
    """
    AgentHTTPHandler.agent = agent
    server = HTTPServer((config.host, config.port), AgentHTTPHandler)

    logger.info(f"Starting {config.agent_type} on {config.host}:{config.port}")
    logger.info(f"Capabilities: {agent.get_capabilities()}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


# =============================================================================
# Caching Utilities
# =============================================================================

class SimpleCache:
    """
    Simple in-memory cache with TTL support.

    For production, replace with Redis.
    """

    def __init__(self, default_ttl_seconds: int = 3600):
        self.default_ttl = default_ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expiry_time)

    def get(self, key: str) -> Any | None:
        """Get a value from cache, returns None if expired or missing."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Set a value in cache with TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        """Delete a key from cache."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        valid_count = sum(
            1 for _, expiry in self._cache.values()
            if time.time() < expiry
        )
        return {
            "total_keys": len(self._cache),
            "valid_keys": valid_count,
            "expired_keys": len(self._cache) - valid_count,
        }
