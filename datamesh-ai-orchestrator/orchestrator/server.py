"""
DATAMESH.AI Orchestrator - FastAPI HTTP Server
===============================================

This module provides the HTTP API server for the A2A orchestrator. It exposes:
- POST /a2a/request - Main A2A request handler
- GET /agents - List registered agents
- GET /agents/{name}/health - Agent health check
- GET /traces/{traceId} - Trace lookup
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from orchestrator.router import (
    A2ARouter,
    A2AResponse,
    ResponseStatus,
    RouterConfig,
    RetryConfig,
    CircuitBreakerConfig,
)
from orchestrator.supervisor import (
    AgentSupervisor,
    HealthStatus,
    SupervisorConfig,
)
from orchestrator.validator import AgentInfo

logger = logging.getLogger(__name__)


# Pydantic models for API
class TraceInfo(BaseModel):
    """Trace context information."""
    traceId: str | None = None
    parentSpanId: str | None = None
    spanId: str | None = None


class CallerInfo(BaseModel):
    """Caller agent information."""
    agent: str
    capability: str | None = None


class CalleeInfo(BaseModel):
    """Callee agent information."""
    agent: str
    capability: str


class OnBehalfOf(BaseModel):
    """Delegation context."""
    userId: str
    roles: list[str] = Field(default_factory=list)
    delegationChain: list[str] = Field(default_factory=list)


class CallConstraints(BaseModel):
    """Call constraints."""
    maxDepth: int = 5
    currentDepth: int = 0


class RequestContext(BaseModel):
    """Request context."""
    onBehalfOf: OnBehalfOf | None = None
    policiesApplied: list[str] = Field(default_factory=list)
    constraints: CallConstraints = Field(default_factory=CallConstraints)
    authContextRef: str | None = None


class PayloadRef(BaseModel):
    """Payload reference."""
    inputSchema: str | None = None
    outputSchema: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class A2ARequestModel(BaseModel):
    """A2A request model."""
    protocolVersion: str = "datamesh.ai/a2a/v1"
    requestId: str
    deadlineMs: int = 30000
    trace: TraceInfo = Field(default_factory=TraceInfo)
    caller: CallerInfo
    callee: CalleeInfo
    context: RequestContext = Field(default_factory=RequestContext)
    payloadRef: PayloadRef = Field(default_factory=PayloadRef)


class A2AResponseModel(BaseModel):
    """A2A response model."""
    requestId: str
    status: str
    trace: TraceInfo | None = None
    output: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    auditRef: str | None = None
    nextPollAfterMs: int | None = None


class AgentListItem(BaseModel):
    """Agent list item."""
    name: str
    displayName: str
    endpoint: str
    capabilities: list[str]
    enabled: bool
    health: str


class AgentHealthResponse(BaseModel):
    """Agent health response."""
    name: str
    status: str
    lastCheckTime: float | None = None
    consecutiveFailures: int = 0
    averageResponseTimeMs: float = 0.0
    lastError: str | None = None


class TraceSpan(BaseModel):
    """Trace span."""
    requestId: str
    spanId: str
    parentSpanId: str | None = None
    caller: str | None = None
    callee: str | None = None
    capability: str | None = None
    timestamp: float
    status: str | None = None
    durationMs: float | None = None


class TraceResponse(BaseModel):
    """Trace response."""
    traceId: str
    spans: list[TraceSpan]


class HealthSummary(BaseModel):
    """Health summary."""
    totalAgents: int
    healthy: int
    unhealthy: int
    degraded: int
    unknown: int


# Global instances (initialized at startup)
_router: A2ARouter | None = None
_supervisor: AgentSupervisor | None = None
_agents: dict[str, AgentInfo] = {}
_config: dict[str, Any] = {}


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load orchestrator configuration from YAML file.

    Args:
        config_path: Path to config file (defaults to ./config.yaml)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Try multiple locations
        possible_paths = [
            Path("config.yaml"),
            Path(__file__).parent.parent / "config.yaml",
            Path(os.environ.get("ORCHESTRATOR_CONFIG", "config.yaml"))
        ]
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path is None or not Path(config_path).exists():
        logger.warning("No config file found, using defaults")
        return get_default_config()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")
    return config


def get_default_config() -> dict[str, Any]:
    """Get default configuration."""
    return {
        "agents": [],
        "policies": {"globalPolicies": [], "classificationPolicies": {}},
        "tracing": {"enabled": False},
        "timeouts": {
            "defaultDeadlineMs": 30000,
            "maxDeadlineMs": 300000,
            "minDeadlineMs": 1000,
            "healthCheckTimeoutMs": 5000,
            "connectionTimeoutMs": 10000,
            "readTimeoutMs": 60000
        },
        "routing": {
            "maxDepth": 5,
            "requireTraceParent": True,
            "allowDelegation": True,
            "circuitBreaker": {
                "enabled": True,
                "failureThreshold": 5,
                "successThreshold": 2,
                "timeoutMs": 60000
            },
            "retry": {
                "enabled": True,
                "maxAttempts": 3,
                "backoffMs": 500,
                "backoffMultiplier": 2.0,
                "maxBackoffMs": 5000,
                "retryableErrors": ["TIMEOUT", "SERVICE_UNAVAILABLE", "CONNECTION_ERROR"]
            }
        },
        "supervision": {
            "healthCheckIntervalMs": 30000,
            "unhealthyThreshold": 3,
            "healthyThreshold": 2,
            "autoRestart": False,
            "alertOnHealthChange": True
        },
        "logging": {
            "redactFields": ["user.token", "secrets.*", "credentials.*", "password", "apiKey"]
        },
        "server": {
            "cors": {
                "enabled": True,
                "allowOrigins": ["*"],
                "allowMethods": ["GET", "POST", "PUT", "DELETE"],
                "allowHeaders": ["*"]
            }
        }
    }


def parse_agents(config: dict[str, Any]) -> dict[str, AgentInfo]:
    """Parse agent configurations from config."""
    agents = {}
    for agent_config in config.get("agents", []):
        agent = AgentInfo(
            name=agent_config["name"],
            display_name=agent_config.get("displayName", agent_config["name"]),
            endpoint=agent_config["endpoint"],
            health_endpoint=agent_config.get("healthEndpoint", "/health"),
            capabilities=agent_config.get("capabilities", []),
            can_call=agent_config.get("canCall", []),
            contract_ref=agent_config.get("contractRef", ""),
            enabled=agent_config.get("enabled", True)
        )
        agents[agent.name] = agent
    return agents


def create_router_config(config: dict[str, Any]) -> RouterConfig:
    """Create router configuration from config."""
    timeouts = config.get("timeouts", {})
    routing = config.get("routing", {})
    logging_config = config.get("logging", {})

    retry_config = routing.get("retry", {})
    cb_config = routing.get("circuitBreaker", {})

    return RouterConfig(
        default_deadline_ms=timeouts.get("defaultDeadlineMs", 30000),
        max_deadline_ms=timeouts.get("maxDeadlineMs", 300000),
        min_deadline_ms=timeouts.get("minDeadlineMs", 1000),
        connection_timeout_ms=timeouts.get("connectionTimeoutMs", 10000),
        read_timeout_ms=timeouts.get("readTimeoutMs", 60000),
        max_depth=routing.get("maxDepth", 5),
        require_trace_parent=routing.get("requireTraceParent", True),
        allow_delegation=routing.get("allowDelegation", True),
        retry=RetryConfig(
            enabled=retry_config.get("enabled", True),
            max_attempts=retry_config.get("maxAttempts", 3),
            backoff_ms=retry_config.get("backoffMs", 500),
            backoff_multiplier=retry_config.get("backoffMultiplier", 2.0),
            max_backoff_ms=retry_config.get("maxBackoffMs", 5000),
            retryable_errors=retry_config.get("retryableErrors", [
                "TIMEOUT", "SERVICE_UNAVAILABLE", "CONNECTION_ERROR"
            ])
        ),
        circuit_breaker=CircuitBreakerConfig(
            enabled=cb_config.get("enabled", True),
            failure_threshold=cb_config.get("failureThreshold", 5),
            success_threshold=cb_config.get("successThreshold", 2),
            timeout_ms=cb_config.get("timeoutMs", 60000)
        ),
        redact_fields=logging_config.get("redactFields", [
            "user.token", "secrets.*", "credentials.*", "password", "apiKey"
        ])
    )


def create_supervisor_config(config: dict[str, Any]) -> SupervisorConfig:
    """Create supervisor configuration from config."""
    supervision = config.get("supervision", {})
    timeouts = config.get("timeouts", {})

    return SupervisorConfig(
        health_check_interval_ms=supervision.get("healthCheckIntervalMs", 30000),
        health_check_timeout_ms=timeouts.get("healthCheckTimeoutMs", 5000),
        unhealthy_threshold=supervision.get("unhealthyThreshold", 3),
        healthy_threshold=supervision.get("healthyThreshold", 2),
        auto_restart=supervision.get("autoRestart", False),
        alert_on_health_change=supervision.get("alertOnHealthChange", True),
        alert_webhook=supervision.get("alertWebhook")
    )


def audit_callback(audit_data: dict[str, Any]) -> None:
    """Callback for audit logging."""
    logger.info(f"AUDIT: {audit_data}")


def health_change_callback(
    agent_name: str,
    old_status: HealthStatus,
    new_status: HealthStatus
) -> None:
    """Callback for health state changes."""
    logger.warning(
        f"Health change: agent={agent_name} old={old_status.value} new={new_status.value}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _router, _supervisor, _agents, _config

    # Load configuration
    config_path = os.environ.get("ORCHESTRATOR_CONFIG")
    _config = load_config(config_path)

    # Parse agents
    _agents = parse_agents(_config)
    logger.info(f"Loaded {len(_agents)} agents")

    # Create router
    router_config = create_router_config(_config)
    policies = _config.get("policies", {})

    _router = A2ARouter(
        agents=_agents,
        config=router_config,
        global_policies=policies.get("globalPolicies", []),
        classification_policies=policies.get("classificationPolicies", {}),
        audit_callback=audit_callback
    )

    # Create supervisor
    supervisor_config = create_supervisor_config(_config)
    _supervisor = AgentSupervisor(
        agents=_agents,
        config=supervisor_config,
        on_health_change=health_change_callback
    )

    # Start supervisor
    await _supervisor.start()
    logger.info("Orchestrator started")

    yield

    # Cleanup
    await _supervisor.stop()
    await _router.close()
    logger.info("Orchestrator stopped")


# Create FastAPI app
app = FastAPI(
    title="DATAMESH.AI Orchestrator",
    description="A2A request routing, validation, and agent supervision",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    """Readiness check endpoint."""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Orchestrator not ready")

    summary = _supervisor.get_summary()
    healthy_count = summary.get("healthy", 0)
    total_count = summary.get("total_agents", 0)

    return {
        "status": "ready",
        "healthy_agents": healthy_count,
        "total_agents": total_count
    }


@app.post("/a2a/request", response_model=A2AResponseModel)
async def handle_a2a_request(
    request: A2ARequestModel,
    http_request: Request
) -> JSONResponse:
    """
    Main A2A request handler.

    Validates the request against contracts and policies, then routes
    it to the appropriate agent.
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    # Convert Pydantic model to dict
    raw_request = request.model_dump(by_alias=True)

    # Extract data classifications from headers if present
    classifications_header = http_request.headers.get("X-Data-Classifications")
    data_classifications = None
    if classifications_header:
        data_classifications = [c.strip() for c in classifications_header.split(",")]

    # Route the request
    response = await _router.route(raw_request, data_classifications)

    # Convert response to dict
    response_data = response.to_dict()

    # Return appropriate status code
    status_code = 200
    if response.status == ResponseStatus.ERROR:
        error_code = response.error.get("code", "") if response.error else ""
        if error_code in ("UNKNOWN_AGENT", "UNKNOWN_CAPABILITY"):
            status_code = 404
        elif error_code in ("FORBIDDEN_CAPABILITY", "POLICY_VIOLATION"):
            status_code = 403
        elif error_code == "INVALID_REQUEST_FORMAT":
            status_code = 400
        else:
            status_code = 500

    return JSONResponse(content=response_data, status_code=status_code)


@app.get("/agents", response_model=list[AgentListItem])
async def list_agents() -> list[dict[str, Any]]:
    """List all registered agents with their status."""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Supervisor not initialized")

    result = []
    for name, agent in _agents.items():
        status = _supervisor.get_agent_status(name)
        result.append({
            "name": agent.name,
            "displayName": agent.display_name,
            "endpoint": agent.endpoint,
            "capabilities": agent.capabilities,
            "enabled": agent.enabled,
            "health": status.value
        })

    return result


@app.get("/agents/{name}/health", response_model=AgentHealthResponse)
async def get_agent_health(name: str) -> dict[str, Any]:
    """Get health status for a specific agent."""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Supervisor not initialized")

    if name not in _agents:
        raise HTTPException(status_code=404, detail=f"Agent not found: {name}")

    state = _supervisor.get_agent_health_state(name)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Health state not found: {name}")

    return {
        "name": name,
        "status": state.current_status.value,
        "lastCheckTime": state.last_check_time if state.last_check_time > 0 else None,
        "consecutiveFailures": state.consecutive_failures,
        "averageResponseTimeMs": round(state.average_response_time_ms, 2),
        "lastError": state.last_error
    }


@app.post("/agents/{name}/health/check")
async def trigger_health_check(name: str) -> dict[str, Any]:
    """Trigger an immediate health check for an agent."""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Supervisor not initialized")

    if name not in _agents:
        raise HTTPException(status_code=404, detail=f"Agent not found: {name}")

    result = await _supervisor.check_agent(name)

    return {
        "name": result.agent_name,
        "status": result.status.value,
        "responseTimeMs": result.response_time_ms,
        "timestamp": result.timestamp,
        "error": result.error
    }


@app.get("/agents/summary", response_model=HealthSummary)
async def get_health_summary() -> dict[str, Any]:
    """Get health summary for all agents."""
    if _supervisor is None:
        raise HTTPException(status_code=503, detail="Supervisor not initialized")

    summary = _supervisor.get_summary()

    return {
        "totalAgents": summary["total_agents"],
        "healthy": summary["healthy"],
        "unhealthy": summary["unhealthy"],
        "degraded": summary["degraded"],
        "unknown": summary["unknown"]
    }


@app.get("/traces/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str) -> dict[str, Any]:
    """
    Get trace data by trace ID.

    Returns all spans associated with the given trace.
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    trace_data = _router.get_trace(trace_id)
    if trace_data is None:
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id}")

    spans = []
    for span_data in trace_data:
        spans.append({
            "requestId": span_data.get("requestId", ""),
            "spanId": span_data.get("spanId", ""),
            "parentSpanId": span_data.get("parentSpanId"),
            "caller": span_data.get("caller"),
            "callee": span_data.get("callee"),
            "capability": span_data.get("capability"),
            "timestamp": span_data.get("timestamp", 0),
            "status": span_data.get("status"),
            "durationMs": span_data.get("duration_ms")
        })

    return {
        "traceId": trace_id,
        "spans": spans
    }


@app.get("/config")
async def get_config() -> dict[str, Any]:
    """Get current orchestrator configuration (sanitized)."""
    # Return sanitized config (no secrets)
    return {
        "agents": len(_agents),
        "routing": {
            "maxDepth": _config.get("routing", {}).get("maxDepth", 5),
            "requireTraceParent": _config.get("routing", {}).get("requireTraceParent", True),
            "circuitBreakerEnabled": _config.get("routing", {}).get("circuitBreaker", {}).get("enabled", True),
            "retryEnabled": _config.get("routing", {}).get("retry", {}).get("enabled", True)
        },
        "timeouts": _config.get("timeouts", {}),
        "tracing": {
            "enabled": _config.get("tracing", {}).get("enabled", False)
        }
    }


def main() -> None:
    """Run the orchestrator server."""
    import uvicorn

    # Get server config
    server_config = _config.get("server", {}) if _config else {}
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)
    workers = server_config.get("workers", 1)
    debug = server_config.get("debug", False)

    uvicorn.run(
        "orchestrator.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()
