#!/usr/bin/env python3
"""
DATAMESH.AI Async A2A Orchestrator
==================================

High-performance async orchestrator using aiohttp with connection pooling.
Designed for production workloads with concurrent request handling.

Features:
- Async HTTP with connection pooling
- Concurrent agent calls where possible
- Semaphore-based concurrency limits
- Graceful shutdown handling
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import aiohttp
from aiohttp import web

# Import resilience patterns
try:
    from resilience import (
        ResilientClient,
        RetryConfig,
        CircuitBreakerConfig,
        CircuitBreaker,
        CircuitOpenError,
    )
    RESILIENCE_ENABLED = True
except ImportError:
    RESILIENCE_ENABLED = False
    CircuitBreaker = None
    CircuitOpenError = Exception

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator-async")


@dataclass
class AgentEndpoint:
    """Configuration for an agent endpoint."""
    name: str
    host: str
    port: int
    capabilities: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class AsyncOrchestratorConfig:
    """Async orchestrator configuration."""
    name: str = "datamesh-orchestrator-async"
    host: str = "0.0.0.0"
    port: int = 8080
    timeout_seconds: int = 120
    max_concurrent_requests: int = 100
    connection_pool_size: int = 50


class AsyncCircuitBreaker:
    """Async-compatible circuit breaker wrapper."""

    def __init__(self, name: str, failure_threshold: int = 5, timeout_seconds: float = 30.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = "closed"
        self._lock = asyncio.Lock()

    async def check(self) -> bool:
        """Check if circuit allows request."""
        async with self._lock:
            if self._state == "open":
                if self._last_failure_time:
                    import time
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.timeout_seconds:
                        self._state = "half_open"
                        logger.info(f"Circuit {self.name}: OPEN → HALF_OPEN")
                        return True
                return False
            return True

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._failure_count = 0
            if self._state == "half_open":
                self._state = "closed"
                logger.info(f"Circuit {self.name}: HALF_OPEN → CLOSED")

    async def record_failure(self) -> None:
        """Record a failed call."""
        import time
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == "half_open":
                self._state = "open"
                logger.warning(f"Circuit {self.name}: HALF_OPEN → OPEN")
            elif self._failure_count >= self.failure_threshold:
                self._state = "open"
                self._failure_count = 0
                logger.warning(f"Circuit {self.name}: CLOSED → OPEN")

    def stats(self) -> dict[str, Any]:
        """Return circuit breaker stats."""
        return {
            "name": self.name,
            "state": self._state,
            "failure_count": self._failure_count,
        }


class A2AProtocolAsync:
    """A2A Protocol message builder for async operations."""

    @staticmethod
    def create_request(
        capability: str,
        payload: dict[str, Any],
        caller: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        user_context: dict[str, Any] | None = None,
        depth: int = 0,
    ) -> dict[str, Any]:
        """Create an A2A protocol request."""
        request_id = str(uuid.uuid4())
        trace_id = trace_id or str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:16]

        return {
            "protocolVersion": "datamesh.ai/a2a/v1",
            "requestId": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace": {
                "traceId": trace_id,
                "parentSpanId": parent_span_id,
                "spanId": span_id,
            },
            "caller": {
                "agent": caller,
                "capability": capability,
            },
            "context": {
                "user": user_context or {},
                "depth": depth,
                "maxDepth": 5,
            },
            "capability": capability,
            "payload": payload,
        }

    @staticmethod
    def create_response(
        request_id: str,
        trace_id: str,
        status: str,
        data: dict[str, Any] | None = None,
        error: str | None = None,
        agent: str = "orchestrator",
    ) -> dict[str, Any]:
        """Create an A2A protocol response."""
        response = {
            "protocolVersion": "datamesh.ai/a2a/v1",
            "requestId": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace": {"traceId": trace_id},
            "status": status,
            "agent": agent,
        }
        if data:
            response["data"] = data
        if error:
            response["error"] = error
        return response


class AsyncA2AOrchestrator:
    """
    Async Agent-to-Agent Orchestrator.

    High-performance orchestrator with:
    - Connection pooling via aiohttp
    - Concurrent agent calls
    - Per-agent circuit breakers
    - Semaphore-based concurrency control
    """

    def __init__(self, config: AsyncOrchestratorConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        # Circuit breakers per agent
        self._circuits: dict[str, AsyncCircuitBreaker] = {}

        # Register agents
        self.agents: dict[str, AgentEndpoint] = {
            "sql-agent": AgentEndpoint(
                name="sql-agent",
                host="localhost",
                port=int(os.environ.get("SQL_AGENT_PORT", "8081")),
                capabilities=["sql.generate", "sql.validate", "sql.execute", "sql.optimize"],
            ),
            "catalog-agent": AgentEndpoint(
                name="catalog-agent",
                host="localhost",
                port=int(os.environ.get("CATALOG_AGENT_PORT", "8082")),
                capabilities=["catalog.resolve", "catalog.lineage", "catalog.search", "catalog.list"],
            ),
            "governance-agent": AgentEndpoint(
                name="governance-agent",
                host="localhost",
                port=int(os.environ.get("GOVERNANCE_AGENT_PORT", "8083")),
                capabilities=["governance.authorize", "governance.audit", "governance.classify"],
            ),
        }

        # Build capability → agent map
        self.capability_map: dict[str, str] = {}
        for agent_name, agent in self.agents.items():
            for cap in agent.capabilities:
                self.capability_map[cap] = agent_name
            # Create circuit breaker for each agent
            self._circuits[agent_name] = AsyncCircuitBreaker(agent_name)

        # Metrics
        self._request_count = 0
        self._error_count = 0

        logger.info(f"Initialized {config.name}")
        logger.info(f"Max concurrent requests: {config.max_concurrent_requests}")
        logger.info(f"Connection pool size: {config.connection_pool_size}")

    async def start(self) -> None:
        """Initialize the HTTP session with connection pooling."""
        connector = aiohttp.TCPConnector(
            limit=self.config.connection_pool_size,
            limit_per_host=20,
            keepalive_timeout=30,
        )
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        )
        logger.info("HTTP session started with connection pooling")

    async def stop(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            logger.info("HTTP session closed")

    async def call_agent(
        self,
        agent_name: str,
        capability: str,
        payload: dict[str, Any],
        trace_id: str,
        user_context: dict[str, Any] | None = None,
        depth: int = 0,
    ) -> dict[str, Any]:
        """
        Call an agent with A2A protocol (async).

        Uses semaphore for concurrency control and circuit breaker for resilience.
        """
        agent = self.agents.get(agent_name)
        if not agent:
            return {"status": "ERROR", "error": f"Unknown agent: {agent_name}"}

        # Check circuit breaker
        circuit = self._circuits.get(agent_name)
        if circuit and not await circuit.check():
            return {
                "status": "ERROR",
                "error": f"Circuit breaker open for {agent_name}",
                "errorCode": "CIRCUIT_OPEN",
            }

        request = A2AProtocolAsync.create_request(
            capability=capability,
            payload=payload,
            caller="orchestrator",
            trace_id=trace_id,
            user_context=user_context,
            depth=depth,
        )

        async with self._semaphore:
            logger.info(f"[{trace_id}] → {agent_name}.{capability}")

            try:
                async with self._session.post(agent.url, json=request) as response:
                    result = await response.json()
                    status = result.get("status", "UNKNOWN")
                    logger.info(f"[{trace_id}] ← {agent_name} ({status})")

                    if circuit:
                        if status == "ERROR":
                            await circuit.record_failure()
                        else:
                            await circuit.record_success()

                    return result

            except aiohttp.ClientConnectorError:
                logger.error(f"[{trace_id}] Agent {agent_name} not available at {agent.url}")
                if circuit:
                    await circuit.record_failure()
                return {"status": "ERROR", "error": f"Agent {agent_name} not available"}
            except asyncio.TimeoutError:
                logger.error(f"[{trace_id}] Timeout calling {agent_name}")
                if circuit:
                    await circuit.record_failure()
                return {"status": "ERROR", "error": f"Timeout calling {agent_name}"}
            except Exception as e:
                logger.error(f"[{trace_id}] Error calling {agent_name}: {e}")
                if circuit:
                    await circuit.record_failure()
                return {"status": "ERROR", "error": str(e)}

    async def handle_nl_query(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Handle a natural language query with full A2A workflow.

        Uses concurrent calls where possible for better performance.
        """
        self._request_count += 1
        trace_id = str(uuid.uuid4())
        request_id = request.get("requestId", str(uuid.uuid4()))
        question = request.get("question", "")
        user_context = request.get("user", {"id": "anonymous", "roles": ["viewer"]})
        execute = request.get("execute", True)

        logger.info(f"[{trace_id}] New NL query: {question[:50]}...")

        a2a_trace = []

        # Step 1: Get available datasets from Catalog Agent
        catalog_response = await self.call_agent(
            "catalog-agent",
            "catalog.list",
            {"database": os.environ.get("ATHENA_DATABASE", "talki_metrics_prod")},
            trace_id,
            user_context,
            depth=1,
        )
        a2a_trace.append({
            "step": 1,
            "agent": "catalog-agent",
            "capability": "catalog.list",
            "status": catalog_response.get("status"),
        })

        available_datasets = []
        if catalog_response.get("status") == "SUCCESS":
            available_datasets = catalog_response.get("data", {}).get("datasets", [])

        # Step 2: Generate SQL using SQL Agent
        sql_gen_response = await self.call_agent(
            "sql-agent",
            "sql.generate",
            {
                "question": question,
                "available_datasets": available_datasets,
            },
            trace_id,
            user_context,
            depth=1,
        )
        a2a_trace.append({
            "step": 2,
            "agent": "sql-agent",
            "capability": "sql.generate",
            "status": sql_gen_response.get("status"),
        })

        if sql_gen_response.get("status") != "SUCCESS":
            self._error_count += 1
            return A2AProtocolAsync.create_response(
                request_id, trace_id, "ERROR",
                error="Failed to generate SQL",
                agent="orchestrator",
            )

        generated_sql = sql_gen_response.get("data", {}).get("sql", "")

        # Step 3: Check authorization with Governance Agent
        gov_response = await self.call_agent(
            "governance-agent",
            "governance.authorize",
            {
                "user": user_context,
                "sql": generated_sql,
                "datasets": available_datasets,
            },
            trace_id,
            user_context,
            depth=1,
        )
        a2a_trace.append({
            "step": 3,
            "agent": "governance-agent",
            "capability": "governance.authorize",
            "status": gov_response.get("status"),
        })

        authorization = gov_response.get("data", {}) if gov_response.get("status") == "SUCCESS" else {
            "authorized": True,
            "note": "Governance agent not available, defaulting to allow",
        }

        # Step 4: Execute SQL (if authorized and requested)
        execution_result = None
        if execute and authorization.get("authorized", True):
            exec_response = await self.call_agent(
                "sql-agent",
                "sql.execute",
                {"sql": generated_sql},
                trace_id,
                user_context,
                depth=1,
            )
            a2a_trace.append({
                "step": 4,
                "agent": "sql-agent",
                "capability": "sql.execute",
                "status": exec_response.get("status"),
            })

            if exec_response.get("status") == "SUCCESS":
                execution_result = exec_response.get("data", {})

        # Build final response
        response_data = {
            "question": question,
            "sql": generated_sql,
            "confidence": sql_gen_response.get("data", {}).get("confidence", 0),
            "authorization": authorization,
            "execution": execution_result,
            "a2a_trace": a2a_trace,
            "datasets_available": len(available_datasets),
        }

        return A2AProtocolAsync.create_response(
            request_id, trace_id, "SUCCESS",
            data=response_data,
            agent="orchestrator",
        )

    async def handle_direct_capability(self, request: dict[str, Any]) -> dict[str, Any]:
        """Route a direct capability request to the appropriate agent."""
        trace_id = str(uuid.uuid4())
        capability = request.get("capability", "")
        payload = request.get("payload", {})
        user_context = request.get("user", {})

        agent_name = self.capability_map.get(capability)
        if not agent_name:
            return A2AProtocolAsync.create_response(
                request.get("requestId", str(uuid.uuid4())),
                trace_id,
                "ERROR",
                error=f"No agent found for capability: {capability}",
            )

        return await self.call_agent(agent_name, capability, payload, trace_id, user_context)

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Main request handler - routes to appropriate workflow."""
        if "question" in request:
            return await self.handle_nl_query(request)

        if "capability" in request:
            return await self.handle_direct_capability(request)

        return {
            "status": "ERROR",
            "error": "Request must contain 'question' or 'capability'",
        }

    async def check_agents_health(self) -> dict[str, Any]:
        """Check health of all registered agents concurrently."""
        health = {"orchestrator": "healthy", "agents": {}}

        async def check_agent(name: str, agent: AgentEndpoint) -> tuple[str, str]:
            try:
                async with self._session.get(f"{agent.url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        return name, "healthy"
                    return name, "unhealthy"
            except Exception:
                return name, "unavailable"

        # Check all agents concurrently
        results = await asyncio.gather(*[
            check_agent(name, agent)
            for name, agent in self.agents.items()
        ])

        for name, status in results:
            health["agents"][name] = status

        # Add circuit breaker stats
        health["circuits"] = {
            name: circuit.stats()
            for name, circuit in self._circuits.items()
        }

        # Add metrics
        health["metrics"] = {
            "total_requests": self._request_count,
            "errors": self._error_count,
            "error_rate": f"{(self._error_count / self._request_count * 100) if self._request_count > 0 else 0:.2f}%",
        }

        return health


# =============================================================================
# Web Application
# =============================================================================

async def create_app(config: AsyncOrchestratorConfig) -> web.Application:
    """Create the aiohttp web application."""
    orchestrator = AsyncA2AOrchestrator(config)

    async def on_startup(app: web.Application) -> None:
        await orchestrator.start()
        app["orchestrator"] = orchestrator

    async def on_cleanup(app: web.Application) -> None:
        await orchestrator.stop()

    async def handle_post(request: web.Request) -> web.Response:
        try:
            data = await request.json()
            result = await orchestrator.handle_request(data)
            return web.json_response(result)
        except json.JSONDecodeError:
            return web.json_response(
                {"status": "ERROR", "error": "Invalid JSON"},
                status=400,
            )
        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return web.json_response(
                {"status": "ERROR", "error": str(e)},
                status=500,
            )

    async def handle_health(request: web.Request) -> web.Response:
        health = await orchestrator.check_agents_health()
        return web.json_response(health)

    async def handle_agents(request: web.Request) -> web.Response:
        agents = {
            name: {
                "url": agent.url,
                "capabilities": agent.capabilities,
            }
            for name, agent in orchestrator.agents.items()
        }
        return web.json_response({"agents": agents})

    async def handle_capabilities(request: web.Request) -> web.Response:
        return web.json_response({"capabilities": orchestrator.capability_map})

    async def handle_metrics(request: web.Request) -> web.Response:
        return web.json_response({
            "total_requests": orchestrator._request_count,
            "errors": orchestrator._error_count,
            "circuits": {
                name: circuit.stats()
                for name, circuit in orchestrator._circuits.items()
            },
        })

    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_post("/", handle_post)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/agents", handle_agents)
    app.router.add_get("/capabilities", handle_capabilities)
    app.router.add_get("/metrics", handle_metrics)

    return app


def main():
    """Run the Async A2A Orchestrator."""
    config = AsyncOrchestratorConfig(
        port=int(os.environ.get("ORCHESTRATOR_PORT", "8080")),
        max_concurrent_requests=int(os.environ.get("MAX_CONCURRENT", "100")),
        connection_pool_size=int(os.environ.get("POOL_SIZE", "50")),
    )

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║         DATAMESH.AI - Async A2A Orchestrator                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Orchestrator: http://localhost:{config.port}                          ║
║  Mode: ASYNC with connection pooling                             ║
║                                                                  ║
║  Registered Agents:                                              ║
║    • sql-agent       → localhost:8081                            ║
║    • catalog-agent   → localhost:8082                            ║
║    • governance-agent → localhost:8083                           ║
║                                                                  ║
║  Configuration:                                                  ║
║    Max concurrent requests: {config.max_concurrent_requests}                              ║
║    Connection pool size: {config.connection_pool_size}                                 ║
║                                                                  ║
║  Endpoints:                                                      ║
║    POST /           - Submit query (NL or direct capability)     ║
║    GET  /health     - Check all agents health                    ║
║    GET  /agents     - List registered agents                     ║
║    GET  /capabilities - List capability routing                  ║
║    GET  /metrics    - Get orchestrator metrics                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    app = asyncio.get_event_loop().run_until_complete(create_app(config))
    web.run_app(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
