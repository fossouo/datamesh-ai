#!/usr/bin/env python3
"""
DATAMESH.AI A2A Orchestrator - AWS Edition
==========================================

Central orchestrator that routes requests between agents using
real Agent-to-Agent (A2A) protocol over HTTP.

Flow:
  User Request → Orchestrator → SQL Agent ──→ Catalog Agent (schema)
                                          ──→ Governance Agent (auth)
                                          ──→ AWS Athena (execute)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import httpx

# Import resilience patterns
try:
    from resilience import (
        ResilientClient,
        RetryConfig,
        CircuitBreakerConfig,
    )
    RESILIENCE_ENABLED = True
except ImportError:
    RESILIENCE_ENABLED = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("orchestrator-a2a")


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
class OrchestratorConfig:
    """Orchestrator configuration."""
    name: str = "datamesh-orchestrator-a2a"
    host: str = "0.0.0.0"
    port: int = 8080
    timeout_seconds: int = 120


class A2AProtocol:
    """A2A Protocol message builder."""

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


class A2AOrchestrator:
    """
    Agent-to-Agent Orchestrator.

    Routes requests to appropriate agents and coordinates
    multi-agent workflows with real HTTP communication.

    Features:
    - Retry with exponential backoff
    - Circuit breaker per agent
    - Request deduplication (idempotency)
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.http_client = httpx.Client(timeout=config.timeout_seconds)

        # Initialize resilient client if available
        if RESILIENCE_ENABLED:
            self.resilient_client = ResilientClient(
                retry_config=RetryConfig(
                    max_retries=3,
                    initial_delay_seconds=0.5,
                    max_delay_seconds=10.0,
                    backoff_multiplier=2.0,
                    jitter=True,
                ),
                circuit_config=CircuitBreakerConfig(
                    failure_threshold=5,
                    success_threshold=2,
                    timeout_seconds=30.0,
                ),
                idempotency_ttl=300,  # 5 minutes
            )
            logger.info("Resilience patterns enabled (retry, circuit breaker, idempotency)")
        else:
            self.resilient_client = None
            logger.warning("Resilience patterns not available, running without retry/circuit breaker")

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

        logger.info(f"Initialized {config.name}")
        logger.info(f"Registered agents: {list(self.agents.keys())}")
        logger.info(f"Capability map: {self.capability_map}")

    def call_agent(
        self,
        agent_name: str,
        capability: str,
        payload: dict[str, Any],
        trace_id: str,
        user_context: dict[str, Any] | None = None,
        depth: int = 0,
        use_idempotency: bool = False,
    ) -> dict[str, Any]:
        """
        Call an agent with A2A protocol.

        Args:
            agent_name: Target agent name
            capability: Capability to invoke
            payload: Request payload
            trace_id: Distributed trace ID
            user_context: User context for authorization
            depth: Current call depth
            use_idempotency: Enable request deduplication

        Returns:
            Agent response
        """
        agent = self.agents.get(agent_name)
        if not agent:
            return {"status": "ERROR", "error": f"Unknown agent: {agent_name}"}

        request = A2AProtocol.create_request(
            capability=capability,
            payload=payload,
            caller="orchestrator",
            trace_id=trace_id,
            user_context=user_context,
            depth=depth,
        )

        logger.info(f"[{trace_id}] → {agent_name}.{capability}")

        # Use resilient client if available
        if self.resilient_client:
            def make_call(req: dict) -> dict:
                response = self.http_client.post(agent.url, json=req)
                return response.json()

            result = self.resilient_client.call(
                endpoint=agent_name,
                request=request,
                call_fn=make_call,
                use_idempotency=use_idempotency,
            )

            # Log idempotency hits
            if result.get("_idempotency_hit"):
                logger.info(f"[{trace_id}] ← {agent_name} (IDEMPOTENCY_HIT)")
            else:
                status = result.get("status", "UNKNOWN")
                logger.info(f"[{trace_id}] ← {agent_name} ({status})")

            return result

        # Fallback to direct call without resilience
        try:
            response = self.http_client.post(agent.url, json=request)
            result = response.json()
            status = result.get("status", "UNKNOWN")
            logger.info(f"[{trace_id}] ← {agent_name} ({status})")
            return result
        except httpx.ConnectError:
            logger.error(f"[{trace_id}] Agent {agent_name} not available at {agent.url}")
            return {"status": "ERROR", "error": f"Agent {agent_name} not available"}
        except Exception as e:
            logger.error(f"[{trace_id}] Error calling {agent_name}: {e}")
            return {"status": "ERROR", "error": str(e)}

    def handle_nl_query(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Handle a natural language query with full A2A workflow.

        Flow:
        1. Call Catalog Agent to list available datasets
        2. Call SQL Agent to generate SQL
        3. Call Governance Agent to check authorization
        4. Call SQL Agent to execute (if authorized)

        Args:
            request: User request with 'question' and optional 'user' context

        Returns:
            Complete response with SQL, results, and audit trail
        """
        trace_id = str(uuid.uuid4())
        request_id = request.get("requestId", str(uuid.uuid4()))
        question = request.get("question", "")
        user_context = request.get("user", {"id": "anonymous", "roles": ["viewer"]})
        execute = request.get("execute", True)

        logger.info(f"[{trace_id}] New NL query: {question[:50]}...")

        a2a_trace = []  # Track all A2A calls

        # Step 1: Get available datasets from Catalog Agent
        logger.info(f"[{trace_id}] Step 1: Catalog discovery")
        catalog_response = self.call_agent(
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
        logger.info(f"[{trace_id}] Step 2: SQL generation")
        sql_gen_response = self.call_agent(
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
            return A2AProtocol.create_response(
                request_id, trace_id, "ERROR",
                error="Failed to generate SQL",
                agent="orchestrator",
            )

        generated_sql = sql_gen_response.get("data", {}).get("sql", "")

        # Step 3: Check authorization with Governance Agent
        logger.info(f"[{trace_id}] Step 3: Authorization check")
        gov_response = self.call_agent(
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
            "authorized": True,  # Default allow if governance agent unavailable
            "note": "Governance agent not available, defaulting to allow",
        }

        # Step 4: Execute SQL (if authorized and requested)
        execution_result = None
        if execute and authorization.get("authorized", True):
            logger.info(f"[{trace_id}] Step 4: SQL execution")
            exec_response = self.call_agent(
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

        return A2AProtocol.create_response(
            request_id, trace_id, "SUCCESS",
            data=response_data,
            agent="orchestrator",
        )

    def handle_direct_capability(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Route a direct capability request to the appropriate agent.

        Args:
            request: Request with 'capability' and 'payload'

        Returns:
            Agent response
        """
        trace_id = str(uuid.uuid4())
        capability = request.get("capability", "")
        payload = request.get("payload", {})
        user_context = request.get("user", {})

        agent_name = self.capability_map.get(capability)
        if not agent_name:
            return A2AProtocol.create_response(
                request.get("requestId", str(uuid.uuid4())),
                trace_id,
                "ERROR",
                error=f"No agent found for capability: {capability}",
            )

        return self.call_agent(agent_name, capability, payload, trace_id, user_context)

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Main request handler - routes to appropriate workflow.

        Args:
            request: Incoming request

        Returns:
            Response
        """
        # If it's a natural language query
        if "question" in request:
            return self.handle_nl_query(request)

        # If it's a direct capability call
        if "capability" in request:
            return self.handle_direct_capability(request)

        return {
            "status": "ERROR",
            "error": "Request must contain 'question' or 'capability'",
        }

    def check_agents_health(self) -> dict[str, Any]:
        """Check health of all registered agents."""
        health = {"orchestrator": "healthy", "agents": {}}

        for agent_name, agent in self.agents.items():
            try:
                response = self.http_client.get(f"{agent.url}/health", timeout=5)
                if response.status_code == 200:
                    health["agents"][agent_name] = "healthy"
                else:
                    health["agents"][agent_name] = "unhealthy"
            except Exception:
                health["agents"][agent_name] = "unavailable"

        # Add resilience stats if available
        if self.resilient_client:
            health["resilience"] = self.resilient_client.get_stats()

        return health


class OrchestratorHandler(BaseHTTPRequestHandler):
    """HTTP handler for orchestrator requests."""

    orchestrator: A2AOrchestrator = None

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            request = json.loads(body)
            response = self.orchestrator.handle_request(request)
            self._send_json(200, response)
        except json.JSONDecodeError:
            self._send_json(400, {"status": "ERROR", "error": "Invalid JSON"})
        except Exception as e:
            self._send_json(500, {"status": "ERROR", "error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            health = self.orchestrator.check_agents_health()
            self._send_json(200, health)
        elif self.path == "/agents":
            agents = {
                name: {
                    "url": agent.url,
                    "capabilities": agent.capabilities,
                }
                for name, agent in self.orchestrator.agents.items()
            }
            self._send_json(200, {"agents": agents})
        elif self.path == "/capabilities":
            self._send_json(200, {"capabilities": self.orchestrator.capability_map})
        else:
            self._send_json(404, {"error": "Not found"})

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        logger.debug(f"{self.address_string()} - {format % args}")


def main():
    """Run the A2A Orchestrator."""
    config = OrchestratorConfig(
        port=int(os.environ.get("ORCHESTRATOR_PORT", "8080")),
    )

    orchestrator = A2AOrchestrator(config)
    OrchestratorHandler.orchestrator = orchestrator

    server = HTTPServer((config.host, config.port), OrchestratorHandler)
    logger.info(f"Starting {config.name} on {config.host}:{config.port}")

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║            DATAMESH.AI - A2A Orchestrator                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Orchestrator: http://localhost:{config.port}                          ║
║                                                                  ║
║  Registered Agents:                                              ║
║    • sql-agent      → localhost:8081                             ║
║    • catalog-agent  → localhost:8082                             ║
║    • governance-agent → localhost:8083                           ║
║                                                                  ║
║  Endpoints:                                                      ║
║    POST /          - Submit query (NL or direct capability)      ║
║    GET  /health    - Check all agents health                     ║
║    GET  /agents    - List registered agents                      ║
║    GET  /capabilities - List capability routing                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down orchestrator...")
        server.shutdown()


if __name__ == "__main__":
    main()
