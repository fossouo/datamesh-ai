#!/usr/bin/env python3
"""
DATAMESH.AI Orchestrator - Reference Implementation
====================================================
Central orchestrator that routes A2A requests between agents.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("orchestrator")


class Orchestrator:
    """Central orchestrator for A2A request routing."""

    def __init__(self, config_path: str):
        """Initialize the orchestrator with configuration."""
        self.config = self._load_config(config_path)
        self.name = self.config["orchestrator"]["name"]
        self.agents = self.config.get("agents", {})
        self.routing_config = self.config.get("routing", {})

        # Build capability to agent mapping
        self.capability_map = self._build_capability_map()

        logger.info(f"Initialized {self.name}")
        logger.info(f"Registered agents: {list(self.agents.keys())}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load orchestrator configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _build_capability_map(self) -> Dict[str, str]:
        """Build mapping from capability to agent."""
        cap_map = {}
        for agent_name, agent_config in self.agents.items():
            for capability in agent_config.get("capabilities", []):
                cap_map[capability] = agent_name
        return cap_map

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming request and route to appropriate agent.

        Args:
            request: The incoming request payload

        Returns:
            Response from the target agent
        """
        request_id = request.get("requestId", str(uuid.uuid4()))
        trace_id = request.get("traceId", str(uuid.uuid4()))
        capability = request.get("capability")

        logger.info(f"[{trace_id}] Received request {request_id} for capability: {capability}")

        # Find the agent for this capability
        agent_name = self.capability_map.get(capability)
        if not agent_name:
            return self._error_response(
                request_id, trace_id,
                f"No agent found for capability: {capability}",
                "CAPABILITY_NOT_FOUND"
            )

        agent_config = self.agents[agent_name]
        logger.info(f"[{trace_id}] Routing to {agent_name} at {agent_config['endpoint']}")

        # Route the request
        try:
            response = await self._route_to_agent(
                agent_name, agent_config, request, trace_id
            )
            return response
        except Exception as e:
            logger.exception(f"[{trace_id}] Error routing request")
            return self._error_response(
                request_id, trace_id,
                f"Error routing to {agent_name}: {str(e)}",
                "ROUTING_ERROR"
            )

    async def _route_to_agent(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        request: Dict[str, Any],
        trace_id: str
    ) -> Dict[str, Any]:
        """
        Route a request to a specific agent.

        In production, this would use HTTP/gRPC with mTLS.
        For demo, we simulate the response.
        """
        request_id = request.get("requestId", str(uuid.uuid4()))
        capability = request.get("capability")
        payload = request.get("payload", {})

        logger.info(f"[{trace_id}] -> {agent_name} ({capability})")

        # Simulate agent processing time
        await asyncio.sleep(0.1)

        # Mock responses for demo
        if capability == "sql.generate":
            return self._mock_sql_generate_response(request_id, trace_id, payload)
        elif capability == "catalog.resolve":
            return self._mock_catalog_resolve_response(request_id, trace_id, payload)
        elif capability == "governance.authorize":
            return self._mock_governance_authorize_response(request_id, trace_id, payload)
        else:
            return {
                "requestId": request_id,
                "traceId": trace_id,
                "status": "SUCCESS",
                "result": {"message": f"Mock response for {capability}"},
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    def _mock_sql_generate_response(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock SQL generation response."""
        question = payload.get("question", "")

        sql = """
SELECT
    r.name AS region,
    tp.year,
    tp.month,
    SUM(rev.amount) AS monthly_revenue
FROM revenue rev
INNER JOIN regions r ON rev.region_id = r.id
INNER JOIN time_periods tp ON rev.period_id = tp.id
WHERE tp.year = 2025
  AND r.country IN ('FR', 'DE', 'ES')
GROUP BY r.name, tp.year, tp.month
ORDER BY tp.year, tp.month, r.name
""".strip()

        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "sql": sql,
                "explanation": f"Generated SQL for: {question}",
                "confidence": 0.92,
                "schema": {
                    "tables": ["revenue", "regions", "time_periods"]
                },
                "authorization": {
                    "authorized": True,
                    "policiesApplied": ["rgpd.yaml"]
                }
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _mock_catalog_resolve_response(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock catalog resolution response."""
        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "tables": ["revenue", "regions", "time_periods"],
                "columns": {
                    "revenue": ["id", "amount", "region_id", "period_id"],
                    "regions": ["id", "name", "country"],
                    "time_periods": ["id", "year", "month"]
                },
                "joins": [
                    {"from": "revenue.region_id", "to": "regions.id"},
                    {"from": "revenue.period_id", "to": "time_periods.id"}
                ]
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _mock_governance_authorize_response(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock governance authorization response."""
        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "authorized": True,
                "restrictions": {
                    "columns": {"revenue.amount": "AGGREGATE_ONLY"},
                    "rows": {"regions.country": ["FR", "DE", "ES"]}
                },
                "policiesApplied": ["rgpd.yaml"]
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _error_response(
        self,
        request_id: str,
        trace_id: str,
        message: str,
        code: str
    ) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "ERROR",
            "error": {
                "code": code,
                "message": message
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    async def run(self):
        """Run the orchestrator server."""
        host = self.config["server"]["host"]
        port = self.config["server"]["port"]

        logger.info(f"Starting {self.name} on {host}:{port}")
        logger.info(f"Capability map: {self.capability_map}")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down orchestrator...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DATAMESH.AI Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to orchestrator configuration YAML"
    )
    args = parser.parse_args()

    orchestrator = Orchestrator(args.config)
    asyncio.run(orchestrator.run())


if __name__ == "__main__":
    main()
