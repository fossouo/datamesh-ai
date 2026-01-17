#!/usr/bin/env python3
"""
SQL Agent - DATAMESH.AI Reference Implementation
=================================================
Interprets natural language questions and generates SQL queries.
Demonstrates A2A protocol integration with Catalog and Governance agents.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("sql-agent")


class SQLAgent:
    """SQL Agent that generates queries from natural language."""

    def __init__(self, config_path: str):
        """Initialize the SQL Agent with configuration."""
        self.config = self._load_config(config_path)
        self.agent_id = self.config["agent"]["id"]
        self.agent_name = self.config["agent"]["name"]
        self.capabilities = self.config.get("capabilities", [])
        self.dependencies = self.config.get("dependencies", [])

        logger.info(f"Initialized {self.agent_name} (ID: {self.agent_id})")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming A2A request.

        Args:
            request: The A2A request payload

        Returns:
            A2A response payload
        """
        request_id = request.get("requestId", str(uuid.uuid4()))
        trace_id = request.get("traceId", str(uuid.uuid4()))
        capability = request.get("capability")
        payload = request.get("payload", {})

        logger.info(f"[{trace_id}] Handling request {request_id} for capability: {capability}")

        try:
            if capability == "sql.generate":
                return await self._generate_sql(request_id, trace_id, payload)
            elif capability == "sql.validate":
                return await self._validate_sql(request_id, trace_id, payload)
            elif capability == "sql.optimize":
                return await self._optimize_sql(request_id, trace_id, payload)
            else:
                return self._error_response(
                    request_id, trace_id,
                    f"Unknown capability: {capability}",
                    "CAPABILITY_NOT_FOUND"
                )
        except Exception as e:
            logger.exception(f"[{trace_id}] Error handling request")
            return self._error_response(request_id, trace_id, str(e), "INTERNAL_ERROR")

    async def _generate_sql(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate SQL from a natural language question."""
        question = payload.get("question")
        user = payload.get("user", {})
        policy_refs = payload.get("policyRefs", [])

        logger.info(f"[{trace_id}] Generating SQL for question: {question}")

        # Step 1: Resolve schema from Catalog Agent
        catalog_response = await self._call_agent(
            "catalog-agent",
            "catalog.resolve",
            {"question": question},
            trace_id
        )

        if catalog_response.get("status") != "SUCCESS":
            return self._error_response(
                request_id, trace_id,
                "Failed to resolve schema from catalog",
                "CATALOG_RESOLUTION_FAILED"
            )

        schema_info = catalog_response.get("result", {})

        # Step 2: Check authorization with Governance Agent
        governance_response = await self._call_agent(
            "governance-agent",
            "governance.authorize",
            {
                "user": user,
                "resource": schema_info.get("tables", []),
                "action": "SELECT",
                "policyRefs": policy_refs
            },
            trace_id
        )

        if governance_response.get("status") != "SUCCESS":
            return self._error_response(
                request_id, trace_id,
                "Authorization denied by governance",
                "AUTHORIZATION_DENIED"
            )

        authorization = governance_response.get("result", {})

        # Step 3: Generate SQL (mock implementation)
        generated_sql = self._mock_generate_sql(question, schema_info, authorization)

        # Step 4: Optionally get lineage
        lineage_response = await self._call_agent(
            "catalog-agent",
            "catalog.lineage",
            {"tables": schema_info.get("tables", [])},
            trace_id
        )

        lineage = lineage_response.get("result", {}) if lineage_response.get("status") == "SUCCESS" else {}

        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "sql": generated_sql["sql"],
                "explanation": generated_sql["explanation"],
                "confidence": generated_sql["confidence"],
                "schema": schema_info,
                "authorization": authorization,
                "lineage": lineage
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    async def _validate_sql(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate SQL syntax and semantics."""
        sql = payload.get("sql", "")

        # Mock validation
        errors = []
        if not sql.strip():
            errors.append("Empty SQL statement")
        if "DROP" in sql.upper() and "TABLE" in sql.upper():
            errors.append("DROP TABLE statements are not allowed")

        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "valid": len(errors) == 0,
                "errors": errors
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    async def _optimize_sql(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize SQL query for performance."""
        sql = payload.get("sql", "")

        # Mock optimization
        improvements = []
        optimized_sql = sql

        if "SELECT *" in sql.upper():
            improvements.append("Consider selecting specific columns instead of SELECT *")

        if "WHERE" not in sql.upper() and "FROM" in sql.upper():
            improvements.append("Consider adding a WHERE clause to limit results")

        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "optimizedSql": optimized_sql,
                "improvements": improvements
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    async def _call_agent(
        self,
        agent_id: str,
        capability: str,
        payload: Dict[str, Any],
        trace_id: str
    ) -> Dict[str, Any]:
        """
        Make an A2A call to another agent.

        In a real implementation, this would use HTTP/gRPC with mTLS.
        This mock implementation simulates the response.
        """
        request_id = str(uuid.uuid4())
        logger.info(f"[{trace_id}] -> {agent_id} ({capability}) request={request_id}")

        # Mock responses for demo purposes
        if agent_id == "catalog-agent":
            if capability == "catalog.resolve":
                return {
                    "status": "SUCCESS",
                    "result": {
                        "tables": ["revenue", "regions", "time_periods"],
                        "columns": {
                            "revenue": ["id", "amount", "region_id", "period_id", "created_at"],
                            "regions": ["id", "name", "country"],
                            "time_periods": ["id", "year", "month", "quarter"]
                        },
                        "joins": [
                            {"from": "revenue.region_id", "to": "regions.id"},
                            {"from": "revenue.period_id", "to": "time_periods.id"}
                        ]
                    }
                }
            elif capability == "catalog.lineage":
                return {
                    "status": "SUCCESS",
                    "result": {
                        "upstream": ["raw_transactions", "erp_system"],
                        "downstream": ["revenue_dashboard", "exec_reports"]
                    }
                }

        elif agent_id == "governance-agent":
            if capability == "governance.authorize":
                return {
                    "status": "SUCCESS",
                    "result": {
                        "authorized": True,
                        "restrictions": {
                            "columns": {"revenue.amount": "AGGREGATE_ONLY"},
                            "rows": {"regions.country": ["FR", "DE", "ES"]}
                        },
                        "policiesApplied": ["rgpd.yaml"]
                    }
                }

        return {"status": "ERROR", "error": "Unknown agent or capability"}

    def _mock_generate_sql(
        self,
        question: str,
        schema_info: Dict[str, Any],
        authorization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock SQL generation (replace with LLM in production)."""
        # This would use Claude or another LLM in production
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
            "sql": sql,
            "explanation": "Aggregates monthly revenue by region for 2025, restricted to authorized countries per RGPD policy.",
            "confidence": 0.92
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
        """Run the agent server."""
        host = self.config["server"]["host"]
        port = self.config["server"]["port"]

        logger.info(f"Starting {self.agent_name} on {host}:{port}")

        # In production, this would start an HTTP/gRPC server
        # For demo, we just keep the agent alive
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down agent...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SQL Agent - DATAMESH.AI")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to agent configuration YAML"
    )
    args = parser.parse_args()

    agent = SQLAgent(args.config)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
