#!/usr/bin/env python3
"""
Catalog Agent - DATAMESH.AI Reference Implementation
=====================================================
Provides schema resolution, metadata lookup, and lineage information.
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
logger = logging.getLogger("catalog-agent")


# Mock catalog data for demo
MOCK_CATALOG = {
    "tables": {
        "revenue": {
            "columns": ["id", "amount", "region_id", "period_id", "created_at", "updated_at"],
            "description": "Monthly revenue records by region",
            "owner": "finance-team",
            "tags": ["pii-free", "revenue", "metrics"]
        },
        "regions": {
            "columns": ["id", "name", "country", "timezone", "manager_id"],
            "description": "Geographic regions for business operations",
            "owner": "operations-team",
            "tags": ["reference", "geography"]
        },
        "time_periods": {
            "columns": ["id", "year", "month", "quarter", "week", "day"],
            "description": "Time dimension table for analytics",
            "owner": "data-platform",
            "tags": ["reference", "time"]
        },
        "customers": {
            "columns": ["id", "name", "email", "phone", "region_id", "created_at"],
            "description": "Customer master data",
            "owner": "crm-team",
            "tags": ["pii", "customer", "gdpr-relevant"]
        },
        "transactions": {
            "columns": ["id", "customer_id", "amount", "product_id", "timestamp"],
            "description": "Individual transaction records",
            "owner": "finance-team",
            "tags": ["transactional", "high-volume"]
        }
    },
    "joins": [
        {"from": "revenue.region_id", "to": "regions.id", "type": "many-to-one"},
        {"from": "revenue.period_id", "to": "time_periods.id", "type": "many-to-one"},
        {"from": "customers.region_id", "to": "regions.id", "type": "many-to-one"},
        {"from": "transactions.customer_id", "to": "customers.id", "type": "many-to-one"}
    ],
    "lineage": {
        "revenue": {
            "upstream": ["raw_transactions", "erp_system", "currency_rates"],
            "downstream": ["revenue_dashboard", "exec_reports", "forecast_model"]
        },
        "customers": {
            "upstream": ["crm_export", "web_signups", "mobile_registrations"],
            "downstream": ["customer_360", "marketing_segments", "churn_model"]
        },
        "transactions": {
            "upstream": ["payment_gateway", "pos_systems", "online_orders"],
            "downstream": ["revenue", "fraud_detection", "analytics_warehouse"]
        }
    }
}


class CatalogAgent:
    """Catalog Agent that provides metadata and lineage services."""

    def __init__(self, config_path: str):
        """Initialize the Catalog Agent with configuration."""
        self.config = self._load_config(config_path)
        self.agent_id = self.config["agent"]["id"]
        self.agent_name = self.config["agent"]["name"]
        self.capabilities = self.config.get("capabilities", [])
        self.catalog = MOCK_CATALOG

        logger.info(f"Initialized {self.agent_name} (ID: {self.agent_id})")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming A2A request."""
        request_id = request.get("requestId", str(uuid.uuid4()))
        trace_id = request.get("traceId", str(uuid.uuid4()))
        capability = request.get("capability")
        payload = request.get("payload", {})

        logger.info(f"[{trace_id}] Handling request {request_id} for capability: {capability}")

        try:
            if capability == "catalog.resolve":
                return await self._resolve(request_id, trace_id, payload)
            elif capability == "catalog.lineage":
                return await self._lineage(request_id, trace_id, payload)
            elif capability == "catalog.search":
                return await self._search(request_id, trace_id, payload)
            else:
                return self._error_response(
                    request_id, trace_id,
                    f"Unknown capability: {capability}",
                    "CAPABILITY_NOT_FOUND"
                )
        except Exception as e:
            logger.exception(f"[{trace_id}] Error handling request")
            return self._error_response(request_id, trace_id, str(e), "INTERNAL_ERROR")

    async def _resolve(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve schema information for tables."""
        question = payload.get("question", "")
        explicit_tables = payload.get("tables", [])

        # Extract table references from question (simple keyword matching for demo)
        detected_tables = self._extract_tables_from_question(question)
        all_tables = list(set(explicit_tables + detected_tables))

        # Build response
        columns = {}
        for table in all_tables:
            if table in self.catalog["tables"]:
                columns[table] = self.catalog["tables"][table]["columns"]

        # Find relevant joins
        joins = [
            j for j in self.catalog["joins"]
            if any(table in j["from"] or table in j["to"] for table in all_tables)
        ]

        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "tables": all_tables,
                "columns": columns,
                "joins": joins
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    async def _lineage(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get lineage for specified tables."""
        tables = payload.get("tables", [])

        upstream = set()
        downstream = set()

        for table in tables:
            if table in self.catalog["lineage"]:
                upstream.update(self.catalog["lineage"][table].get("upstream", []))
                downstream.update(self.catalog["lineage"][table].get("downstream", []))

        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "upstream": list(upstream),
                "downstream": list(downstream)
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    async def _search(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Search for tables and columns."""
        query = payload.get("query", "").lower()
        limit = payload.get("limit", 10)

        results = []
        for table_name, table_info in self.catalog["tables"].items():
            score = 0
            if query in table_name.lower():
                score += 0.8
            if query in table_info["description"].lower():
                score += 0.5
            if any(query in tag.lower() for tag in table_info["tags"]):
                score += 0.3

            if score > 0:
                results.append({
                    "name": table_name,
                    "type": "table",
                    "description": table_info["description"],
                    "score": min(score, 1.0)
                })

        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:limit]

        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "results": results
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def _extract_tables_from_question(self, question: str) -> List[str]:
        """Extract table references from a natural language question."""
        question_lower = question.lower()
        detected = []

        # Simple keyword matching for demo
        keywords_to_tables = {
            "revenue": "revenue",
            "sales": "revenue",
            "income": "revenue",
            "region": "regions",
            "geography": "regions",
            "country": "regions",
            "time": "time_periods",
            "month": "time_periods",
            "year": "time_periods",
            "quarter": "time_periods",
            "customer": "customers",
            "client": "customers",
            "transaction": "transactions",
            "purchase": "transactions",
            "order": "transactions"
        }

        for keyword, table in keywords_to_tables.items():
            if keyword in question_lower and table not in detected:
                detected.append(table)

        return detected

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

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down agent...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Catalog Agent - DATAMESH.AI")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to agent configuration YAML"
    )
    args = parser.parse_args()

    agent = CatalogAgent(args.config)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
