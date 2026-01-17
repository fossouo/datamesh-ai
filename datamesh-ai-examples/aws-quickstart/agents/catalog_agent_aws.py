"""
AWS-Enhanced Catalog Agent for DataMesh.AI

This agent uses AWS Glue Data Catalog for real schema resolution,
replacing the mock catalog with actual Talki metrics table metadata.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

# Add connectors to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../datamesh-ai-connectors/aws-athena"))

from glue_catalog import GlueCatalogClient, GlueCatalogConfig
from resolver_aws import AWSSchemaResolver, create_talki_resolver


# =============================================================================
# Schema Cache (reduces AWS Glue API calls)
# =============================================================================

class SchemaCache:
    """
    In-memory cache for schema resolution results.

    Reduces AWS Glue API calls for frequently accessed schemas.
    Default TTL: 4 hours (schemas rarely change).
    """

    DEFAULT_TTL_SECONDS = 4 * 60 * 60  # 4 hours

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[Any, float]] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get cached value if not expired."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                self._hits += 1
                return value
            del self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Cache a value with TTL."""
        self._cache[key] = (value, time.time() + self.ttl)

    def invalidate(self, pattern: str | None = None) -> int:
        """Invalidate cache entries matching pattern (or all if None)."""
        if pattern is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        keys_to_delete = [k for k in self._cache if pattern in k]
        for k in keys_to_delete:
            del self._cache[k]
        return len(keys_to_delete)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total * 100) if total > 0 else 0:.1f}%",
            "cached_keys": len(self._cache),
        }

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("catalog-agent-aws")


@dataclass
class AgentConfig:
    """Configuration for the AWS Catalog Agent."""
    agent_id: str = "catalog-agent-aws-001"
    host: str = "0.0.0.0"
    port: int = 8082
    aws_region: str = "eu-west-1"
    stage: str = "prod"


class CatalogAgentAWS:
    """
    Catalog Agent with AWS Glue integration.

    Capabilities:
    - catalog.resolve: Resolve dataset URI to schema from Glue
    - catalog.lineage: Get lineage information
    - catalog.search: Search for datasets
    - catalog.list: List available datasets
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.resolver = create_talki_resolver(stage=config.stage)
        self.schema_cache = SchemaCache(ttl_seconds=4 * 60 * 60)  # 4 hour TTL
        self.list_cache = SchemaCache(ttl_seconds=60 * 60)  # 1 hour TTL for listings
        logger.info(
            f"Initialized {config.agent_id}",
            extra={"stage": config.stage, "region": config.aws_region},
        )

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Route request to appropriate capability handler."""
        capability = request.get("capability", "")
        payload = request.get("payload", {})
        request_id = request.get("requestId", str(uuid.uuid4()))

        logger.info(f"[{request_id}] Handling capability: {capability}")

        try:
            if capability == "catalog.resolve":
                result = self._handle_resolve(payload)
            elif capability == "catalog.lineage":
                result = self._handle_lineage(payload)
            elif capability == "catalog.search":
                result = self._handle_search(payload)
            elif capability == "catalog.list":
                result = self._handle_list(payload)
            else:
                return self._error_response(
                    request_id, f"Unknown capability: {capability}"
                )

            return self._success_response(request_id, capability, result)

        except Exception as e:
            logger.error(f"[{request_id}] Error: {e}", exc_info=True)
            return self._error_response(request_id, str(e))

    def _handle_resolve(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Resolve a dataset URI to its schema (with caching)."""
        dataset_uri = payload.get("dataset") or payload.get("dataset_uri")
        if not dataset_uri:
            raise ValueError("Missing 'dataset' or 'dataset_uri' in payload")

        # Check cache first
        cache_key = f"schema:{dataset_uri}"
        cached = self.schema_cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {dataset_uri}")
            return {**cached, "cache_hit": True}

        # Get schema from AWS Glue
        schema = self.resolver.resolve(dataset_uri)
        details = self.resolver.get_table_details(dataset_uri)

        result = {
            "dataset_uri": dataset_uri,
            "fields": schema,
            "field_count": len(schema),
            "location": details.get("location") if details else None,
            "table_type": details.get("table_type") if details else None,
            "source": "aws_glue",
            "cache_hit": False,
        }

        # Cache the result
        self.schema_cache.set(cache_key, result)
        return result

    def _handle_lineage(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Get lineage information for a dataset."""
        dataset_uri = payload.get("dataset") or payload.get("dataset_uri")
        if not dataset_uri:
            raise ValueError("Missing 'dataset' or 'dataset_uri' in payload")

        lineage = self.resolver.get_lineage(dataset_uri)

        # For Talki metrics, add known lineage relationships
        if "session_logs" in dataset_uri:
            lineage["upstream"] = ["Lambda CloudWatch Logs", "Kinesis Firehose"]
            lineage["downstream"] = ["analytics_dashboard", "cost_reports"]
        elif "interaction_cost" in dataset_uri:
            lineage["upstream"] = ["session_logs", "pricing_config"]
            lineage["downstream"] = ["monthly_cost_summary"]

        return lineage

    def _handle_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Search for datasets matching a pattern."""
        search_text = payload.get("query") or payload.get("search_text", "")
        results = self.resolver.search(search_text)

        return {
            "query": search_text,
            "results": results,
            "count": len(results),
        }

    def _handle_list(self, payload: dict[str, Any]) -> dict[str, Any]:
        """List all available datasets."""
        database_filter = payload.get("database")

        if database_filter:
            tables = self.resolver.list_tables(database_filter)
            datasets = [
                f"catalog://{database_filter}.{t}" for t in tables
            ]
        else:
            datasets = self.resolver.list_datasets()

        return {
            "datasets": datasets,
            "count": len(datasets),
            "databases": self.resolver.list_databases(),
        }

    def _success_response(
        self, request_id: str, capability: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a success response."""
        return {
            "status": "SUCCESS",
            "requestId": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": self.config.agent_id,
            "capability": capability,
            "data": data,
        }

    def _error_response(self, request_id: str, error: str) -> dict[str, Any]:
        """Build an error response."""
        return {
            "status": "ERROR",
            "requestId": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "agent": self.config.agent_id,
            "error": error,
        }


class CatalogAgentHandler(BaseHTTPRequestHandler):
    """HTTP handler for Catalog Agent requests."""

    agent: CatalogAgentAWS = None  # Set by server

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            request = json.loads(body)
            response = self.agent.handle_request(request)
            self._send_json(200, response)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "healthy",
                "agent": self.agent.config.agent_id,
                "capabilities": [
                    "catalog.resolve",
                    "catalog.lineage",
                    "catalog.search",
                    "catalog.list",
                ],
                "cache": {
                    "schema": self.agent.schema_cache.stats(),
                    "list": self.agent.list_cache.stats(),
                },
            })
        elif self.path == "/datasets":
            datasets = self.agent.resolver.list_datasets()
            self._send_json(200, {"datasets": datasets, "count": len(datasets)})
        elif self.path == "/cache/invalidate":
            count = self.agent.schema_cache.invalidate()
            count += self.agent.list_cache.invalidate()
            self._send_json(200, {"invalidated": count})
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
    """Run the AWS Catalog Agent."""
    config = AgentConfig(
        stage=os.environ.get("STAGE", "dev"),
        aws_region=os.environ.get("AWS_REGION", "eu-west-1"),
        port=int(os.environ.get("CATALOG_AGENT_PORT", "8082")),
    )

    agent = CatalogAgentAWS(config)
    CatalogAgentHandler.agent = agent

    server = HTTPServer((config.host, config.port), CatalogAgentHandler)
    logger.info(f"Starting {config.agent_id} on {config.host}:{config.port}")
    logger.info(f"AWS Region: {config.aws_region}, Stage: {config.stage}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
