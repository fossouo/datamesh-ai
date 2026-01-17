#!/usr/bin/env python3
"""
Governance Agent for DataMesh.AI

Handles authorization, policy enforcement, and audit logging
for data access requests in the A2A workflow.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("governance-agent")


@dataclass
class Policy:
    """Data governance policy."""
    name: str
    description: str
    rules: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Configuration for the Governance Agent."""
    agent_id: str = "governance-agent-001"
    host: str = "0.0.0.0"
    port: int = 8083


# Define Talki-specific policies
TALKI_POLICIES = {
    "talki-data-privacy": Policy(
        name="talki-data-privacy",
        description="GDPR-compliant data access policy for Talki user data",
        rules=[
            {
                "name": "allow_hashed_identifiers",
                "column_pattern": r".*_id_hash$",
                "action": "allow",
                "description": "Pre-hashed identifiers are safe to access",
            },
            {
                "name": "mask_raw_identifiers",
                "column_pattern": r"^(family_id|child_id|user_id|session_id)$",
                "action": "mask",
                "mask_method": "hash_sha256",
                "description": "Raw identifiers must be masked",
            },
            {
                "name": "allow_metrics",
                "column_pattern": r"^(cost_|latency_|duration_|input_|output_|total_)",
                "action": "allow",
                "description": "Performance and cost metrics are allowed",
            },
            {
                "name": "allow_dimensions",
                "column_pattern": r"^(language|region|model_|stage|source|year|month|day)",
                "action": "allow",
                "description": "Dimension columns for analysis",
            },
        ],
    ),
    "analyst-read-only": Policy(
        name="analyst-read-only",
        description="Read-only access for analyst role",
        rules=[
            {
                "name": "block_writes",
                "sql_pattern": r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)\b",
                "action": "deny",
                "description": "Write operations are blocked for analysts",
            },
            {
                "name": "require_limit",
                "sql_pattern": r"^SELECT.*(?!LIMIT)",
                "action": "warn",
                "description": "Consider adding LIMIT to queries",
            },
        ],
    ),
}


class GovernanceAgent:
    """
    Governance Agent for authorization and policy enforcement.

    Capabilities:
    - governance.authorize: Check if a request is authorized
    - governance.audit: Log an audit event
    - governance.classify: Classify data sensitivity
    """

    ROLE_PERMISSIONS = {
        "admin": ["read", "write", "delete", "admin"],
        "analyst": ["read"],
        "viewer": ["read"],
        "data-team": ["read", "write"],
        "finance-team": ["read"],
    }

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.policies = TALKI_POLICIES
        self.audit_log: list[dict[str, Any]] = []
        logger.info(f"Initialized {config.agent_id}")
        logger.info(f"Loaded {len(self.policies)} policies")

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Route request to appropriate capability handler."""
        capability = request.get("capability", "")
        payload = request.get("payload", {})
        request_id = request.get("requestId", str(uuid.uuid4()))
        trace = request.get("trace", {})
        trace_id = trace.get("traceId", str(uuid.uuid4()))

        logger.info(f"[{trace_id}] Handling capability: {capability}")

        try:
            if capability == "governance.authorize":
                result = self._handle_authorize(payload, request)
            elif capability == "governance.audit":
                result = self._handle_audit(payload)
            elif capability == "governance.classify":
                result = self._handle_classify(payload)
            else:
                return self._error_response(
                    request_id, trace_id, f"Unknown capability: {capability}"
                )

            # Log audit event
            self._log_audit_event(capability, payload, result, request)

            return self._success_response(request_id, trace_id, capability, result)

        except Exception as e:
            logger.error(f"[{trace_id}] Error: {e}", exc_info=True)
            return self._error_response(request_id, trace_id, str(e))

    def _handle_authorize(
        self, payload: dict[str, Any], request: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Check if a request is authorized.

        Evaluates:
        1. User roles and permissions
        2. SQL operation safety
        3. Column-level access rules
        4. Row-level restrictions
        """
        user = payload.get("user", request.get("context", {}).get("user", {}))
        sql = payload.get("sql", "")
        datasets = payload.get("datasets", [])

        user_id = user.get("id", "anonymous")
        user_roles = user.get("roles", ["viewer"])

        # Check role permissions
        allowed_actions = set()
        for role in user_roles:
            allowed_actions.update(self.ROLE_PERMISSIONS.get(role, []))

        # Check SQL safety (read-only enforcement)
        sql_check = self._check_sql_safety(sql)
        if not sql_check["safe"] and "write" not in allowed_actions:
            return {
                "authorized": False,
                "reason": sql_check["reason"],
                "user_id": user_id,
                "roles": user_roles,
            }

        # Check column access rules
        column_restrictions = self._get_column_restrictions(sql, user_roles)

        # Check row-level restrictions
        row_restrictions = self._get_row_restrictions(user_roles, datasets)

        # Get applied policies
        policies_applied = ["talki-data-privacy"]
        if "analyst" in user_roles:
            policies_applied.append("analyst-read-only")

        return {
            "authorized": True,
            "user_id": user_id,
            "roles": user_roles,
            "permissions": list(allowed_actions),
            "column_restrictions": column_restrictions,
            "row_restrictions": row_restrictions,
            "policies_applied": policies_applied,
            "warnings": sql_check.get("warnings", []),
        }

    def _check_sql_safety(self, sql: str) -> dict[str, Any]:
        """Check SQL for dangerous operations."""
        sql_upper = sql.upper()
        warnings = []

        # Check for write operations
        write_ops = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
        for op in write_ops:
            if re.search(rf"\b{op}\b", sql_upper):
                return {
                    "safe": False,
                    "reason": f"Write operation '{op}' is not allowed",
                }

        # Check for SELECT *
        if re.search(r"SELECT\s+\*", sql_upper):
            warnings.append("Consider specifying columns instead of SELECT *")

        # Check for missing LIMIT
        if "SELECT" in sql_upper and "LIMIT" not in sql_upper:
            warnings.append("Consider adding LIMIT to prevent large result sets")

        # Check for partition filters (Athena optimization)
        if "session_logs" in sql.lower():
            if not any(p in sql.lower() for p in ["year", "month", "day"]):
                warnings.append("Add partition filters (year, month, day) to reduce cost")

        return {"safe": True, "warnings": warnings}

    def _get_column_restrictions(
        self, sql: str, roles: list[str]
    ) -> dict[str, str]:
        """Get column-level restrictions based on policies."""
        restrictions = {}

        # Apply talki-data-privacy policy
        policy = self.policies.get("talki-data-privacy")
        if policy:
            # Check for raw identifier columns
            raw_id_pattern = r"\b(family_id|child_id|user_id)\b"
            if re.search(raw_id_pattern, sql, re.IGNORECASE):
                for match in re.findall(raw_id_pattern, sql, re.IGNORECASE):
                    restrictions[match] = "MUST_USE_HASH_VERSION"

        # Analysts can only see aggregated cost data
        if "analyst" in roles and "cost" in sql.lower():
            restrictions["cost_usd"] = "AGGREGATE_ONLY"

        return restrictions

    def _get_row_restrictions(
        self, roles: list[str], datasets: list[str]
    ) -> dict[str, list[str]]:
        """Get row-level restrictions based on roles."""
        restrictions = {}

        # Example: Regional restrictions
        if "viewer" in roles and "admin" not in roles:
            # Viewers can only see certain regions
            restrictions["region"] = ["europe", "latam"]  # Exclude MENA

        return restrictions

    def _handle_audit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Log an audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": payload.get("event_type", "unknown"),
            "user_id": payload.get("user_id"),
            "action": payload.get("action"),
            "resource": payload.get("resource"),
            "outcome": payload.get("outcome", "success"),
            "details": payload.get("details", {}),
        }

        self.audit_log.append(event)
        logger.info(f"Audit: {event['event_type']} by {event['user_id']}")

        return {
            "logged": True,
            "event_id": str(uuid.uuid4()),
            "timestamp": event["timestamp"],
        }

    def _handle_classify(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Classify data sensitivity level."""
        column_name = payload.get("column", "")
        table_name = payload.get("table", "")

        # Classification rules
        if re.match(r".*_id$", column_name) and not column_name.endswith("_hash"):
            return {
                "column": column_name,
                "classification": "PII",
                "sensitivity": "high",
                "handling": "Must be hashed or masked",
            }

        if re.match(r".*_hash$", column_name):
            return {
                "column": column_name,
                "classification": "PSEUDONYMIZED",
                "sensitivity": "medium",
                "handling": "Safe for analytics",
            }

        if re.match(r"(cost|price|amount|revenue)", column_name, re.IGNORECASE):
            return {
                "column": column_name,
                "classification": "FINANCIAL",
                "sensitivity": "medium",
                "handling": "Aggregate access only for non-finance roles",
            }

        return {
            "column": column_name,
            "classification": "GENERAL",
            "sensitivity": "low",
            "handling": "Standard access",
        }

    def _log_audit_event(
        self,
        capability: str,
        payload: dict[str, Any],
        result: dict[str, Any],
        request: dict[str, Any],
    ) -> None:
        """Log governance action to audit trail."""
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "capability": capability,
            "user": payload.get("user", {}).get("id", "unknown"),
            "authorized": result.get("authorized", True),
            "trace_id": request.get("trace", {}).get("traceId"),
        }
        self.audit_log.append(event)

    def _success_response(
        self, request_id: str, trace_id: str, capability: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "status": "SUCCESS",
            "requestId": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace": {"traceId": trace_id},
            "agent": self.config.agent_id,
            "capability": capability,
            "data": data,
        }

    def _error_response(
        self, request_id: str, trace_id: str, error: str
    ) -> dict[str, Any]:
        return {
            "status": "ERROR",
            "requestId": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace": {"traceId": trace_id},
            "agent": self.config.agent_id,
            "error": error,
        }


class GovernanceAgentHandler(BaseHTTPRequestHandler):
    """HTTP handler for Governance Agent requests."""

    agent: GovernanceAgent = None

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            request = json.loads(body)
            response = self.agent.handle_request(request)
            self._send_json(200, response)
        except json.JSONDecodeError:
            self._send_json(400, {"status": "ERROR", "error": "Invalid JSON"})
        except Exception as e:
            self._send_json(500, {"status": "ERROR", "error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "healthy",
                "agent": self.agent.config.agent_id,
                "capabilities": [
                    "governance.authorize",
                    "governance.audit",
                    "governance.classify",
                ],
                "policies_loaded": list(self.agent.policies.keys()),
            })
        elif self.path == "/policies":
            policies = {
                name: {
                    "description": p.description,
                    "rule_count": len(p.rules),
                }
                for name, p in self.agent.policies.items()
            }
            self._send_json(200, {"policies": policies})
        elif self.path == "/audit":
            self._send_json(200, {
                "audit_log": self.agent.audit_log[-100:],  # Last 100 events
                "total_events": len(self.agent.audit_log),
            })
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
    """Run the Governance Agent."""
    config = AgentConfig(
        port=int(os.environ.get("GOVERNANCE_AGENT_PORT", "8083")),
    )

    agent = GovernanceAgent(config)
    GovernanceAgentHandler.agent = agent

    server = HTTPServer((config.host, config.port), GovernanceAgentHandler)
    logger.info(f"Starting {config.agent_id} on {config.host}:{config.port}")
    logger.info(f"Policies: {list(agent.policies.keys())}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
