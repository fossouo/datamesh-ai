#!/usr/bin/env python3
"""
Governance Agent - DATAMESH.AI Reference Implementation
========================================================
Handles authorization, policy enforcement, and compliance checks.
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
from typing import Any, Dict, List, Optional, Set

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("governance-agent")


# Role-based access control configuration
RBAC_CONFIG = {
    "roles": {
        "analyst": {
            "permissions": ["SELECT"],
            "allowed_tables": ["revenue", "regions", "time_periods"],
            "denied_tables": ["customers"],
            "column_restrictions": {
                "revenue.amount": "AGGREGATE_ONLY"
            },
            "row_filters": {
                "regions.country": ["FR", "DE", "ES", "IT", "NL"]
            }
        },
        "data_engineer": {
            "permissions": ["SELECT", "INSERT", "UPDATE"],
            "allowed_tables": ["*"],
            "denied_tables": [],
            "column_restrictions": {},
            "row_filters": {}
        },
        "admin": {
            "permissions": ["SELECT", "INSERT", "UPDATE", "DELETE", "ADMIN"],
            "allowed_tables": ["*"],
            "denied_tables": [],
            "column_restrictions": {},
            "row_filters": {}
        },
        "compliance_officer": {
            "permissions": ["SELECT"],
            "allowed_tables": ["*"],
            "denied_tables": [],
            "column_restrictions": {},
            "row_filters": {},
            "audit_access": True
        }
    }
}

# Data classification levels
DATA_CLASSIFICATIONS = {
    "customers.email": "PII",
    "customers.phone": "PII",
    "customers.name": "PII",
    "transactions.amount": "CONFIDENTIAL",
    "revenue.amount": "CONFIDENTIAL",
    "regions.name": "PUBLIC",
    "time_periods.year": "PUBLIC"
}


class GovernanceAgent:
    """Governance Agent that enforces policies and authorization."""

    def __init__(self, config_path: str):
        """Initialize the Governance Agent with configuration."""
        self.config = self._load_config(config_path)
        self.agent_id = self.config["agent"]["id"]
        self.agent_name = self.config["agent"]["name"]
        self.capabilities = self.config.get("capabilities", [])
        self.policies = self._load_policies()
        self.rbac = RBAC_CONFIG

        logger.info(f"Initialized {self.agent_name} (ID: {self.agent_id})")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_policies(self) -> Dict[str, Any]:
        """Load policy files from the policies directory."""
        policies = {}
        policies_dir = self.config.get("policies", {}).get("directory", "../../policies")
        policy_files = self.config.get("policies", {}).get("files", [])

        for policy_file in policy_files:
            policy_path = Path(policies_dir) / policy_file
            if policy_path.exists():
                with open(policy_path, 'r') as f:
                    policies[policy_file] = yaml.safe_load(f)
                logger.info(f"Loaded policy: {policy_file}")

        return policies

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming A2A request."""
        request_id = request.get("requestId", str(uuid.uuid4()))
        trace_id = request.get("traceId", str(uuid.uuid4()))
        capability = request.get("capability")
        payload = request.get("payload", {})

        logger.info(f"[{trace_id}] Handling request {request_id} for capability: {capability}")

        try:
            if capability == "governance.authorize":
                return await self._authorize(request_id, trace_id, payload)
            elif capability == "governance.audit":
                return await self._audit(request_id, trace_id, payload)
            elif capability == "governance.classify":
                return await self._classify(request_id, trace_id, payload)
            else:
                return self._error_response(
                    request_id, trace_id,
                    f"Unknown capability: {capability}",
                    "CAPABILITY_NOT_FOUND"
                )
        except Exception as e:
            logger.exception(f"[{trace_id}] Error handling request")
            return self._error_response(request_id, trace_id, str(e), "INTERNAL_ERROR")

    async def _authorize(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check authorization for a user action."""
        user = payload.get("user", {})
        resources = payload.get("resource", [])
        action = payload.get("action", "SELECT")
        policy_refs = payload.get("policyRefs", [])

        user_id = user.get("id", "unknown")
        user_roles = user.get("roles", [])

        logger.info(f"[{trace_id}] Authorizing user {user_id} for {action} on {resources}")

        # Aggregate permissions from all roles
        allowed_tables: Set[str] = set()
        denied_tables: Set[str] = set()
        allowed_actions: Set[str] = set()
        column_restrictions: Dict[str, str] = {}
        row_filters: Dict[str, List[str]] = {}
        policies_applied: List[str] = []

        for role in user_roles:
            if role in self.rbac["roles"]:
                role_config = self.rbac["roles"][role]

                # Actions
                allowed_actions.update(role_config["permissions"])

                # Tables
                if "*" in role_config["allowed_tables"]:
                    allowed_tables.add("*")
                else:
                    allowed_tables.update(role_config["allowed_tables"])
                denied_tables.update(role_config["denied_tables"])

                # Restrictions
                column_restrictions.update(role_config.get("column_restrictions", {}))
                for col, values in role_config.get("row_filters", {}).items():
                    if col not in row_filters:
                        row_filters[col] = []
                    row_filters[col].extend(values)

        # Check action permission
        if action not in allowed_actions:
            return {
                "requestId": request_id,
                "traceId": trace_id,
                "status": "SUCCESS",
                "result": {
                    "authorized": False,
                    "denialReason": f"Action '{action}' not permitted for user roles: {user_roles}",
                    "policiesApplied": []
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

        # Check table access
        for resource in resources:
            table = resource.split(".")[0] if "." in resource else resource
            if table in denied_tables:
                return {
                    "requestId": request_id,
                    "traceId": trace_id,
                    "status": "SUCCESS",
                    "result": {
                        "authorized": False,
                        "denialReason": f"Access to table '{table}' denied for user roles: {user_roles}",
                        "policiesApplied": []
                    },
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            if "*" not in allowed_tables and table not in allowed_tables:
                return {
                    "requestId": request_id,
                    "traceId": trace_id,
                    "status": "SUCCESS",
                    "result": {
                        "authorized": False,
                        "denialReason": f"Table '{table}' not in allowed list for user roles: {user_roles}",
                        "policiesApplied": []
                    },
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }

        # Apply RGPD policy if referenced
        if any("rgpd" in ref.lower() for ref in policy_refs):
            policies_applied.append("rgpd.yaml")
            # Add RGPD-specific restrictions
            if "customers" in resources:
                column_restrictions["customers.email"] = "MASKED"
                column_restrictions["customers.phone"] = "MASKED"

        # Apply data retention policy if referenced
        if any("retention" in ref.lower() for ref in policy_refs):
            policies_applied.append("data-retention.yaml")

        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "authorized": True,
                "restrictions": {
                    "columns": column_restrictions,
                    "rows": row_filters
                },
                "policiesApplied": policies_applied
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    async def _audit(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Log an audit event."""
        audit_id = str(uuid.uuid4())

        audit_entry = {
            "auditId": audit_id,
            "traceId": trace_id,
            "user": payload.get("user", {}),
            "resource": payload.get("resource", []),
            "action": payload.get("action", ""),
            "result": payload.get("result", ""),
            "timestamp": payload.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            "agentId": self.agent_id
        }

        # Write to audit log
        audit_dir = Path(self.config.get("audit", {}).get("storage", "../../logs/audit"))
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_file = audit_dir / f"{audit_id}.json"

        with open(audit_file, 'w') as f:
            json.dump(audit_entry, f, indent=2)

        logger.info(f"[{trace_id}] Audit log written: {audit_id}")

        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "auditId": audit_id,
                "logged": True
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    async def _classify(
        self,
        request_id: str,
        trace_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify data sensitivity."""
        tables = payload.get("tables", [])
        columns = payload.get("columns", [])

        classifications = {}

        # Classify tables
        for table in tables:
            # Check if any columns in this table are classified
            table_classifications = [
                (col, level)
                for col, level in DATA_CLASSIFICATIONS.items()
                if col.startswith(f"{table}.")
            ]
            if table_classifications:
                # Table classification is the highest level of its columns
                levels = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "PII", "RESTRICTED"]
                max_level = max(
                    (levels.index(level) for _, level in table_classifications),
                    default=0
                )
                classifications[table] = levels[max_level]
            else:
                classifications[table] = "UNCLASSIFIED"

        # Classify specific columns
        for column in columns:
            if column in DATA_CLASSIFICATIONS:
                classifications[column] = DATA_CLASSIFICATIONS[column]
            else:
                classifications[column] = "UNCLASSIFIED"

        return {
            "requestId": request_id,
            "traceId": trace_id,
            "status": "SUCCESS",
            "result": {
                "classifications": classifications
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
    parser = argparse.ArgumentParser(description="Governance Agent - DATAMESH.AI")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to agent configuration YAML"
    )
    args = parser.parse_args()

    agent = GovernanceAgent(args.config)
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
