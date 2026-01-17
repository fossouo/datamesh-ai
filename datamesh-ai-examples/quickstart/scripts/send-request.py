#!/usr/bin/env python3
"""
DATAMESH.AI - Sample Request Sender
====================================
Sends a sample A2A request to the orchestrator and displays the response.
"""

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# For demo purposes, we simulate the request/response locally
# In production, this would use HTTP/gRPC to call the orchestrator


def load_request(request_path: str) -> Dict[str, Any]:
    """Load a request from a JSON file."""
    with open(request_path, 'r') as f:
        return json.load(f)


def create_a2a_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a full A2A request envelope."""
    return {
        "requestId": str(uuid.uuid4()),
        "traceId": str(uuid.uuid4()),
        "capability": "sql.generate",
        "payload": request_data,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def simulate_orchestrator_response(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate the orchestrator processing the request.

    In production, this would be an HTTP/gRPC call to the orchestrator.
    """
    question = request["payload"].get("question", "")
    user = request["payload"].get("user", {})
    policy_refs = request["payload"].get("policyRefs", [])

    # Simulated A2A flow trace
    print("\n" + "=" * 60)
    print("A2A Flow Trace")
    print("=" * 60)
    print(f"TraceId: {request['traceId']}")
    print(f"RequestId: {request['requestId']}")
    print("")
    print(f"[orchestrator] Received request for: sql.generate")
    print(f"[orchestrator] -> sql-agent (sql.generate)")
    print(f"  [sql-agent] -> catalog-agent (catalog.resolve)")
    print(f"  [catalog-agent] Resolved tables: revenue, regions, time_periods")
    print(f"  [sql-agent] <- catalog-agent (SUCCESS)")
    print(f"  [sql-agent] -> governance-agent (governance.authorize)")
    print(f"  [governance-agent] User {user.get('id')} authorized with restrictions")
    print(f"  [governance-agent] Policies applied: {policy_refs}")
    print(f"  [sql-agent] <- governance-agent (SUCCESS)")
    print(f"  [sql-agent] Generating SQL query...")
    print(f"  [sql-agent] -> catalog-agent (catalog.lineage)")
    print(f"  [catalog-agent] Lineage: raw_transactions -> revenue -> revenue_dashboard")
    print(f"  [sql-agent] <- catalog-agent (SUCCESS)")
    print(f"[orchestrator] <- sql-agent (SUCCESS)")
    print("=" * 60)

    # Simulated response
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
        "requestId": request["requestId"],
        "traceId": request["traceId"],
        "status": "SUCCESS",
        "result": {
            "sql": sql,
            "explanation": f"Generated SQL for: '{question}'. Aggregates monthly revenue by region for 2025, restricted to authorized countries (FR, DE, ES) per RGPD policy.",
            "confidence": 0.92,
            "schema": {
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
            "authorization": {
                "authorized": True,
                "restrictions": {
                    "columns": {"revenue.amount": "AGGREGATE_ONLY"},
                    "rows": {"regions.country": ["FR", "DE", "ES"]}
                },
                "policiesApplied": ["rgpd.yaml"]
            },
            "lineage": {
                "upstream": ["raw_transactions", "erp_system", "currency_rates"],
                "downstream": ["revenue_dashboard", "exec_reports", "forecast_model"]
            }
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def display_response(response: Dict[str, Any]) -> None:
    """Display the response in a formatted way."""
    print("\n" + "=" * 60)
    print("Response")
    print("=" * 60)

    print(f"\nStatus: {response['status']}")
    print(f"RequestId: {response['requestId']}")
    print(f"TraceId: {response['traceId']}")

    if response["status"] == "SUCCESS":
        result = response["result"]

        print("\n--- Generated SQL ---")
        print(result["sql"])

        print("\n--- Explanation ---")
        print(result["explanation"])

        print(f"\nConfidence: {result['confidence']:.0%}")

        print("\n--- Schema ---")
        print(f"Tables: {', '.join(result['schema']['tables'])}")

        print("\n--- Authorization ---")
        auth = result["authorization"]
        print(f"Authorized: {auth['authorized']}")
        if auth.get("restrictions", {}).get("columns"):
            print(f"Column restrictions: {auth['restrictions']['columns']}")
        if auth.get("restrictions", {}).get("rows"):
            print(f"Row restrictions: {auth['restrictions']['rows']}")
        print(f"Policies applied: {', '.join(auth.get('policiesApplied', []))}")

        print("\n--- Lineage ---")
        lineage = result.get("lineage", {})
        if lineage.get("upstream"):
            print(f"Upstream: {' -> '.join(lineage['upstream'])}")
        if lineage.get("downstream"):
            print(f"Downstream: {' -> '.join(lineage['downstream'])}")
    else:
        error = response.get("error", {})
        print(f"\nError Code: {error.get('code')}")
        print(f"Error Message: {error.get('message')}")

    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DATAMESH.AI - Send a sample A2A request"
    )
    parser.add_argument(
        "request_file",
        type=str,
        help="Path to the request JSON file"
    )
    parser.add_argument(
        "--orchestrator",
        type=str,
        default="http://localhost:8080",
        help="Orchestrator endpoint (default: http://localhost:8080)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DATAMESH.AI - A2A Demo Request")
    print("=" * 60)

    # Load the request
    print(f"\nLoading request from: {args.request_file}")
    request_data = load_request(args.request_file)

    print("\n--- Request ---")
    print(json.dumps(request_data, indent=2))

    # Create full A2A request
    a2a_request = create_a2a_request(request_data)

    # Send to orchestrator (simulated for demo)
    print(f"\nSending to orchestrator: {args.orchestrator}")
    response = simulate_orchestrator_response(a2a_request)

    # Display the response
    display_response(response)

    # Write audit log
    audit_dir = Path(__file__).parent.parent / "logs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_file = audit_dir / f"{a2a_request['requestId']}.json"

    audit_entry = {
        "requestId": a2a_request["requestId"],
        "traceId": a2a_request["traceId"],
        "request": request_data,
        "response": response,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    with open(audit_file, 'w') as f:
        json.dump(audit_entry, f, indent=2)

    print(f"\nAudit log written to: {audit_file}")
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
