#!/usr/bin/env python3
"""
Test A2A (Agent-to-Agent) Flow

Demonstrates the full DataMesh.AI A2A workflow:
  User → Orchestrator → SQL Agent → Catalog Agent
                                  → Governance Agent
                                  → Athena (execute)
"""

import json
import subprocess
import sys
import time

import httpx

ORCHESTRATOR_URL = "http://localhost:8080"
SQL_AGENT_URL = "http://localhost:8081"
CATALOG_AGENT_URL = "http://localhost:8082"
GOVERNANCE_AGENT_URL = "http://localhost:8083"


def print_header(text: str):
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)


def print_step(step: int, text: str):
    print(f"\n  [{step}] {text}")
    print("  " + "─" * 60)


def check_agent(name: str, url: str) -> bool:
    """Check if an agent is healthy."""
    try:
        response = httpx.get(f"{url}/health", timeout=5)
        if response.status_code == 200:
            print(f"    ✅ {name}: healthy")
            return True
    except Exception:
        pass
    print(f"    ❌ {name}: not available")
    return False


def test_direct_agent_calls():
    """Test each agent directly."""
    print_header("1. DIRECT AGENT COMMUNICATION TEST")

    # Test Catalog Agent
    print_step(1, "Catalog Agent → List datasets")
    response = httpx.post(
        CATALOG_AGENT_URL,
        json={"capability": "catalog.list", "payload": {}},
        timeout=30,
    )
    result = response.json()
    if result.get("status") == "SUCCESS":
        datasets = result.get("data", {}).get("datasets", [])
        print(f"    Found {len(datasets)} datasets")
        for ds in datasets[:3]:
            print(f"      • {ds.split('.')[-1]}")
    else:
        print(f"    Error: {result.get('error')}")

    # Test SQL Agent
    print_step(2, "SQL Agent → Generate SQL")
    response = httpx.post(
        SQL_AGENT_URL,
        json={
            "capability": "sql.generate",
            "payload": {"question": "Show costs by language"},
        },
        timeout=30,
    )
    result = response.json()
    if result.get("status") == "SUCCESS":
        sql = result.get("data", {}).get("sql", "")
        print(f"    Generated SQL ({len(sql)} chars):")
        for line in sql.split("\n")[:3]:
            print(f"      {line}")
        print("      ...")
    else:
        print(f"    Error: {result.get('error')}")

    # Test Governance Agent
    print_step(3, "Governance Agent → Authorize request")
    response = httpx.post(
        GOVERNANCE_AGENT_URL,
        json={
            "capability": "governance.authorize",
            "payload": {
                "user": {"id": "analyst-01", "roles": ["analyst"]},
                "sql": "SELECT language, SUM(cost_usd) FROM session_logs GROUP BY language",
            },
        },
        timeout=30,
    )
    result = response.json()
    if result.get("status") == "SUCCESS":
        data = result.get("data", {})
        print(f"    Authorized: {data.get('authorized')}")
        print(f"    Policies: {data.get('policies_applied')}")
        if data.get("warnings"):
            print(f"    Warnings: {data.get('warnings')}")
    else:
        print(f"    Error: {result.get('error')}")


def test_orchestrator_a2a():
    """Test the full A2A flow through orchestrator."""
    print_header("2. FULL A2A ORCHESTRATOR FLOW")

    print_step(1, "Sending NL query through orchestrator")
    print("    Question: 'Show me session costs by language'")
    print("    User: analyst-01 (roles: analyst, data-team)")
    print()

    start_time = time.time()
    response = httpx.post(
        ORCHESTRATOR_URL,
        json={
            "question": "Show me session costs by language",
            "user": {
                "id": "analyst-01",
                "roles": ["analyst", "data-team"],
            },
            "execute": True,
        },
        timeout=120,
    )
    elapsed = time.time() - start_time
    result = response.json()

    if result.get("status") == "SUCCESS":
        data = result.get("data", {})

        print("  A2A TRACE:")
        print("  " + "─" * 60)
        for step in data.get("a2a_trace", []):
            status_icon = "✅" if step.get("status") == "SUCCESS" else "❌"
            print(f"    Step {step['step']}: {step['agent']}.{step['capability']} {status_icon}")

        print()
        print("  RESULTS:")
        print("  " + "─" * 60)
        print(f"    SQL Generated ({data.get('confidence', 0)*100:.0f}% confidence):")
        for line in data.get("sql", "").split("\n")[:5]:
            print(f"      {line}")

        auth = data.get("authorization", {})
        print(f"\n    Authorization: {'✅ Allowed' if auth.get('authorized') else '❌ Denied'}")
        print(f"    Policies: {auth.get('policies_applied', [])}")

        exec_data = data.get("execution", {})
        if exec_data:
            print(f"\n    Execution:")
            print(f"      Rows returned: {exec_data.get('row_count', 0)}")
            print(f"      Bytes scanned: {exec_data.get('bytes_scanned', 0):,}")
            print(f"      Time: {exec_data.get('execution_time_ms', 0)}ms")

            rows = exec_data.get("rows", [])
            if rows:
                print(f"\n    Data Preview:")
                for row in rows[:5]:
                    print(f"      {json.dumps(row)}")

        print(f"\n    Total A2A time: {elapsed:.2f}s")

    else:
        print(f"    ❌ Error: {result.get('error')}")


def test_governance_enforcement():
    """Test governance enforcement scenarios."""
    print_header("3. GOVERNANCE ENFORCEMENT TEST")

    scenarios = [
        {
            "name": "Analyst tries SELECT *",
            "user": {"id": "analyst-01", "roles": ["analyst"]},
            "sql": "SELECT * FROM session_logs",
            "expect_warnings": True,
        },
        {
            "name": "Analyst tries DELETE (should fail)",
            "user": {"id": "analyst-01", "roles": ["analyst"]},
            "sql": "DELETE FROM session_logs WHERE year = '2024'",
            "expect_denied": True,
        },
        {
            "name": "Admin can do anything",
            "user": {"id": "admin-01", "roles": ["admin"]},
            "sql": "SELECT * FROM session_logs",
            "expect_warnings": True,
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print_step(i, scenario["name"])
        response = httpx.post(
            GOVERNANCE_AGENT_URL,
            json={
                "capability": "governance.authorize",
                "payload": {
                    "user": scenario["user"],
                    "sql": scenario["sql"],
                },
            },
            timeout=30,
        )
        result = response.json()
        data = result.get("data", {})

        authorized = data.get("authorized", False)
        if scenario.get("expect_denied"):
            if not authorized:
                print(f"    ✅ Correctly denied: {data.get('reason')}")
            else:
                print(f"    ❌ Should have been denied!")
        else:
            if authorized:
                print(f"    ✅ Authorized")
                if data.get("warnings"):
                    print(f"    ⚠️  Warnings: {data.get('warnings')}")
            else:
                print(f"    ❌ Denied: {data.get('reason')}")


def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "   🔗 DATAMESH.AI - A2A Communication Test".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    # Check all agents
    print_header("0. AGENT HEALTH CHECK")
    agents_ok = True
    agents_ok &= check_agent("Orchestrator", ORCHESTRATOR_URL)
    agents_ok &= check_agent("SQL Agent", SQL_AGENT_URL)
    agents_ok &= check_agent("Catalog Agent", CATALOG_AGENT_URL)
    agents_ok &= check_agent("Governance Agent", GOVERNANCE_AGENT_URL)

    if not agents_ok:
        print("\n  ❌ Some agents are not available.")
        print("  Start all agents with: make run-a2a")
        return 1

    # Run tests
    test_direct_agent_calls()
    test_orchestrator_a2a()
    test_governance_enforcement()

    print_header("TEST COMPLETE")
    print("""
  The A2A workflow demonstrated:

  1. Orchestrator receives natural language query
  2. Orchestrator calls Catalog Agent to list datasets
  3. Orchestrator calls SQL Agent to generate SQL
  4. Orchestrator calls Governance Agent to authorize
  5. Orchestrator calls SQL Agent to execute via Athena
  6. Results returned with full audit trail

  All agents communicated via HTTP using A2A protocol!
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
