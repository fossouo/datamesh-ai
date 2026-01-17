# DATAMESH.AI Quickstart — Local A2A Runbook

This quickstart spins up three reference agents (`sql-agent`, `catalog-agent`, `governance-agent`) plus the orchestrator locally. In three commands you can trigger a real Agent-to-Agent (A2A) flow:

```
1. git clone https://github.com/datamesh-ai/datamesh-ai-examples.git
2. cd datamesh-ai-examples/quickstart && make bootstrap
3. make run-demo
```

Each step is detailed below for transparency and enterprise readiness.

---

## 0. Prerequisites

- Docker + Docker Compose
- Python 3.11 (for local tooling)
- `make` (used for scripted steps)

Ports used:
- 8080 — orchestrator API
- 4317 — OpenTelemetry collector (optional)
- 9092 — Kafka (optional if you switch to async mode)

---

## 1. Clone the example repo

```bash
git clone https://github.com/datamesh-ai/datamesh-ai-examples.git
cd datamesh-ai-examples/quickstart
```

This directory contains:

- `agents/sql-agent/agent.yaml`
- `agents/catalog-agent/agent.yaml`
- `agents/governance-agent/agent.yaml`
- `orchestrator/config.yaml`
- `Makefile` (bootstrap + demo commands)

---

## 2. Bootstrap the environment

```bash
make bootstrap
```

Bootstrap performs:

1. Creates a Python virtualenv (`.venv`) with the minimal runtime.
2. Installs dependencies for the three agents and orchestrator.
3. Generates local TLS certificates (`certs/`) for mutual TLS between components.
4. Loads baseline policies into `policies/` (e.g., `rgpd.yaml`, `data-retention.yaml`).
5. Starts Docker Compose stack (Postgres catalog, policy mock, OpenTelemetry collector).

Everything is logged under `bootlogs/`.

---

## 3. Run the demo (A2A flow)

```bash
make run-demo
```

`run-demo` does:

1. Launches the orchestrator (`datamesh-ai-orchestrator`) with the agent registry pointing to the three local agents.
2. Starts each agent as a lightweight async worker.
3. Sends a sample request (`requests/monthly_churn.sql.json`) to the orchestrator:

   ```json
   {
     "question": "Show monthly revenue per region for 2025",
     "user": {"id": "user-42", "roles": ["analyst"]},
     "policyRefs": ["policies/rgpd.yaml"]
   }
   ```

4. SQL Agent interprets the question, then invokes:
   - Catalog Agent → capability `catalog.resolve` (fetch schema)
   - Governance Agent → capability `governance.authorize` (check permission)
   - Catalog Agent → capability `catalog.lineage` (optional lineage)
5. SQL Agent assembles the final query and returns a plan (deterministic mode) to the demo CLI.

During the run:

- Traces appear in your OpenTelemetry collector (`localhost:4318` if enabled).
- Audit logs are written to `logs/audit/`.
- Each A2A hop is printed in the terminal with `traceId`, `requestId`, `status`.

---

## 4. Inspecting the flow

### Orchestrator logs

```bash
tail -f logs/orchestrator.log
```

You should see entries such as:

```
[sql-agent] -> catalog-agent (catalog.resolve) trace=a1b2c3 request=req-123
[catalog-agent] -> governance-agent (governance.authorize) trace=a1b2c3 request=req-124
```

### Agent audit trails

Stored under `logs/audit/<agent>/<requestId>.json`. Example:

```json
{
  "requestId": "req-124",
  "status": "SUCCESS",
  "calledBy": "catalog-agent",
  "capability": "catalog.resolve",
  "traceId": "a1b2c3d4",
  "policiesApplied": ["policies/rgpd.yaml"]
}
```

---

## 5. Switching to async mode (optional)

```bash
ASYNC=1 make run-demo
```

- Orchestrator routes SQL Agent → Catalog Agent via Kafka.
- Responses use `status: IN_PROGRESS` with `nextPollAfterMs`.
- Demonstrates the same protocol on a message bus.

---

## 6. Cleanup

```bash
make down
```

Stops containers, tears down agents, and removes transient logs (keeps audit logs unless `CLEAN_AUDIT=1 make down`).

---

## 7. Extending the demo

- Add new capabilities by editing `agents/*/agent.yaml` and re-running `make run-demo`.
- Plug in your own catalog or governance service by swapping connectors in `connectors/`.
- Point the orchestrator to an enterprise OpenTelemetry endpoint rather than local collector.

---

## Recap (3 commands)

```
git clone https://github.com/datamesh-ai/datamesh-ai-examples.git
cd datamesh-ai-examples/quickstart && make bootstrap
make run-demo
```

You now have a reproducible, enterprise-ready A2A flow running locally, demonstrating SQL Agent delegating to Catalog and Governance Agents with full trace/audit coverage. This quickstart is the foundation for extending DATAMESH.AI into your own environment. 
