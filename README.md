# DATAMESH.AI — Open Agent Framework for Enterprise Data Mesh

DATAMESH.AI is an open-source stack for building, deploying, and orchestrating AI agents directly on enterprise data platforms. It embraces Data Mesh principles, Agent-to-Agent (A2A) collaboration, and zero vendor lock-in.

## Core Principles

- **Open by design**: OSS-first, permissive licenses, no proprietary protocol.
- **No vendor lock-in**: LLM-, storage-, compute-, and cloud-agnostic.
- **Agent-to-Agent from day 1**: agents collaborate, delegate, explain, and validate each other.
- **Enterprise-grade**: security, governance, auditability, and deterministic execution when required.

## Modular Architecture

```
github.com/datamesh-ai/
├── datamesh-ai-core          # Agent lifecycle, registry, A2A comms, tracing
├── datamesh-ai-agents        # SQL / Spark / Hadoop / Catalog / Governance agents
├── datamesh-ai-connectors    # SQL engines, Spark, Hadoop, object storage
├── datamesh-ai-orchestrator  # A2A delegation, validation, supervision
├── datamesh-ai-interfaces    # CLI, API, UI, notebook adapters
├── datamesh-ai-examples      # Quickstarts, enterprise playbooks
└── datamesh-ai-docs          # Specs, contracts, governance guides
```

Each repo is swappable and versioned independently to avoid tight coupling.

## A2A Flow Example

“Generate a Spark job to compute monthly churn”:

1. User → SQL Agent
2. SQL Agent → Catalog Agent (schema validation)
3. SQL Agent → Governance Agent (permission check)
4. SQL Agent → Spark Agent (execution plan + run)
5. Spark Agent → SQL Agent (results + explainability)

No single agent is “smart alone”; collaboration is enforced.

## LLM & Tooling Strategy

- LLM-agnostic: OpenAI-compatible APIs, open-source LLMs, on-prem deployments.
- Tooling-friendly: dbt, Trino/Presto abstractions, open metadata standards.
- LLMs are replaceable dependencies, not architectural anchors.

## Governance & Trust

Every action is logged, traceable, reproducible, and explainable. No black boxes.

## Next Steps

1. Draft the official English README across repos.
2. Define the Agent Contract v1 (YAML/JSON).
3. Design the minimal A2A protocol.
4. Provide quickstarts (local + enterprise).

Contributions and discussions are welcome—DATAMESH.AI aims to become the open foundation for enterprise AI agents on data. 
