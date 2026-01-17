# DATAMESH.AI — Agent Contract v1

This specification defines the Agent Contract v1 used by DATAMESH.AI to describe agents in a vendor-neutral, auditable, and A2A-friendly manner. It is:

- **LLM-agnostic** – no assumptions about the underlying model.
- **Runtime-agnostic** – portable across orchestrators and execution environments.
- **Security & governance-first** – permissions, policies, and guardrails are explicit.
- **Composable** – standardized interfaces let agents call agents.

---

## 1) Minimal Contract (YAML)

File suggestion: `agent.yaml`

```yaml
apiVersion: datamesh.ai/v1
kind: Agent

metadata:
  name: sql-agent
  displayName: SQL Agent
  description: >
    Generates, validates and optimizes SQL queries against governed datasets.
  version: 1.0.0
  owner:
    team: data-platform
    email: data-platform@example.com
  labels:
    domain: finance
    maturity: beta

spec:
  runtime:
    entrypoint: "datamesh_ai_agents.sql:handler"
    timeoutMs: 30000
    concurrency:
      maxParallel: 10
    retries:
      maxAttempts: 2
      backoffMs: 500

  capabilities:
    - id: sql.generate
      description: Generate SQL from a natural language request.
      inputSchemaRef: "schemas/sql.generate.input.json"
      outputSchemaRef: "schemas/sql.generate.output.json"
    - id: sql.explain
      description: Explain a SQL query and its expected execution plan.
      inputSchemaRef: "schemas/sql.explain.input.json"
      outputSchemaRef: "schemas/sql.explain.output.json"
    - id: sql.optimize
      description: Suggest safe optimizations for a SQL query.
      inputSchemaRef: "schemas/sql.optimize.input.json"
      outputSchemaRef: "schemas/sql.optimize.output.json"

  tools:
    - name: trino
      kind: Connector
      ref: "connectors/trino"
      allow:
        - "query"
        - "explain"
      deny:
        - "write"

  dataAccess:
    defaultPolicy: "deny"
    allowedDatasets:
      - uri: "catalog://finance.customer_transactions"
        permissions:
          - "read"
        constraints:
          rowFilters:
            - "country IN (${user.allowed_countries})"
          columnMasking:
            - column: "card_number"
              method: "mask_last4"

  governance:
    classificationAwareness:
      enabled: true
      blockedClassifications:
        - "PII"
        - "SECRET"
    approvalRequiredFor:
      - "sql.generate"
    policyRefs:
      - "policies/rgpd.yaml"
      - "policies/data-retention.yaml"

  a2a:
    canCall:
      - agentRef: "catalog-agent"
        forCapabilities: ["catalog.resolve", "catalog.lineage"]
      - agentRef: "governance-agent"
        forCapabilities: ["governance.authorize", "governance.redact"]
    callConstraints:
      maxDepth: 3
      requireTraceParent: true
      allowedOnBehalfOf: true

  observability:
    tracing:
      enabled: true
      standard: "opentelemetry"
    audit:
      enabled: true
      logInputs: true
      logOutputs: true
      redactFields:
        - "user.token"
        - "secrets.*"

  safety:
    mode: "governed"
    requireHumanConfirmationFor:
      - "sql.optimize"
    invariants:
      - "NO_WRITE_OPERATIONS"
      - "NO_SCHEMA_CHANGES"

  interfaceHints:
    preferredInputs:
      - name: question
        type: string
        example: "Show me monthly revenue for 2025 by region"
    outputFormats:
      - "json"
      - "markdown"
```

---

## 2) Field-by-field Explanation

- **apiVersion / kind**: Versioned contract (`datamesh.ai/v1`) and resource type (`Agent`).
- **metadata**: Stable identifiers, friendly display names, ownership, labels for governance and domain alignment.
- **spec.runtime**: Entrypoint, timeout, concurrency, and retry policy to ensure predictable operations.
- **spec.capabilities**: Machine-readable capability IDs plus input/output schema references to enforce strict contracts.
- **spec.tools**: Connectors the agent can use with explicit allow/deny scopes.
- **spec.dataAccess**: Least-privilege dataset permissions; constraints support row-level filters and column masking.
- **spec.governance**: Classification awareness, approvals, and links to policy definitions (Rego, YAML, etc.).
- **spec.a2a**: Declares which agents + capabilities can be invoked and under what constraints (depth, trace propagation).
- **spec.observability**: Tracing and audit controls, including redaction lists.
- **spec.safety**: Guardrails and invariants (e.g., no write operations) plus optional human confirmation.
- **spec.interfaceHints**: Optional UI hints decoupled from agent logic.

---

## 3) Capability Schemas (Example)

`schemas/sql.generate.input.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["question", "context"],
  "properties": {
    "question": {"type": "string"},
    "context": {
      "type": "object",
      "properties": {
        "user": {
          "type": "object",
          "properties": {
            "id": {"type": "string"},
            "roles": {"type": "array", "items": {"type": "string"}}
          },
          "required": ["id"]
        },
        "datasetHints": {
          "type": "array",
          "items": {"type": "string"}
        },
        "constraints": {
          "type": "object",
          "properties": {
            "maxRows": {"type": "integer", "minimum": 1, "maximum": 100000}
          }
        }
      }
    }
  }
}
```

---

## 4) Design Notes

- **Strict I/O contracts** prevent “agent hallucinations” at integration boundaries.
- **Least privilege**: tools and datasets are explicitly allowed, limiting blast radius.
- **A2A is first-class**: collaboration is declared in the contract, enabling deterministic orchestration.
- **Auditability**: every action can be traced and reviewed; redactions prevent leakage.
- **Portability**: no dependency on a specific LLM, runtime, or vendor.

---

## 5) Recommended Conventions

- Capability IDs: `<domain>.<verb>` (e.g., `sql.generate`, `spark.plan`, `catalog.lineage`).
- Default policy for data access: `deny`.
- Prefer catalog URIs (`catalog://...`) over raw connection strings.
- Keep UI hints separate from execution logic.

---

## 6) Next: A2A Protocol Minimal

This contract feeds the A2A protocol: it defines who can call what, with which schemas, and under which governance constraints. The upcoming A2A spec will leverage `spec.a2a`, `spec.capabilities`, and `spec.governance` to orchestrate agent interactions safely.
