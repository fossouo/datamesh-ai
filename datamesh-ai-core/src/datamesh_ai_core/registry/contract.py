"""
Agent Contract - DATAMESH.AI Agent Contract v1 implementation.

Defines the structure and validation for agent contracts following
the datamesh.ai/v1 specification.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import yaml
import json


class DataAccessPolicy(Enum):
    """Default data access policy."""
    ALLOW = "allow"
    DENY = "deny"


class SafetyMode(Enum):
    """Agent safety modes."""
    GOVERNED = "governed"
    AUTONOMOUS = "autonomous"
    SUPERVISED = "supervised"


@dataclass
class Capability:
    """Agent capability definition."""
    id: str
    description: str
    input_schema_ref: str
    output_schema_ref: str

    @classmethod
    def from_dict(cls, data: dict) -> "Capability":
        return cls(
            id=data["id"],
            description=data["description"],
            input_schema_ref=data.get("inputSchemaRef", ""),
            output_schema_ref=data.get("outputSchemaRef", ""),
        )


@dataclass
class Tool:
    """Tool/connector definition."""
    name: str
    kind: str
    ref: str
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Tool":
        return cls(
            name=data["name"],
            kind=data["kind"],
            ref=data["ref"],
            allow=data.get("allow", []),
            deny=data.get("deny", []),
        )


@dataclass
class A2ACallTarget:
    """A2A call target definition."""
    agent_ref: str
    for_capabilities: list[str]

    @classmethod
    def from_dict(cls, data: dict) -> "A2ACallTarget":
        return cls(
            agent_ref=data["agentRef"],
            for_capabilities=data.get("forCapabilities", []),
        )


@dataclass
class A2AConfig:
    """A2A configuration."""
    can_call: list[A2ACallTarget] = field(default_factory=list)
    max_depth: int = 3
    require_trace_parent: bool = True
    allowed_on_behalf_of: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "A2AConfig":
        can_call = [A2ACallTarget.from_dict(c) for c in data.get("canCall", [])]
        constraints = data.get("callConstraints", {})
        return cls(
            can_call=can_call,
            max_depth=constraints.get("maxDepth", 3),
            require_trace_parent=constraints.get("requireTraceParent", True),
            allowed_on_behalf_of=constraints.get("allowedOnBehalfOf", False),
        )


@dataclass
class GovernanceConfig:
    """Governance configuration."""
    classification_awareness_enabled: bool = True
    blocked_classifications: list[str] = field(default_factory=list)
    approval_required_for: list[str] = field(default_factory=list)
    policy_refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "GovernanceConfig":
        class_awareness = data.get("classificationAwareness", {})
        return cls(
            classification_awareness_enabled=class_awareness.get("enabled", True),
            blocked_classifications=class_awareness.get("blockedClassifications", []),
            approval_required_for=data.get("approvalRequiredFor", []),
            policy_refs=data.get("policyRefs", []),
        )


@dataclass
class ObservabilityConfig:
    """Observability configuration."""
    tracing_enabled: bool = True
    tracing_standard: str = "opentelemetry"
    audit_enabled: bool = True
    log_inputs: bool = True
    log_outputs: bool = True
    redact_fields: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "ObservabilityConfig":
        tracing = data.get("tracing", {})
        audit = data.get("audit", {})
        return cls(
            tracing_enabled=tracing.get("enabled", True),
            tracing_standard=tracing.get("standard", "opentelemetry"),
            audit_enabled=audit.get("enabled", True),
            log_inputs=audit.get("logInputs", True),
            log_outputs=audit.get("logOutputs", True),
            redact_fields=audit.get("redactFields", []),
        )


@dataclass
class SafetyConfig:
    """Safety configuration."""
    mode: SafetyMode = SafetyMode.GOVERNED
    require_human_confirmation_for: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SafetyConfig":
        mode_str = data.get("mode", "governed")
        return cls(
            mode=SafetyMode(mode_str),
            require_human_confirmation_for=data.get("requireHumanConfirmationFor", []),
            invariants=data.get("invariants", []),
        )


@dataclass
class RuntimeConfig:
    """Runtime configuration."""
    entrypoint: str
    timeout_ms: int = 30000
    max_parallel: int = 10
    max_retries: int = 2
    backoff_ms: int = 500

    @classmethod
    def from_dict(cls, data: dict) -> "RuntimeConfig":
        concurrency = data.get("concurrency", {})
        retries = data.get("retries", {})
        return cls(
            entrypoint=data["entrypoint"],
            timeout_ms=data.get("timeoutMs", 30000),
            max_parallel=concurrency.get("maxParallel", 10),
            max_retries=retries.get("maxAttempts", 2),
            backoff_ms=retries.get("backoffMs", 500),
        )


@dataclass
class AgentMetadata:
    """Agent metadata."""
    name: str
    display_name: str
    description: str
    version: str
    owner_team: str
    owner_email: str
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentMetadata":
        owner = data.get("owner", {})
        return cls(
            name=data["name"],
            display_name=data.get("displayName", data["name"]),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            owner_team=owner.get("team", ""),
            owner_email=owner.get("email", ""),
            labels=data.get("labels", {}),
        )


@dataclass
class AgentContract:
    """
    DATAMESH.AI Agent Contract v1.

    Defines the complete contract for an agent including metadata,
    capabilities, tools, data access, governance, A2A config, and safety.
    """
    api_version: str
    kind: str
    metadata: AgentMetadata
    runtime: RuntimeConfig
    capabilities: list[Capability]
    tools: list[Tool] = field(default_factory=list)
    a2a: Optional[A2AConfig] = None
    governance: Optional[GovernanceConfig] = None
    observability: Optional[ObservabilityConfig] = None
    safety: Optional[SafetyConfig] = None

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "AgentContract":
        """Parse agent contract from YAML string."""
        data = yaml.safe_load(yaml_content)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, file_path: str) -> "AgentContract":
        """Load agent contract from YAML file."""
        with open(file_path, "r") as f:
            return cls.from_yaml(f.read())

    @classmethod
    def from_dict(cls, data: dict) -> "AgentContract":
        """Parse agent contract from dictionary."""
        spec = data.get("spec", {})

        capabilities = [
            Capability.from_dict(c) for c in spec.get("capabilities", [])
        ]
        tools = [Tool.from_dict(t) for t in spec.get("tools", [])]

        return cls(
            api_version=data.get("apiVersion", "datamesh.ai/v1"),
            kind=data.get("kind", "Agent"),
            metadata=AgentMetadata.from_dict(data.get("metadata", {})),
            runtime=RuntimeConfig.from_dict(spec.get("runtime", {})),
            capabilities=capabilities,
            tools=tools,
            a2a=A2AConfig.from_dict(spec["a2a"]) if "a2a" in spec else None,
            governance=GovernanceConfig.from_dict(spec["governance"]) if "governance" in spec else None,
            observability=ObservabilityConfig.from_dict(spec["observability"]) if "observability" in spec else None,
            safety=SafetyConfig.from_dict(spec["safety"]) if "safety" in spec else None,
        )

    def to_dict(self) -> dict:
        """Convert contract to dictionary."""
        return {
            "apiVersion": self.api_version,
            "kind": self.kind,
            "metadata": {
                "name": self.metadata.name,
                "displayName": self.metadata.display_name,
                "description": self.metadata.description,
                "version": self.metadata.version,
                "owner": {
                    "team": self.metadata.owner_team,
                    "email": self.metadata.owner_email,
                },
                "labels": self.metadata.labels,
            },
            "spec": {
                "runtime": {
                    "entrypoint": self.runtime.entrypoint,
                    "timeoutMs": self.runtime.timeout_ms,
                    "concurrency": {"maxParallel": self.runtime.max_parallel},
                    "retries": {
                        "maxAttempts": self.runtime.max_retries,
                        "backoffMs": self.runtime.backoff_ms,
                    },
                },
                "capabilities": [
                    {
                        "id": c.id,
                        "description": c.description,
                        "inputSchemaRef": c.input_schema_ref,
                        "outputSchemaRef": c.output_schema_ref,
                    }
                    for c in self.capabilities
                ],
            },
        }

    def get_capability(self, capability_id: str) -> Optional[Capability]:
        """Get capability by ID."""
        for cap in self.capabilities:
            if cap.id == capability_id:
                return cap
        return None

    def can_call_agent(self, agent_ref: str, capability: str) -> bool:
        """Check if this agent can call another agent's capability."""
        if not self.a2a:
            return False
        for target in self.a2a.can_call:
            if target.agent_ref == agent_ref and capability in target.for_capabilities:
                return True
        return False

    def validate(self) -> list[str]:
        """Validate the contract and return list of errors."""
        errors = []

        if not self.metadata.name:
            errors.append("metadata.name is required")
        if not self.runtime.entrypoint:
            errors.append("spec.runtime.entrypoint is required")
        if not self.capabilities:
            errors.append("spec.capabilities must have at least one capability")

        for cap in self.capabilities:
            if not cap.id:
                errors.append("capability.id is required")
            if not cap.input_schema_ref:
                errors.append(f"capability {cap.id} missing inputSchemaRef")
            if not cap.output_schema_ref:
                errors.append(f"capability {cap.id} missing outputSchemaRef")

        return errors
