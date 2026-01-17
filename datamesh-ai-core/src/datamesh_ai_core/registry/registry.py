"""
Agent Registry - Central registry for agent discovery and management.

Provides agent registration, discovery, and capability lookup services.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from .contract import AgentContract

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Central registry for DATAMESH.AI agents.

    Manages agent contracts, provides capability discovery, and
    validates A2A call permissions.
    """

    def __init__(self):
        self._agents: dict[str, AgentContract] = {}
        self._capability_index: dict[str, list[str]] = {}  # capability -> [agent names]

    def register(self, contract: AgentContract) -> None:
        """Register an agent contract."""
        errors = contract.validate()
        if errors:
            raise ValueError(f"Invalid contract: {', '.join(errors)}")

        agent_name = contract.metadata.name
        self._agents[agent_name] = contract

        # Index capabilities
        for cap in contract.capabilities:
            if cap.id not in self._capability_index:
                self._capability_index[cap.id] = []
            if agent_name not in self._capability_index[cap.id]:
                self._capability_index[cap.id].append(agent_name)

        logger.info(f"Registered agent: {agent_name} with {len(contract.capabilities)} capabilities")

    def register_from_yaml(self, yaml_path: str) -> AgentContract:
        """Register an agent from a YAML file."""
        contract = AgentContract.from_yaml_file(yaml_path)
        self.register(contract)
        return contract

    def register_from_directory(self, directory: str, filename: str = "agent.yaml") -> list[AgentContract]:
        """Register all agents from a directory tree."""
        contracts = []
        for root, dirs, files in os.walk(directory):
            if filename in files:
                yaml_path = os.path.join(root, filename)
                try:
                    contract = self.register_from_yaml(yaml_path)
                    contracts.append(contract)
                except Exception as e:
                    logger.error(f"Failed to register agent from {yaml_path}: {e}")
        return contracts

    def unregister(self, agent_name: str) -> bool:
        """Unregister an agent."""
        if agent_name not in self._agents:
            return False

        contract = self._agents[agent_name]

        # Remove from capability index
        for cap in contract.capabilities:
            if cap.id in self._capability_index:
                self._capability_index[cap.id] = [
                    a for a in self._capability_index[cap.id] if a != agent_name
                ]

        del self._agents[agent_name]
        logger.info(f"Unregistered agent: {agent_name}")
        return True

    def get(self, agent_name: str) -> Optional[AgentContract]:
        """Get an agent contract by name."""
        return self._agents.get(agent_name)

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def list_capabilities(self) -> list[str]:
        """List all registered capabilities."""
        return list(self._capability_index.keys())

    def find_agents_by_capability(self, capability_id: str) -> list[str]:
        """Find agents that expose a given capability."""
        return self._capability_index.get(capability_id, [])

    def can_call(self, caller: str, callee: str, capability: str) -> bool:
        """Check if caller agent can call callee's capability."""
        caller_contract = self._agents.get(caller)
        callee_contract = self._agents.get(callee)

        if not caller_contract or not callee_contract:
            return False

        # Check caller has permission to call callee
        if not caller_contract.can_call_agent(callee, capability):
            logger.warning(f"{caller} not authorized to call {callee}.{capability}")
            return False

        # Check callee exposes the capability
        if not callee_contract.get_capability(capability):
            logger.warning(f"{callee} does not expose capability {capability}")
            return False

        return True

    def get_call_constraints(self, caller: str) -> dict:
        """Get A2A call constraints for an agent."""
        contract = self._agents.get(caller)
        if not contract or not contract.a2a:
            return {"maxDepth": 1, "requireTraceParent": True, "allowedOnBehalfOf": False}

        return {
            "maxDepth": contract.a2a.max_depth,
            "requireTraceParent": contract.a2a.require_trace_parent,
            "allowedOnBehalfOf": contract.a2a.allowed_on_behalf_of,
        }

    def get_governance_config(self, agent_name: str) -> Optional[dict]:
        """Get governance configuration for an agent."""
        contract = self._agents.get(agent_name)
        if not contract or not contract.governance:
            return None

        return {
            "classificationAwarenessEnabled": contract.governance.classification_awareness_enabled,
            "blockedClassifications": contract.governance.blocked_classifications,
            "approvalRequiredFor": contract.governance.approval_required_for,
            "policyRefs": contract.governance.policy_refs,
        }

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_name: str) -> bool:
        return agent_name in self._agents
