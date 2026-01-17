"""
DATAMESH.AI Orchestrator
========================

A2A request routing, validation, and agent supervision for the DATAMESH.AI framework.

This package provides:
- Router: A2A request routing with contract validation
- Validator: Contract and policy validation
- Supervisor: Agent health monitoring and supervision
- Server: FastAPI HTTP endpoints for A2A communication
"""

from orchestrator.router import A2ARouter
from orchestrator.validator import ContractValidator, PolicyValidator
from orchestrator.supervisor import AgentSupervisor

__version__ = "1.0.0"
__all__ = [
    "A2ARouter",
    "ContractValidator",
    "PolicyValidator",
    "AgentSupervisor",
]
