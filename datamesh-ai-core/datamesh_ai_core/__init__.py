"""
DATAMESH.AI Core - Agent lifecycle, registry, A2A communications, and tracing.

This package provides the foundational components for building AI agents
in the DATAMESH.AI ecosystem.
"""

from datamesh_ai_core.models import (
    A2ARequest,
    A2AResponse,
    A2AStatus,
    TraceContext,
    AgentCapability,
    AgentMetadata,
)
from datamesh_ai_core.agent import Agent, AgentState
from datamesh_ai_core.registry import AgentRegistry
from datamesh_ai_core.a2a import A2AHandler, A2AClient
from datamesh_ai_core.tracing import TracingManager, create_trace_context
from datamesh_ai_core.contracts import ContractLoader, ContractValidator

__version__ = "0.1.0"
__all__ = [
    # Models
    "A2ARequest",
    "A2AResponse",
    "A2AStatus",
    "TraceContext",
    "AgentCapability",
    "AgentMetadata",
    # Agent
    "Agent",
    "AgentState",
    # Registry
    "AgentRegistry",
    # A2A Protocol
    "A2AHandler",
    "A2AClient",
    # Tracing
    "TracingManager",
    "create_trace_context",
    # Contracts
    "ContractLoader",
    "ContractValidator",
]
