"""
SQL Agent - Generates, validates and optimizes SQL queries.

This agent provides natural language to SQL conversion with
full governance awareness and A2A collaboration.
"""

__version__ = "1.0.0"

from .handler import SQLAgentHandler, handle
from .generator import SQLGenerator
from .optimizer import SQLOptimizer
from .explainer import SQLExplainer

__all__ = [
    "SQLAgentHandler",
    "handle",
    "SQLGenerator",
    "SQLOptimizer",
    "SQLExplainer",
]
