"""
Governance Agent - Data governance, authorization, and redaction.

This agent enforces data governance policies, handles authorization
decisions, and applies data redaction based on classification.
"""

__version__ = "1.0.0"

from .handler import GovernanceAgentHandler, handle
from .authorizer import Authorizer, AuthorizationResult
from .redactor import Redactor, RedactionRule

__all__ = [
    "GovernanceAgentHandler",
    "handle",
    "Authorizer",
    "AuthorizationResult",
    "Redactor",
    "RedactionRule",
]
