"""
Governance Agent Handler - A2A protocol handler for governance operations.

Handles incoming A2A requests and routes them to appropriate
authorization and redaction implementations.
"""

import logging
from typing import Any, Optional

from datamesh_ai_core.a2a import A2AMessage, A2AResponse

from .authorizer import Authorizer, AuthorizationResult
from .redactor import Redactor

logger = logging.getLogger(__name__)


class GovernanceAgentHandler:
    """
    Handler for Governance Agent A2A requests.

    Routes requests to authorization and redaction handlers
    based on the requested capability.
    """

    def __init__(self):
        self.authorizer = Authorizer()
        self.redactor = Redactor()

    async def handle(self, message: A2AMessage) -> A2AResponse:
        """Handle an incoming A2A request."""
        capability = message.callee.capability

        logger.info(
            f"Governance Agent handling {capability} "
            f"request={message.request_id} trace={message.trace.trace_id}"
        )

        try:
            if capability == "governance.authorize":
                return await self._handle_authorize(message)
            elif capability == "governance.redact":
                return await self._handle_redact(message)
            elif capability == "governance.classify":
                return await self._handle_classify(message)
            else:
                return A2AResponse.error(
                    request_id=message.request_id,
                    trace=message.trace,
                    code="UNKNOWN_CAPABILITY",
                    message=f"Unknown capability: {capability}",
                )

        except Exception as e:
            logger.exception(f"Error handling {capability}")
            return A2AResponse.error(
                request_id=message.request_id,
                trace=message.trace,
                code="HANDLER_ERROR",
                message=str(e),
            )

    async def _handle_authorize(self, message: A2AMessage) -> A2AResponse:
        """Handle governance.authorize capability."""
        data = message.payload_ref.data
        user = data.get("user", {})
        action = data.get("action", "read")
        resources = data.get("resources", [])
        context = data.get("context", {})

        # Get policies from request context
        policies = []
        if message.context:
            policies = message.context.policies_applied

        # Perform authorization check
        result = await self.authorizer.authorize(
            user=user,
            action=action,
            resources=resources,
            policies=policies,
            context=context,
        )

        if result.authorized:
            return A2AResponse.success(
                request_id=message.request_id,
                trace=message.trace,
                data={
                    "authorized": True,
                    "user": user,
                    "action": action,
                    "resources": resources,
                    "constraints": result.constraints,
                    "policyRefs": result.policy_refs,
                },
                schema="schemas/governance.authorize.output.json",
                audit_ref=f"audits/governance-agent/{message.request_id}.json",
            )
        else:
            return A2AResponse.success(
                request_id=message.request_id,
                trace=message.trace,
                data={
                    "authorized": False,
                    "reason": result.reason,
                    "user": user,
                    "action": action,
                    "resources": resources,
                    "policyRefs": result.policy_refs,
                },
                schema="schemas/governance.authorize.output.json",
                audit_ref=f"audits/governance-agent/{message.request_id}.json",
            )

    async def _handle_redact(self, message: A2AMessage) -> A2AResponse:
        """Handle governance.redact capability."""
        data = message.payload_ref.data
        records = data.get("records", [])
        classifications = data.get("classifications", {})
        user = data.get("user", {})

        # Get policies from request context
        policies = []
        if message.context:
            policies = message.context.policies_applied

        # Apply redaction
        redacted_records, redaction_report = await self.redactor.redact(
            records=records,
            classifications=classifications,
            user=user,
            policies=policies,
        )

        return A2AResponse.success(
            request_id=message.request_id,
            trace=message.trace,
            data={
                "records": redacted_records,
                "redactionReport": redaction_report,
            },
            schema="schemas/governance.redact.output.json",
            audit_ref=f"audits/governance-agent/{message.request_id}.json",
        )

    async def _handle_classify(self, message: A2AMessage) -> A2AResponse:
        """Handle governance.classify capability."""
        data = message.payload_ref.data
        fields = data.get("fields", [])
        sample_data = data.get("sampleData", {})

        # Classify fields
        classifications = self._classify_fields(fields, sample_data)

        return A2AResponse.success(
            request_id=message.request_id,
            trace=message.trace,
            data={
                "classifications": classifications,
            },
            schema="schemas/governance.classify.output.json",
            audit_ref=f"audits/governance-agent/{message.request_id}.json",
        )

    def _classify_fields(
        self,
        fields: list[dict],
        sample_data: dict,
    ) -> dict[str, list[str]]:
        """Classify fields based on name patterns and sample data."""
        classifications = {}

        pii_patterns = [
            "email", "phone", "ssn", "social_security", "address",
            "name", "first_name", "last_name", "birth", "dob",
        ]
        sensitive_patterns = [
            "password", "secret", "token", "key", "credential",
            "card", "account", "salary", "income",
        ]

        for field in fields:
            field_name = field.get("name", "").lower()
            field_classifications = []

            # Check PII patterns
            for pattern in pii_patterns:
                if pattern in field_name:
                    field_classifications.append("PII")
                    break

            # Check sensitive patterns
            for pattern in sensitive_patterns:
                if pattern in field_name:
                    field_classifications.append("SENSITIVE")
                    break

            if field_classifications:
                classifications[field.get("name", "")] = field_classifications

        return classifications


# Global handler instance
_handler: Optional[GovernanceAgentHandler] = None


async def handle(message: A2AMessage) -> A2AResponse:
    """Entry point for Governance Agent A2A requests."""
    global _handler
    if _handler is None:
        _handler = GovernanceAgentHandler()
    return await _handler.handle(message)
