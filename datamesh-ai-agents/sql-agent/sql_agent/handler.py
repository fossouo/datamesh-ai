"""
SQL Agent Handler - A2A protocol handler for SQL operations.

Handles incoming A2A requests and routes them to appropriate
capability implementations.
"""

import logging
from typing import Any, Optional

from datamesh_ai_core.a2a import A2AMessage, A2AResponse, A2AClient, TraceContext
from datamesh_ai_core.registry import AgentContract

from .generator import SQLGenerator
from .optimizer import SQLOptimizer
from .explainer import SQLExplainer

logger = logging.getLogger(__name__)


class SQLAgentHandler:
    """
    Handler for SQL Agent A2A requests.

    Routes requests to appropriate capability handlers and
    manages A2A calls to dependent agents.
    """

    def __init__(
        self,
        contract: Optional[AgentContract] = None,
        a2a_client: Optional[A2AClient] = None,
    ):
        self.contract = contract
        self.a2a_client = a2a_client or A2AClient(
            agent_name="sql-agent",
            agent_capability="sql.generate",
        )
        self.generator = SQLGenerator(self.a2a_client)
        self.optimizer = SQLOptimizer()
        self.explainer = SQLExplainer()

    async def handle(self, message: A2AMessage) -> A2AResponse:
        """Handle an incoming A2A request."""
        capability = message.callee.capability

        logger.info(
            f"SQL Agent handling {capability} "
            f"request={message.request_id} trace={message.trace.trace_id}"
        )

        try:
            if capability == "sql.generate":
                return await self._handle_generate(message)
            elif capability == "sql.explain":
                return await self._handle_explain(message)
            elif capability == "sql.optimize":
                return await self._handle_optimize(message)
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

    async def _handle_generate(self, message: A2AMessage) -> A2AResponse:
        """Handle sql.generate capability."""
        data = message.payload_ref.data
        question = data.get("question", "")
        context = data.get("context", {})
        user = context.get("user", {})
        dataset_hints = context.get("datasetHints", [])

        # Step 1: Resolve schema from Catalog Agent
        schema_info = await self._resolve_schema(message, dataset_hints)
        if not schema_info:
            return A2AResponse.error(
                request_id=message.request_id,
                trace=message.trace,
                code="SCHEMA_RESOLUTION_FAILED",
                message="Failed to resolve schema from catalog",
            )

        # Step 2: Check authorization from Governance Agent
        auth_result = await self._check_authorization(message, user, dataset_hints)
        if not auth_result.get("authorized", False):
            return A2AResponse.error(
                request_id=message.request_id,
                trace=message.trace,
                code="FORBIDDEN",
                message=auth_result.get("reason", "Access denied"),
                policy_refs=auth_result.get("policyRefs", []),
            )

        # Step 3: Generate SQL
        sql_result = await self.generator.generate(
            question=question,
            schema_info=schema_info,
            constraints=context.get("constraints", {}),
            user_context=user,
        )

        return A2AResponse.success(
            request_id=message.request_id,
            trace=message.trace,
            data={
                "sql": sql_result["sql"],
                "explanation": sql_result["explanation"],
                "tables": sql_result["tables"],
                "confidence": sql_result["confidence"],
            },
            schema="schemas/sql.generate.output.json",
            audit_ref=f"audits/sql-agent/{message.request_id}.json",
        )

    async def _handle_explain(self, message: A2AMessage) -> A2AResponse:
        """Handle sql.explain capability."""
        data = message.payload_ref.data
        sql = data.get("sql", "")

        explanation = await self.explainer.explain(sql)

        return A2AResponse.success(
            request_id=message.request_id,
            trace=message.trace,
            data={
                "explanation": explanation["explanation"],
                "executionPlan": explanation["execution_plan"],
                "estimatedCost": explanation["estimated_cost"],
                "warnings": explanation["warnings"],
            },
            schema="schemas/sql.explain.output.json",
            audit_ref=f"audits/sql-agent/{message.request_id}.json",
        )

    async def _handle_optimize(self, message: A2AMessage) -> A2AResponse:
        """Handle sql.optimize capability."""
        data = message.payload_ref.data
        sql = data.get("sql", "")

        optimizations = await self.optimizer.optimize(sql)

        return A2AResponse.success(
            request_id=message.request_id,
            trace=message.trace,
            data={
                "originalSql": sql,
                "optimizedSql": optimizations["optimized_sql"],
                "suggestions": optimizations["suggestions"],
                "estimatedImprovement": optimizations["estimated_improvement"],
            },
            schema="schemas/sql.optimize.output.json",
            audit_ref=f"audits/sql-agent/{message.request_id}.json",
        )

    async def _resolve_schema(
        self,
        message: A2AMessage,
        dataset_hints: list[str],
    ) -> Optional[dict]:
        """Call Catalog Agent to resolve schema."""
        try:
            response = await self.a2a_client.call(
                callee_agent="catalog-agent",
                callee_capability="catalog.resolve",
                data={
                    "datasets": dataset_hints or ["catalog://finance.revenue"],
                    "fields": ["*"],
                },
                input_schema="schemas/catalog.resolve.input.json",
                output_schema="schemas/catalog.resolve.output.json",
                parent_trace=message.trace,
                on_behalf_of=message.context.on_behalf_of if message.context else None,
                policies=message.context.policies_applied if message.context else None,
            )

            if response.is_success and response.output:
                return response.output.data
            else:
                logger.warning(f"Schema resolution failed: {response.error}")
                return None

        except Exception as e:
            logger.error(f"Failed to call catalog-agent: {e}")
            return None

    async def _check_authorization(
        self,
        message: A2AMessage,
        user: dict,
        datasets: list[str],
    ) -> dict:
        """Call Governance Agent to check authorization."""
        try:
            response = await self.a2a_client.call(
                callee_agent="governance-agent",
                callee_capability="governance.authorize",
                data={
                    "user": user,
                    "action": "read",
                    "resources": datasets or ["catalog://finance.revenue"],
                    "context": {
                        "capability": "sql.generate",
                        "agent": "sql-agent",
                    },
                },
                input_schema="schemas/governance.authorize.input.json",
                output_schema="schemas/governance.authorize.output.json",
                parent_trace=message.trace,
                on_behalf_of=message.context.on_behalf_of if message.context else None,
                policies=message.context.policies_applied if message.context else None,
            )

            if response.is_success and response.output:
                return response.output.data
            else:
                return {
                    "authorized": False,
                    "reason": response.error.message if response.error else "Unknown error",
                }

        except Exception as e:
            logger.error(f"Failed to call governance-agent: {e}")
            return {"authorized": False, "reason": str(e)}


# Global handler instance
_handler: Optional[SQLAgentHandler] = None


async def handle(message: A2AMessage) -> A2AResponse:
    """Entry point for SQL Agent A2A requests."""
    global _handler
    if _handler is None:
        _handler = SQLAgentHandler()
    return await _handler.handle(message)
