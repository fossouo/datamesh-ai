"""
A2A Client - HTTP client for Agent-to-Agent communication.

Provides a simple interface for sending A2A messages between agents.
"""

import asyncio
import logging
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
import json

from .message import (
    A2AMessage,
    A2AResponse,
    A2AStatus,
    TraceContext,
    OnBehalfOf,
)

logger = logging.getLogger(__name__)


@dataclass
class A2AClientConfig:
    """A2A client configuration."""
    base_url: str = "http://localhost:8080"
    timeout_ms: int = 30000
    max_retries: int = 2
    backoff_ms: int = 500
    verify_ssl: bool = True
    mtls_cert_path: Optional[str] = None
    mtls_key_path: Optional[str] = None


class A2AClient:
    """
    HTTP client for A2A protocol communication.

    Handles message routing, retry logic, and response parsing.
    """

    def __init__(
        self,
        config: Optional[A2AClientConfig] = None,
        agent_name: str = "",
        agent_capability: str = "",
    ):
        self.config = config or A2AClientConfig()
        self.agent_name = agent_name
        self.agent_capability = agent_capability
        self._handlers: dict[str, Callable] = {}

    async def call(
        self,
        callee_agent: str,
        callee_capability: str,
        data: dict[str, Any],
        input_schema: str = "",
        output_schema: str = "",
        deadline_ms: Optional[int] = None,
        parent_trace: Optional[TraceContext] = None,
        on_behalf_of: Optional[OnBehalfOf] = None,
        policies: Optional[list[str]] = None,
    ) -> A2AResponse:
        """
        Make an A2A call to another agent.

        Args:
            callee_agent: Target agent name
            callee_capability: Target capability ID
            data: Request payload data
            input_schema: Input schema reference
            output_schema: Output schema reference
            deadline_ms: Request deadline in milliseconds
            parent_trace: Parent trace context for propagation
            on_behalf_of: Delegation context
            policies: List of policy references

        Returns:
            A2AResponse from the callee agent
        """
        message = A2AMessage.new(
            caller_agent=self.agent_name,
            caller_capability=self.agent_capability,
            callee_agent=callee_agent,
            callee_capability=callee_capability,
            data=data,
            input_schema=input_schema,
            output_schema=output_schema,
            deadline_ms=deadline_ms or self.config.timeout_ms,
            parent_trace=parent_trace,
            on_behalf_of=on_behalf_of,
            policies=policies,
        )

        return await self._send_with_retry(message)

    async def _send_with_retry(self, message: A2AMessage) -> A2AResponse:
        """Send message with retry logic."""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._send(message)

                # Handle IN_PROGRESS responses with polling
                if response.is_in_progress:
                    response = await self._poll_until_complete(message, response)

                return response

            except Exception as e:
                last_error = e
                logger.warning(
                    f"A2A call failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}"
                )

                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.backoff_ms / 1000)

        # All retries failed
        return A2AResponse.error(
            request_id=message.request_id,
            trace=message.trace,
            code="A2A_CALL_FAILED",
            message=f"Failed after {self.config.max_retries + 1} attempts: {last_error}",
        )

    async def _send(self, message: A2AMessage) -> A2AResponse:
        """Send a single A2A message."""
        # Check if we have a local handler registered
        handler_key = f"{message.callee.agent}:{message.callee.capability}"
        if handler_key in self._handlers:
            return await self._handlers[handler_key](message)

        # HTTP transport (requires aiohttp)
        try:
            import aiohttp

            url = f"{self.config.base_url}/a2a/{message.callee.agent}/{message.callee.capability}"

            ssl_context = None
            if self.config.mtls_cert_path and self.config.mtls_key_path:
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.load_cert_chain(
                    self.config.mtls_cert_path,
                    self.config.mtls_key_path,
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=message.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=message.deadline_ms / 1000),
                    ssl=ssl_context if self.config.verify_ssl else False,
                ) as resp:
                    response_data = await resp.json()
                    return A2AResponse.from_dict(response_data)

        except ImportError:
            # Fallback for sync context or missing aiohttp
            logger.warning("aiohttp not available, using local handlers only")
            return A2AResponse.error(
                request_id=message.request_id,
                trace=message.trace,
                code="TRANSPORT_UNAVAILABLE",
                message="HTTP transport not available (aiohttp not installed)",
            )

    async def _poll_until_complete(
        self,
        original_message: A2AMessage,
        in_progress_response: A2AResponse,
    ) -> A2AResponse:
        """Poll for completion of async operation."""
        response = in_progress_response
        total_wait = 0
        max_wait = original_message.deadline_ms

        while response.is_in_progress and total_wait < max_wait:
            wait_ms = response.next_poll_after_ms or 2000
            await asyncio.sleep(wait_ms / 1000)
            total_wait += wait_ms

            # Poll for status
            try:
                import aiohttp

                url = f"{self.config.base_url}/a2a/status/{original_message.request_id}"

                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        response_data = await resp.json()
                        response = A2AResponse.from_dict(response_data)

            except Exception as e:
                logger.warning(f"Poll failed: {e}")
                continue

        if response.is_in_progress:
            return A2AResponse.error(
                request_id=original_message.request_id,
                trace=original_message.trace,
                code="DEADLINE_EXCEEDED",
                message=f"Operation did not complete within {max_wait}ms",
            )

        return response

    def register_handler(
        self,
        agent_name: str,
        capability: str,
        handler: Callable[[A2AMessage], A2AResponse],
    ) -> None:
        """
        Register a local handler for testing or in-process communication.

        Args:
            agent_name: Target agent name
            capability: Target capability ID
            handler: Async function that handles the message
        """
        key = f"{agent_name}:{capability}"
        self._handlers[key] = handler
        logger.debug(f"Registered local handler for {key}")

    def unregister_handler(self, agent_name: str, capability: str) -> bool:
        """Unregister a local handler."""
        key = f"{agent_name}:{capability}"
        if key in self._handlers:
            del self._handlers[key]
            return True
        return False


class A2AServer:
    """
    Simple A2A server for handling incoming requests.

    Used by agents to receive and process A2A messages.
    """

    def __init__(self, agent_name: str, port: int = 8080):
        self.agent_name = agent_name
        self.port = port
        self._handlers: dict[str, Callable] = {}

    def capability(self, capability_id: str):
        """Decorator to register a capability handler."""
        def decorator(func: Callable[[A2AMessage], A2AResponse]):
            self._handlers[capability_id] = func
            return func
        return decorator

    async def handle_request(self, message: A2AMessage) -> A2AResponse:
        """Handle an incoming A2A request."""
        capability = message.callee.capability

        if capability not in self._handlers:
            return A2AResponse.error(
                request_id=message.request_id,
                trace=message.trace,
                code="UNKNOWN_CAPABILITY",
                message=f"Capability {capability} not found on {self.agent_name}",
            )

        try:
            handler = self._handlers[capability]
            return await handler(message)
        except Exception as e:
            logger.exception(f"Handler error for {capability}")
            return A2AResponse.error(
                request_id=message.request_id,
                trace=message.trace,
                code="HANDLER_ERROR",
                message=str(e),
            )

    async def start(self):
        """Start the A2A server."""
        try:
            from aiohttp import web

            app = web.Application()
            app.router.add_post(
                f"/a2a/{self.agent_name}/{{capability}}",
                self._http_handler,
            )

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.port)
            await site.start()

            logger.info(f"A2A server started for {self.agent_name} on port {self.port}")

        except ImportError:
            logger.error("aiohttp required for HTTP server")
            raise

    async def _http_handler(self, request):
        """HTTP request handler."""
        from aiohttp import web

        try:
            data = await request.json()
            message = A2AMessage.from_dict(data)
            response = await self.handle_request(message)
            return web.json_response(response.to_dict())
        except Exception as e:
            return web.json_response(
                {"error": str(e)},
                status=500,
            )
