"""
A2A (Agent-to-Agent) protocol implementation.

This module provides request/response handling for the A2A protocol,
including client and handler classes for agent communication.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Type
from urllib.parse import urljoin

from datamesh_ai_core.models import (
    A2ARequest,
    A2AResponse,
    A2AStatus,
    TraceContext,
)
from datamesh_ai_core.tracing import create_trace_context


class A2AError(Exception):
    """Base exception for A2A protocol errors."""

    def __init__(self, message: str, code: str = "A2A_ERROR"):
        """
        Initialize A2A error.

        Args:
            message: Error message
            code: Error code
        """
        super().__init__(message)
        self.code = code


class A2ATimeoutError(A2AError):
    """Raised when A2A request times out."""

    def __init__(self, request_id: str, timeout_ms: int):
        super().__init__(
            f"Request {request_id} timed out after {timeout_ms}ms",
            code="TIMEOUT",
        )
        self.request_id = request_id
        self.timeout_ms = timeout_ms


class A2AConnectionError(A2AError):
    """Raised when connection to agent fails."""

    def __init__(self, endpoint: str, reason: str):
        super().__init__(
            f"Failed to connect to {endpoint}: {reason}",
            code="CONNECTION_ERROR",
        )
        self.endpoint = endpoint


class A2ATransport(ABC):
    """
    Abstract base class for A2A transport implementations.

    Subclasses implement actual transport mechanisms (HTTP, gRPC, etc.)
    """

    @abstractmethod
    async def send(
        self,
        endpoint: str,
        request: A2ARequest,
        timeout_ms: Optional[int] = None,
    ) -> A2AResponse:
        """
        Send an A2A request.

        Args:
            endpoint: Target agent endpoint
            request: A2A request to send
            timeout_ms: Request timeout in milliseconds

        Returns:
            A2A response

        Raises:
            A2AError: On transport errors
        """
        pass


class InMemoryTransport(A2ATransport):
    """
    In-memory transport for testing and local agent communication.

    This transport routes requests directly to registered handlers
    without network overhead.
    """

    def __init__(self):
        """Initialize the in-memory transport."""
        self._handlers: Dict[str, Callable] = {}
        self._logger = logging.getLogger("datamesh.a2a.transport.memory")

    def register_handler(
        self,
        agent_id: str,
        handler: Callable[[A2ARequest], A2AResponse],
    ) -> None:
        """
        Register a handler for an agent.

        Args:
            agent_id: Agent identifier
            handler: Request handler function
        """
        self._handlers[agent_id] = handler
        self._logger.debug(f"Registered handler for: {agent_id}")

    def unregister_handler(self, agent_id: str) -> None:
        """
        Unregister a handler.

        Args:
            agent_id: Agent identifier
        """
        self._handlers.pop(agent_id, None)
        self._logger.debug(f"Unregistered handler for: {agent_id}")

    async def send(
        self,
        endpoint: str,
        request: A2ARequest,
        timeout_ms: Optional[int] = None,
    ) -> A2AResponse:
        """
        Send request to registered handler.

        Args:
            endpoint: Target agent (used as agent_id lookup)
            request: A2A request
            timeout_ms: Timeout in milliseconds

        Returns:
            A2A response
        """
        target = request.target_agent
        handler = self._handlers.get(target)

        if not handler:
            return A2AResponse(
                request_id=request.request_id,
                source_agent=target,
                target_agent=request.source_agent,
                status=A2AStatus.ERROR,
                error={
                    "code": "AGENT_NOT_FOUND",
                    "message": f"No handler registered for agent: {target}",
                },
            )

        try:
            if timeout_ms:
                result = await asyncio.wait_for(
                    self._execute_handler(handler, request),
                    timeout=timeout_ms / 1000,
                )
            else:
                result = await self._execute_handler(handler, request)
            return result
        except asyncio.TimeoutError:
            raise A2ATimeoutError(request.request_id, timeout_ms or 0)
        except Exception as e:
            self._logger.exception(f"Handler error: {e}")
            return A2AResponse(
                request_id=request.request_id,
                source_agent=target,
                target_agent=request.source_agent,
                status=A2AStatus.ERROR,
                error={
                    "code": "HANDLER_ERROR",
                    "message": str(e),
                },
            )

    async def _execute_handler(
        self,
        handler: Callable,
        request: A2ARequest,
    ) -> A2AResponse:
        """Execute handler (sync or async)."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(request)
        else:
            return handler(request)


class A2AClient:
    """
    Client for making A2A requests to other agents.

    This class provides a high-level interface for agent-to-agent
    communication, handling request creation, sending, and response
    processing.
    """

    def __init__(
        self,
        source_agent: str,
        transport: Optional[A2ATransport] = None,
        default_timeout_ms: int = 30000,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the A2A client.

        Args:
            source_agent: ID of the source agent
            transport: Transport implementation
            default_timeout_ms: Default timeout in milliseconds
            logger: Optional logger instance
        """
        self._source_agent = source_agent
        self._transport = transport or InMemoryTransport()
        self._default_timeout_ms = default_timeout_ms
        self._logger = logger or logging.getLogger(
            f"datamesh.a2a.client.{source_agent}"
        )

    @property
    def source_agent(self) -> str:
        """Get the source agent ID."""
        return self._source_agent

    def create_request(
        self,
        target_agent: str,
        capability: str,
        payload: Optional[Dict[str, Any]] = None,
        deadline_ms: Optional[int] = None,
        trace_context: Optional[TraceContext] = None,
        correlation_id: Optional[str] = None,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> A2ARequest:
        """
        Create an A2A request.

        Args:
            target_agent: Target agent ID
            capability: Capability to invoke
            payload: Request payload
            deadline_ms: Request deadline
            trace_context: Trace context
            correlation_id: Correlation ID
            priority: Request priority
            metadata: Additional metadata

        Returns:
            A2A request object
        """
        return A2ARequest(
            source_agent=self._source_agent,
            target_agent=target_agent,
            capability=capability,
            payload=payload or {},
            deadline_ms=deadline_ms or self._default_timeout_ms,
            trace_context=trace_context or create_trace_context(),
            correlation_id=correlation_id,
            priority=priority,
            metadata=metadata or {},
        )

    async def send(
        self,
        request: A2ARequest,
        endpoint: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> A2AResponse:
        """
        Send an A2A request.

        Args:
            request: The request to send
            endpoint: Target endpoint (optional)
            timeout_ms: Timeout override

        Returns:
            A2A response
        """
        self._logger.debug(
            f"Sending request {request.request_id} to {request.target_agent}"
        )

        response = await self._transport.send(
            endpoint=endpoint or request.target_agent,
            request=request,
            timeout_ms=timeout_ms or request.deadline_ms,
        )

        self._logger.debug(
            f"Received response {request.request_id}: {response.status}"
        )

        return response

    async def invoke(
        self,
        target_agent: str,
        capability: str,
        payload: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        trace_context: Optional[TraceContext] = None,
    ) -> A2AResponse:
        """
        Create and send an A2A request in one call.

        Args:
            target_agent: Target agent ID
            capability: Capability to invoke
            payload: Request payload
            endpoint: Target endpoint
            timeout_ms: Request timeout
            trace_context: Trace context

        Returns:
            A2A response
        """
        request = self.create_request(
            target_agent=target_agent,
            capability=capability,
            payload=payload,
            deadline_ms=timeout_ms,
            trace_context=trace_context,
        )

        return await self.send(
            request=request,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
        )


class A2AHandler:
    """
    Handler for processing incoming A2A requests.

    This class provides middleware-style request processing with
    support for pre/post processing hooks.
    """

    def __init__(
        self,
        agent_id: str,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the A2A handler.

        Args:
            agent_id: ID of the handling agent
            logger: Optional logger instance
        """
        self._agent_id = agent_id
        self._logger = logger or logging.getLogger(
            f"datamesh.a2a.handler.{agent_id}"
        )

        # Capability handlers
        self._handlers: Dict[str, Callable] = {}

        # Middleware
        self._pre_processors: list = []
        self._post_processors: list = []

    @property
    def agent_id(self) -> str:
        """Get the agent ID."""
        return self._agent_id

    def register(
        self,
        capability: str,
        handler: Callable[[A2ARequest], A2AResponse],
    ) -> None:
        """
        Register a capability handler.

        Args:
            capability: Capability name
            handler: Handler function
        """
        self._handlers[capability] = handler
        self._logger.debug(f"Registered handler for: {capability}")

    def unregister(self, capability: str) -> None:
        """
        Unregister a capability handler.

        Args:
            capability: Capability name
        """
        self._handlers.pop(capability, None)
        self._logger.debug(f"Unregistered handler for: {capability}")

    def add_pre_processor(
        self,
        processor: Callable[[A2ARequest], Optional[A2ARequest]],
    ) -> None:
        """
        Add a pre-processor middleware.

        Args:
            processor: Pre-processor function
        """
        self._pre_processors.append(processor)

    def add_post_processor(
        self,
        processor: Callable[[A2AResponse], A2AResponse],
    ) -> None:
        """
        Add a post-processor middleware.

        Args:
            processor: Post-processor function
        """
        self._post_processors.append(processor)

    async def handle(self, request: A2ARequest) -> A2AResponse:
        """
        Handle an incoming A2A request.

        Args:
            request: The incoming request

        Returns:
            A2A response
        """
        start_time = datetime.utcnow()

        self._logger.debug(
            f"Handling request {request.request_id} "
            f"for capability: {request.capability}"
        )

        try:
            # Run pre-processors
            processed_request = request
            for processor in self._pre_processors:
                result = await self._execute(processor, processed_request)
                if result is None:
                    # Pre-processor rejected the request
                    return A2AResponse.error(
                        request,
                        error_code="REQUEST_REJECTED",
                        error_message="Request rejected by pre-processor",
                    )
                processed_request = result

            # Find handler
            handler = self._handlers.get(processed_request.capability)
            if not handler:
                return A2AResponse.error(
                    request,
                    error_code="CAPABILITY_NOT_FOUND",
                    error_message=f"Unknown capability: {processed_request.capability}",
                )

            # Execute handler
            response = await self._execute(handler, processed_request)

            # Run post-processors
            for processor in self._post_processors:
                response = await self._execute(processor, response)

            # Set duration
            duration_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            response.duration_ms = duration_ms

            return response

        except Exception as e:
            self._logger.exception(f"Handler error: {e}")
            duration_ms = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            return A2AResponse.error(
                request,
                error_code="HANDLER_ERROR",
                error_message=str(e),
                duration_ms=duration_ms,
            )

    async def _execute(self, func: Callable, *args) -> Any:
        """Execute a function (sync or async)."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args)
        else:
            return func(*args)

    def capability(self, name: str):
        """
        Decorator to register a capability handler.

        Usage:
            handler = A2AHandler("my-agent")

            @handler.capability("my-capability")
            async def handle_capability(request: A2ARequest) -> A2AResponse:
                return A2AResponse.success(request, {"result": "ok"})

        Args:
            name: Capability name

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            self.register(name, func)
            return func
        return decorator
