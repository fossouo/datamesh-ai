"""
Base Agent class with lifecycle management.

This module provides the foundational Agent class that all DATAMESH.AI
agents should inherit from.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

from datamesh_ai_core.contracts import AgentContract, ContractLoader
from datamesh_ai_core.models import (
    A2ARequest,
    A2AResponse,
    A2AStatus,
    AgentCapability,
    AgentMetadata,
    TraceContext,
)
from datamesh_ai_core.tracing import TracingManager, create_trace_context


class AgentState(str, Enum):
    """Agent lifecycle states."""

    CREATED = "CREATED"
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class AgentError(Exception):
    """Base exception for agent errors."""

    def __init__(self, message: str, code: str = "AGENT_ERROR"):
        """
        Initialize agent error.

        Args:
            message: Error message
            code: Error code
        """
        super().__init__(message)
        self.code = code


class CapabilityNotFoundError(AgentError):
    """Raised when a requested capability is not found."""

    def __init__(self, capability: str):
        super().__init__(
            f"Capability not found: {capability}",
            code="CAPABILITY_NOT_FOUND",
        )


class AgentStateError(AgentError):
    """Raised when an operation is invalid for the current state."""

    def __init__(self, current_state: AgentState, expected_states: List[AgentState]):
        states = ", ".join(s.value for s in expected_states)
        super().__init__(
            f"Invalid state: {current_state.value}. Expected one of: {states}",
            code="INVALID_STATE",
        )


class Agent(ABC):
    """
    Base class for all DATAMESH.AI agents.

    This class provides:
    - Lifecycle management (init, start, stop)
    - Contract loading from agent.yaml
    - Capability registration and invocation
    - Request handling with tracing
    - Health check support

    Subclasses should implement:
    - _initialize(): Agent-specific initialization
    - _shutdown(): Agent-specific cleanup
    - Capability handlers via @capability decorator
    """

    def __init__(
        self,
        contract: Optional[AgentContract] = None,
        contract_path: Optional[Union[str, Path]] = None,
        tracing_manager: Optional[TracingManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the agent.

        Args:
            contract: Pre-loaded AgentContract
            contract_path: Path to agent.yaml file
            tracing_manager: Optional tracing manager
            logger: Optional logger instance
        """
        # Load contract
        if contract:
            self._contract = contract
        elif contract_path:
            loader = ContractLoader()
            self._contract = loader.load_from_file(contract_path)
        else:
            raise ValueError("Either contract or contract_path must be provided")

        # Initialize state
        self._state = AgentState.CREATED
        self._state_changed_at = datetime.utcnow()

        # Setup logging
        self._logger = logger or logging.getLogger(
            f"datamesh.agent.{self._contract.agent_id}"
        )

        # Setup tracing
        self._tracing = tracing_manager or TracingManager()

        # Capability handlers registry
        self._capability_handlers: Dict[str, Callable] = {}

        # Register capabilities from subclass
        self._register_capabilities()

    @property
    def agent_id(self) -> str:
        """Get the agent ID."""
        return self._contract.agent_id

    @property
    def name(self) -> str:
        """Get the agent name."""
        return self._contract.name

    @property
    def version(self) -> str:
        """Get the agent version."""
        return self._contract.version

    @property
    def state(self) -> AgentState:
        """Get the current agent state."""
        return self._state

    @property
    def contract(self) -> AgentContract:
        """Get the agent contract."""
        return self._contract

    @property
    def metadata(self) -> AgentMetadata:
        """Get agent metadata for registry."""
        return self._contract.to_metadata()

    @property
    def capabilities(self) -> List[AgentCapability]:
        """Get list of agent capabilities."""
        return self._contract.capabilities

    def _set_state(self, new_state: AgentState) -> None:
        """
        Update agent state with logging.

        Args:
            new_state: New state to set
        """
        old_state = self._state
        self._state = new_state
        self._state_changed_at = datetime.utcnow()
        self._logger.info(f"State changed: {old_state.value} -> {new_state.value}")

    def _ensure_state(self, *expected_states: AgentState) -> None:
        """
        Ensure agent is in one of the expected states.

        Args:
            expected_states: Valid states for the operation

        Raises:
            AgentStateError: If not in expected state
        """
        if self._state not in expected_states:
            raise AgentStateError(self._state, list(expected_states))

    def _register_capabilities(self) -> None:
        """
        Register capability handlers from subclass.

        This method looks for methods decorated with @capability
        and registers them as handlers.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if callable(attr) and hasattr(attr, "_capability_name"):
                capability_name = attr._capability_name
                self._capability_handlers[capability_name] = attr
                self._logger.debug(f"Registered capability: {capability_name}")

    def register_capability_handler(
        self,
        capability_name: str,
        handler: Callable,
    ) -> None:
        """
        Register a capability handler.

        Args:
            capability_name: Name of the capability
            handler: Handler function
        """
        self._capability_handlers[capability_name] = handler
        self._logger.debug(f"Registered capability handler: {capability_name}")

    async def initialize(self) -> None:
        """
        Initialize the agent.

        This starts the agent lifecycle, calling _initialize() for
        subclass-specific setup.
        """
        self._ensure_state(AgentState.CREATED)
        self._set_state(AgentState.INITIALIZING)

        try:
            await self._initialize()
            self._set_state(AgentState.READY)
            self._logger.info(f"Agent {self.agent_id} initialized successfully")
        except Exception as e:
            self._set_state(AgentState.ERROR)
            self._logger.error(f"Initialization failed: {e}")
            raise

    async def start(self) -> None:
        """
        Start the agent.

        Makes the agent ready to process requests.
        """
        self._ensure_state(AgentState.READY, AgentState.PAUSED)
        self._set_state(AgentState.RUNNING)
        self._logger.info(f"Agent {self.agent_id} started")

    async def pause(self) -> None:
        """
        Pause the agent.

        Temporarily stops processing new requests.
        """
        self._ensure_state(AgentState.RUNNING)
        self._set_state(AgentState.PAUSED)
        self._logger.info(f"Agent {self.agent_id} paused")

    async def stop(self) -> None:
        """
        Stop the agent.

        Performs graceful shutdown, calling _shutdown() for
        subclass-specific cleanup.
        """
        self._ensure_state(
            AgentState.RUNNING,
            AgentState.PAUSED,
            AgentState.READY,
            AgentState.ERROR,
        )
        self._set_state(AgentState.STOPPING)

        try:
            await self._shutdown()
            self._set_state(AgentState.STOPPED)
            self._logger.info(f"Agent {self.agent_id} stopped")
        except Exception as e:
            self._set_state(AgentState.ERROR)
            self._logger.error(f"Shutdown failed: {e}")
            raise

    @asynccontextmanager
    async def lifespan(self):
        """
        Context manager for agent lifecycle.

        Usage:
            async with agent.lifespan():
                # Agent is running
                ...
        """
        await self.initialize()
        await self.start()
        try:
            yield self
        finally:
            await self.stop()

    async def handle_request(self, request: A2ARequest) -> A2AResponse:
        """
        Handle an incoming A2A request.

        Args:
            request: The A2A request to handle

        Returns:
            A2A response
        """
        start_time = datetime.utcnow()

        # Validate state
        if self._state != AgentState.RUNNING:
            return A2AResponse.error(
                request,
                error_code="AGENT_NOT_RUNNING",
                error_message=f"Agent is not running (state: {self._state.value})",
            )

        # Create/propagate trace context
        trace_context = request.trace_context or create_trace_context()

        # Start tracing span
        with self._tracing.start_span(
            f"handle_{request.capability}",
            trace_context=trace_context,
        ) as span:
            try:
                # Find capability handler
                handler = self._capability_handlers.get(request.capability)
                if not handler:
                    return A2AResponse.error(
                        request,
                        error_code="CAPABILITY_NOT_FOUND",
                        error_message=f"Unknown capability: {request.capability}",
                    )

                # Execute handler
                result = await self._execute_handler(handler, request)

                # Calculate duration
                duration_ms = int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )

                return A2AResponse.success(
                    request,
                    payload=result,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                self._logger.exception(f"Error handling request: {e}")
                duration_ms = int(
                    (datetime.utcnow() - start_time).total_seconds() * 1000
                )
                return A2AResponse.error(
                    request,
                    error_code="HANDLER_ERROR",
                    error_message=str(e),
                    duration_ms=duration_ms,
                )

    async def _execute_handler(
        self,
        handler: Callable,
        request: A2ARequest,
    ) -> Dict[str, Any]:
        """
        Execute a capability handler.

        Args:
            handler: The handler function
            request: The A2A request

        Returns:
            Handler result
        """
        if asyncio.iscoroutinefunction(handler):
            return await handler(request)
        else:
            return handler(request)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health check result with status and details
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "state": self._state.value,
            "state_changed_at": self._state_changed_at.isoformat(),
            "healthy": self._state == AgentState.RUNNING,
            "capabilities": [c.name for c in self.capabilities],
        }

    @abstractmethod
    async def _initialize(self) -> None:
        """
        Subclass-specific initialization.

        Override this method to perform agent-specific setup
        such as connecting to databases, loading models, etc.
        """
        pass

    @abstractmethod
    async def _shutdown(self) -> None:
        """
        Subclass-specific shutdown.

        Override this method to perform agent-specific cleanup
        such as closing connections, releasing resources, etc.
        """
        pass


def capability(name: str):
    """
    Decorator to mark a method as a capability handler.

    Usage:
        class MyAgent(Agent):
            @capability("my-capability")
            async def handle_my_capability(self, request: A2ARequest):
                return {"result": "success"}

    Args:
        name: The capability name

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        func._capability_name = name
        return func
    return decorator
