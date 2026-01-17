"""
Agent Lifecycle Manager - Manages agent state and lifecycle events.

Provides state machine for agent lifecycle and hooks for
initialization, health checks, and graceful shutdown.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from ..registry import AgentContract

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent lifecycle states."""
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class AgentHealth:
    """Agent health status."""
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"


@dataclass
class LifecycleEvent:
    """Lifecycle event record."""
    state: AgentState
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)


class AgentLifecycle:
    """
    Manages the lifecycle of a DATAMESH.AI agent.

    Provides state transitions, health checks, and lifecycle hooks.
    """

    def __init__(self, contract: AgentContract):
        self.contract = contract
        self.agent_name = contract.metadata.name
        self._state = AgentState.CREATED
        self._health = AgentHealth(
            status="unknown",
            last_check=datetime.utcnow(),
        )
        self._events: list[LifecycleEvent] = []
        self._hooks: dict[str, list[Callable]] = {
            "on_init": [],
            "on_ready": [],
            "on_start": [],
            "on_stop": [],
            "on_error": [],
            "on_health_check": [],
        }
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None

        # Record creation event
        self._record_event(AgentState.CREATED)

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def health(self) -> AgentHealth:
        return self._health

    @property
    def is_ready(self) -> bool:
        return self._state in (AgentState.READY, AgentState.RUNNING)

    def on(self, event: str, callback: Callable) -> None:
        """Register a lifecycle hook."""
        if event in self._hooks:
            self._hooks[event].append(callback)
        else:
            raise ValueError(f"Unknown lifecycle event: {event}")

    async def initialize(self) -> bool:
        """Initialize the agent."""
        if self._state != AgentState.CREATED:
            logger.warning(f"Cannot initialize agent in state {self._state}")
            return False

        self._transition_to(AgentState.INITIALIZING)

        try:
            # Run initialization hooks
            await self._run_hooks("on_init")

            self._transition_to(AgentState.READY)
            await self._run_hooks("on_ready")

            logger.info(f"Agent {self.agent_name} initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Agent {self.agent_name} initialization failed: {e}")
            self._transition_to(AgentState.ERROR, {"error": str(e)})
            await self._run_hooks("on_error", error=e)
            return False

    async def start(self) -> bool:
        """Start the agent."""
        if self._state != AgentState.READY:
            logger.warning(f"Cannot start agent in state {self._state}")
            return False

        self._running = True
        self._transition_to(AgentState.RUNNING)

        # Start health check loop
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        await self._run_hooks("on_start")

        logger.info(f"Agent {self.agent_name} started")
        return True

    async def stop(self) -> bool:
        """Stop the agent gracefully."""
        if self._state not in (AgentState.RUNNING, AgentState.PAUSED, AgentState.READY):
            logger.warning(f"Cannot stop agent in state {self._state}")
            return False

        self._transition_to(AgentState.STOPPING)
        self._running = False

        # Stop health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        await self._run_hooks("on_stop")

        self._transition_to(AgentState.STOPPED)
        logger.info(f"Agent {self.agent_name} stopped")
        return True

    async def pause(self) -> bool:
        """Pause the agent."""
        if self._state != AgentState.RUNNING:
            return False

        self._transition_to(AgentState.PAUSED)
        logger.info(f"Agent {self.agent_name} paused")
        return True

    async def resume(self) -> bool:
        """Resume a paused agent."""
        if self._state != AgentState.PAUSED:
            return False

        self._transition_to(AgentState.RUNNING)
        logger.info(f"Agent {self.agent_name} resumed")
        return True

    async def check_health(self) -> AgentHealth:
        """Perform health check."""
        try:
            details = {}
            errors = []

            # Run health check hooks
            for callback in self._hooks["on_health_check"]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        result = await callback()
                    else:
                        result = callback()

                    if isinstance(result, dict):
                        details.update(result)
                except Exception as e:
                    errors.append(str(e))

            # Determine overall status
            if errors:
                status = "unhealthy" if len(errors) > 1 else "degraded"
            else:
                status = "healthy"

            self._health = AgentHealth(
                status=status,
                last_check=datetime.utcnow(),
                details=details,
                errors=errors,
            )

        except Exception as e:
            self._health = AgentHealth(
                status="unhealthy",
                last_check=datetime.utcnow(),
                errors=[str(e)],
            )

        return self._health

    def get_events(self) -> list[LifecycleEvent]:
        """Get lifecycle event history."""
        return self._events.copy()

    def get_status(self) -> dict:
        """Get current agent status."""
        return {
            "agent": self.agent_name,
            "state": self._state.value,
            "health": {
                "status": self._health.status,
                "lastCheck": self._health.last_check.isoformat(),
                "details": self._health.details,
                "errors": self._health.errors,
            },
            "uptime": self._calculate_uptime(),
            "version": self.contract.metadata.version,
        }

    def _transition_to(self, state: AgentState, details: Optional[dict] = None) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = state
        self._record_event(state, details)
        logger.debug(f"Agent {self.agent_name}: {old_state.value} -> {state.value}")

    def _record_event(self, state: AgentState, details: Optional[dict] = None) -> None:
        """Record a lifecycle event."""
        self._events.append(LifecycleEvent(
            state=state,
            timestamp=datetime.utcnow(),
            details=details or {},
        ))

    async def _run_hooks(self, event: str, **kwargs) -> None:
        """Run all hooks for an event."""
        for callback in self._hooks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(**kwargs)
                else:
                    callback(**kwargs)
            except Exception as e:
                logger.error(f"Hook {event} error: {e}")

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self.check_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    def _calculate_uptime(self) -> Optional[float]:
        """Calculate uptime in seconds."""
        start_event = None
        for event in self._events:
            if event.state == AgentState.RUNNING:
                start_event = event
                break

        if start_event and self._state == AgentState.RUNNING:
            delta = datetime.utcnow() - start_event.timestamp
            return delta.total_seconds()

        return None
