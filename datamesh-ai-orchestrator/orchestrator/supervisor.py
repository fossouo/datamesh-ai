"""
DATAMESH.AI Orchestrator - Agent Supervisor
============================================

This module provides agent supervision and health monitoring. It:
- Periodically checks agent health
- Tracks agent status (healthy, unhealthy, unknown)
- Triggers alerts on health state changes
- Provides health status for routing decisions
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import httpx

from orchestrator.validator import AgentInfo

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status for agents."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    agent_name: str
    status: HealthStatus
    response_time_ms: float
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class AgentHealthState:
    """Tracks health state for an agent."""
    agent_name: str
    current_status: HealthStatus = HealthStatus.UNKNOWN
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_time: float = 0.0
    last_healthy_time: float = 0.0
    last_error: Optional[str] = None
    average_response_time_ms: float = 0.0
    check_count: int = 0
    failure_count: int = 0


@dataclass
class SupervisorConfig:
    """Configuration for the agent supervisor."""
    # Health check interval in milliseconds
    health_check_interval_ms: int = 30000

    # Timeout for health checks in milliseconds
    health_check_timeout_ms: int = 5000

    # Number of consecutive failures before marking unhealthy
    unhealthy_threshold: int = 3

    # Number of consecutive successes to mark healthy again
    healthy_threshold: int = 2

    # Whether to automatically restart unhealthy agents (if supported)
    auto_restart: bool = False

    # Whether to alert on health state changes
    alert_on_health_change: bool = True

    # Optional webhook URL for alerts
    alert_webhook: Optional[str] = None

    # Maximum history entries to keep per agent
    max_history_entries: int = 100


class AgentSupervisor:
    """
    Supervises agent health and provides status monitoring.

    This supervisor:
    - Periodically checks agent health endpoints
    - Tracks health state transitions
    - Triggers alerts when health changes
    - Provides agent status for routing decisions
    """

    def __init__(
        self,
        agents: dict[str, AgentInfo],
        config: SupervisorConfig,
        on_health_change: Callable[[str, HealthStatus, HealthStatus], None] | None = None,
        on_alert: Callable[[dict[str, Any]], None] | None = None
    ):
        """
        Initialize the agent supervisor.

        Args:
            agents: Dictionary mapping agent names to AgentInfo
            config: Supervisor configuration
            on_health_change: Callback for health state changes
            on_alert: Callback for sending alerts
        """
        self.agents = agents
        self.config = config
        self.on_health_change = on_health_change
        self.on_alert = on_alert

        # Health state per agent
        self._health_states: dict[str, AgentHealthState] = {}
        for name in agents:
            self._health_states[name] = AgentHealthState(agent_name=name)

        # Health check history per agent
        self._history: dict[str, list[HealthCheckResult]] = {
            name: [] for name in agents
        }

        # HTTP client for health checks
        self._client: Optional[httpx.AsyncClient] = None

        # Background task handle
        self._check_task: Optional[asyncio.Task] = None

        # Running flag
        self._running = False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.health_check_timeout_ms / 1000,
                    read=self.config.health_check_timeout_ms / 1000,
                    write=10.0,
                    pool=5.0
                )
            )
        return self._client

    async def start(self) -> None:
        """Start the supervisor background tasks."""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Agent supervisor started")

    async def stop(self) -> None:
        """Stop the supervisor and cleanup resources."""
        self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None

        if self._client:
            await self._client.aclose()
            self._client = None

        logger.info("Agent supervisor stopped")

    async def _health_check_loop(self) -> None:
        """Background loop for periodic health checks."""
        while self._running:
            try:
                await self.check_all_agents()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

            await asyncio.sleep(self.config.health_check_interval_ms / 1000)

    async def check_all_agents(self) -> dict[str, HealthCheckResult]:
        """
        Check health of all registered agents.

        Returns:
            Dictionary mapping agent names to health check results
        """
        tasks = [
            self.check_agent(name)
            for name, agent in self.agents.items()
            if agent.enabled
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            result.agent_name: result
            for result in results
            if isinstance(result, HealthCheckResult)
        }

    async def check_agent(self, agent_name: str) -> HealthCheckResult:
        """
        Check health of a specific agent.

        Args:
            agent_name: Name of the agent to check

        Returns:
            HealthCheckResult with status and details
        """
        if agent_name not in self.agents:
            return HealthCheckResult(
                agent_name=agent_name,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0,
                timestamp=time.time(),
                error=f"Unknown agent: {agent_name}"
            )

        agent = self.agents[agent_name]
        start_time = time.time()

        try:
            client = await self._get_client()
            health_url = f"{agent.endpoint}{agent.health_endpoint}"

            response = await client.get(health_url)
            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                try:
                    details = response.json()
                except Exception:
                    details = {"raw": response.text}

                result = HealthCheckResult(
                    agent_name=agent_name,
                    status=HealthStatus.HEALTHY,
                    response_time_ms=response_time,
                    timestamp=time.time(),
                    details=details
                )
            elif response.status_code == 503:
                # Service degraded
                result = HealthCheckResult(
                    agent_name=agent_name,
                    status=HealthStatus.DEGRADED,
                    response_time_ms=response_time,
                    timestamp=time.time(),
                    error=f"Service degraded: {response.status_code}"
                )
            else:
                result = HealthCheckResult(
                    agent_name=agent_name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=response_time,
                    timestamp=time.time(),
                    error=f"Health check failed with status: {response.status_code}"
                )

        except httpx.TimeoutException:
            result = HealthCheckResult(
                agent_name=agent_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                error="Health check timed out"
            )

        except httpx.ConnectError as e:
            result = HealthCheckResult(
                agent_name=agent_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                error=f"Connection failed: {str(e)}"
            )

        except Exception as e:
            result = HealthCheckResult(
                agent_name=agent_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
                error=f"Health check error: {str(e)}"
            )

        # Update state and history
        self._update_health_state(result)
        self._add_to_history(result)

        return result

    def _update_health_state(self, result: HealthCheckResult) -> None:
        """Update agent health state based on check result."""
        state = self._health_states[result.agent_name]
        previous_status = state.current_status

        state.last_check_time = result.timestamp
        state.check_count += 1

        # Update response time average
        if result.status == HealthStatus.HEALTHY:
            if state.average_response_time_ms == 0:
                state.average_response_time_ms = result.response_time_ms
            else:
                # Exponential moving average
                alpha = 0.3
                state.average_response_time_ms = (
                    alpha * result.response_time_ms +
                    (1 - alpha) * state.average_response_time_ms
                )

        if result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED):
            state.consecutive_successes += 1
            state.consecutive_failures = 0
            state.last_healthy_time = result.timestamp

            if state.consecutive_successes >= self.config.healthy_threshold:
                if result.status == HealthStatus.DEGRADED:
                    state.current_status = HealthStatus.DEGRADED
                else:
                    state.current_status = HealthStatus.HEALTHY

        else:  # UNHEALTHY or UNKNOWN
            state.consecutive_failures += 1
            state.consecutive_successes = 0
            state.failure_count += 1
            state.last_error = result.error

            if state.consecutive_failures >= self.config.unhealthy_threshold:
                state.current_status = HealthStatus.UNHEALTHY

        # Check for state change
        if previous_status != state.current_status:
            self._handle_health_change(
                result.agent_name,
                previous_status,
                state.current_status
            )

    def _handle_health_change(
        self,
        agent_name: str,
        old_status: HealthStatus,
        new_status: HealthStatus
    ) -> None:
        """Handle health state change."""
        logger.warning(
            f"Agent '{agent_name}' health changed: {old_status.value} -> {new_status.value}"
        )

        # Call callback
        if self.on_health_change:
            try:
                self.on_health_change(agent_name, old_status, new_status)
            except Exception as e:
                logger.error(f"Error in health change callback: {e}")

        # Send alert
        if self.config.alert_on_health_change:
            self._send_alert({
                "type": "health_change",
                "agent": agent_name,
                "previous_status": old_status.value,
                "current_status": new_status.value,
                "timestamp": time.time()
            })

    def _send_alert(self, alert_data: dict[str, Any]) -> None:
        """Send an alert."""
        if self.on_alert:
            try:
                self.on_alert(alert_data)
            except Exception as e:
                logger.error(f"Error sending alert: {e}")

        if self.config.alert_webhook:
            # Fire and forget webhook call
            asyncio.create_task(self._send_webhook_alert(alert_data))

    async def _send_webhook_alert(self, alert_data: dict[str, Any]) -> None:
        """Send alert via webhook."""
        if not self.config.alert_webhook:
            return

        try:
            client = await self._get_client()
            await client.post(
                self.config.alert_webhook,
                json=alert_data,
                timeout=5.0
            )
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def _add_to_history(self, result: HealthCheckResult) -> None:
        """Add health check result to history."""
        history = self._history[result.agent_name]
        history.append(result)

        # Trim history if needed
        if len(history) > self.config.max_history_entries:
            self._history[result.agent_name] = history[-self.config.max_history_entries:]

    def get_agent_status(self, agent_name: str) -> HealthStatus:
        """
        Get current health status for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Current HealthStatus
        """
        if agent_name not in self._health_states:
            return HealthStatus.UNKNOWN
        return self._health_states[agent_name].current_status

    def get_agent_health_state(self, agent_name: str) -> AgentHealthState | None:
        """
        Get full health state for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentHealthState or None if agent not found
        """
        return self._health_states.get(agent_name)

    def get_all_health_states(self) -> dict[str, AgentHealthState]:
        """
        Get health states for all agents.

        Returns:
            Dictionary mapping agent names to health states
        """
        return self._health_states.copy()

    def get_health_history(
        self,
        agent_name: str,
        limit: int | None = None
    ) -> list[HealthCheckResult]:
        """
        Get health check history for an agent.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of entries to return

        Returns:
            List of HealthCheckResult entries
        """
        history = self._history.get(agent_name, [])
        if limit:
            return history[-limit:]
        return history.copy()

    def is_agent_available(self, agent_name: str) -> bool:
        """
        Check if an agent is available for routing.

        Args:
            agent_name: Name of the agent

        Returns:
            True if agent is healthy or degraded
        """
        status = self.get_agent_status(agent_name)
        return status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNKNOWN)

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of all agent health states.

        Returns:
            Dictionary with health summary
        """
        summary = {
            "total_agents": len(self.agents),
            "healthy": 0,
            "unhealthy": 0,
            "degraded": 0,
            "unknown": 0,
            "agents": {}
        }

        for name, state in self._health_states.items():
            agent = self.agents.get(name)
            if agent and not agent.enabled:
                continue

            status_key = state.current_status.value
            summary[status_key] = summary.get(status_key, 0) + 1

            summary["agents"][name] = {
                "status": state.current_status.value,
                "last_check": state.last_check_time,
                "consecutive_failures": state.consecutive_failures,
                "average_response_time_ms": round(state.average_response_time_ms, 2),
                "last_error": state.last_error
            }

        return summary
