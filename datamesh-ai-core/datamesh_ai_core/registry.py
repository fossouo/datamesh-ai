"""
Agent registry for discovery and lookup.

This module provides the AgentRegistry class for registering,
discovering, and managing agents in the DATAMESH.AI ecosystem.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from datamesh_ai_core.models import AgentCapability, AgentMetadata


class AgentRegistryError(Exception):
    """Base exception for registry errors."""
    pass


class AgentNotFoundError(AgentRegistryError):
    """Raised when an agent is not found in the registry."""

    def __init__(self, agent_id: str):
        super().__init__(f"Agent not found: {agent_id}")
        self.agent_id = agent_id


class AgentAlreadyExistsError(AgentRegistryError):
    """Raised when trying to register an agent that already exists."""

    def __init__(self, agent_id: str):
        super().__init__(f"Agent already exists: {agent_id}")
        self.agent_id = agent_id


class RegistryEntry:
    """
    Entry in the agent registry.

    Contains agent metadata and registration information.
    """

    def __init__(
        self,
        metadata: AgentMetadata,
        registered_at: Optional[datetime] = None,
        last_heartbeat: Optional[datetime] = None,
        ttl_seconds: int = 300,
    ):
        """
        Initialize registry entry.

        Args:
            metadata: Agent metadata
            registered_at: Registration timestamp
            last_heartbeat: Last heartbeat timestamp
            ttl_seconds: Time-to-live for the entry
        """
        self.metadata = metadata
        self.registered_at = registered_at or datetime.utcnow()
        self.last_heartbeat = last_heartbeat or self.registered_at
        self.ttl_seconds = ttl_seconds

    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        expiry = self.last_heartbeat + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry

    def update_heartbeat(self) -> None:
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.model_dump(),
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "is_expired": self.is_expired,
        }


class AgentRegistry:
    """
    Registry for agent discovery and management.

    This class provides:
    - Agent registration and deregistration
    - Agent lookup by ID, capability, or tags
    - Heartbeat management and TTL-based expiry
    - Event callbacks for registry changes

    The registry can be used in-memory or extended for
    distributed backends (Redis, etcd, etc.).
    """

    def __init__(
        self,
        default_ttl_seconds: int = 300,
        cleanup_interval_seconds: int = 60,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the registry.

        Args:
            default_ttl_seconds: Default TTL for entries
            cleanup_interval_seconds: Interval for cleanup task
            logger: Optional logger instance
        """
        self._entries: Dict[str, RegistryEntry] = {}
        self._capability_index: Dict[str, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}

        self._default_ttl = default_ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds

        self._logger = logger or logging.getLogger("datamesh.registry")

        # Event callbacks
        self._on_register_callbacks: List[Callable] = []
        self._on_deregister_callbacks: List[Callable] = []

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the registry cleanup task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._logger.info("Registry started")

    async def stop(self) -> None:
        """Stop the registry cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        self._logger.info("Registry stopped")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired entries."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Cleanup error: {e}")

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from the registry."""
        expired = [
            agent_id
            for agent_id, entry in self._entries.items()
            if entry.is_expired
        ]

        for agent_id in expired:
            await self.deregister(agent_id)
            self._logger.info(f"Removed expired agent: {agent_id}")

    def _update_indices(self, metadata: AgentMetadata, add: bool = True) -> None:
        """
        Update capability and tag indices.

        Args:
            metadata: Agent metadata
            add: True to add, False to remove
        """
        # Update capability index
        for capability in metadata.capabilities:
            if add:
                if capability.name not in self._capability_index:
                    self._capability_index[capability.name] = set()
                self._capability_index[capability.name].add(metadata.agent_id)
            else:
                if capability.name in self._capability_index:
                    self._capability_index[capability.name].discard(metadata.agent_id)
                    if not self._capability_index[capability.name]:
                        del self._capability_index[capability.name]

        # Update tag index
        for tag in metadata.tags:
            if add:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(metadata.agent_id)
            else:
                if tag in self._tag_index:
                    self._tag_index[tag].discard(metadata.agent_id)
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]

    async def register(
        self,
        metadata: AgentMetadata,
        ttl_seconds: Optional[int] = None,
    ) -> RegistryEntry:
        """
        Register an agent.

        Args:
            metadata: Agent metadata
            ttl_seconds: Optional TTL override

        Returns:
            Registry entry

        Raises:
            AgentAlreadyExistsError: If agent already registered
        """
        if metadata.agent_id in self._entries:
            raise AgentAlreadyExistsError(metadata.agent_id)

        entry = RegistryEntry(
            metadata=metadata,
            ttl_seconds=ttl_seconds or self._default_ttl,
        )

        self._entries[metadata.agent_id] = entry
        self._update_indices(metadata, add=True)

        self._logger.info(f"Registered agent: {metadata.agent_id}")

        # Fire callbacks
        for callback in self._on_register_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metadata)
                else:
                    callback(metadata)
            except Exception as e:
                self._logger.error(f"Register callback error: {e}")

        return entry

    async def deregister(self, agent_id: str) -> None:
        """
        Deregister an agent.

        Args:
            agent_id: Agent identifier

        Raises:
            AgentNotFoundError: If agent not found
        """
        if agent_id not in self._entries:
            raise AgentNotFoundError(agent_id)

        entry = self._entries.pop(agent_id)
        self._update_indices(entry.metadata, add=False)

        self._logger.info(f"Deregistered agent: {agent_id}")

        # Fire callbacks
        for callback in self._on_deregister_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(entry.metadata)
                else:
                    callback(entry.metadata)
            except Exception as e:
                self._logger.error(f"Deregister callback error: {e}")

    async def update(self, metadata: AgentMetadata) -> RegistryEntry:
        """
        Update an agent's registration.

        Args:
            metadata: Updated agent metadata

        Returns:
            Updated registry entry

        Raises:
            AgentNotFoundError: If agent not found
        """
        if metadata.agent_id not in self._entries:
            raise AgentNotFoundError(metadata.agent_id)

        entry = self._entries[metadata.agent_id]

        # Update indices
        self._update_indices(entry.metadata, add=False)
        self._update_indices(metadata, add=True)

        # Update entry
        entry.metadata = metadata
        entry.update_heartbeat()

        self._logger.debug(f"Updated agent: {metadata.agent_id}")

        return entry

    async def heartbeat(self, agent_id: str) -> None:
        """
        Update agent heartbeat.

        Args:
            agent_id: Agent identifier

        Raises:
            AgentNotFoundError: If agent not found
        """
        if agent_id not in self._entries:
            raise AgentNotFoundError(agent_id)

        self._entries[agent_id].update_heartbeat()
        self._logger.debug(f"Heartbeat received: {agent_id}")

    def get(self, agent_id: str) -> Optional[AgentMetadata]:
        """
        Get agent metadata by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent metadata or None if not found
        """
        entry = self._entries.get(agent_id)
        return entry.metadata if entry and not entry.is_expired else None

    def get_entry(self, agent_id: str) -> Optional[RegistryEntry]:
        """
        Get registry entry by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Registry entry or None if not found
        """
        entry = self._entries.get(agent_id)
        return entry if entry and not entry.is_expired else None

    def find_by_capability(self, capability_name: str) -> List[AgentMetadata]:
        """
        Find agents by capability.

        Args:
            capability_name: Capability name to search for

        Returns:
            List of matching agent metadata
        """
        agent_ids = self._capability_index.get(capability_name, set())
        return [
            self._entries[agent_id].metadata
            for agent_id in agent_ids
            if agent_id in self._entries
            and not self._entries[agent_id].is_expired
        ]

    def find_by_tag(self, tag: str) -> List[AgentMetadata]:
        """
        Find agents by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching agent metadata
        """
        agent_ids = self._tag_index.get(tag, set())
        return [
            self._entries[agent_id].metadata
            for agent_id in agent_ids
            if agent_id in self._entries
            and not self._entries[agent_id].is_expired
        ]

    def find_by_tags(self, tags: List[str], match_all: bool = True) -> List[AgentMetadata]:
        """
        Find agents by multiple tags.

        Args:
            tags: Tags to search for
            match_all: If True, agent must have all tags

        Returns:
            List of matching agent metadata
        """
        if not tags:
            return []

        tag_sets = [
            self._tag_index.get(tag, set())
            for tag in tags
        ]

        if match_all:
            agent_ids = set.intersection(*tag_sets) if tag_sets else set()
        else:
            agent_ids = set.union(*tag_sets) if tag_sets else set()

        return [
            self._entries[agent_id].metadata
            for agent_id in agent_ids
            if agent_id in self._entries
            and not self._entries[agent_id].is_expired
        ]

    def list_all(self) -> List[AgentMetadata]:
        """
        List all registered agents.

        Returns:
            List of all agent metadata
        """
        return [
            entry.metadata
            for entry in self._entries.values()
            if not entry.is_expired
        ]

    def list_capabilities(self) -> List[str]:
        """
        List all available capabilities.

        Returns:
            List of capability names
        """
        return list(self._capability_index.keys())

    def list_tags(self) -> List[str]:
        """
        List all available tags.

        Returns:
            List of tags
        """
        return list(self._tag_index.keys())

    def count(self) -> int:
        """
        Get the number of registered agents.

        Returns:
            Number of agents
        """
        return sum(1 for e in self._entries.values() if not e.is_expired)

    def on_register(self, callback: Callable) -> None:
        """
        Register a callback for agent registration events.

        Args:
            callback: Function to call when agent is registered
        """
        self._on_register_callbacks.append(callback)

    def on_deregister(self, callback: Callable) -> None:
        """
        Register a callback for agent deregistration events.

        Args:
            callback: Function to call when agent is deregistered
        """
        self._on_deregister_callbacks.append(callback)
