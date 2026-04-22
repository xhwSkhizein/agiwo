"""Runtime agent cache and lifecycle management for session-bound execution."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from agiwo.agent import Agent
from agiwo.scheduler.engine import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.exceptions import BaseAgentNotFoundError
from server.models.session import ChannelChatSessionStore, Session
from server.config import ConsoleConfig
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.runtime.agent_factory import materialize_agent

logger = get_logger(__name__)

ConfigSnapshot = tuple[
    str,
    str,
    str,
    str,
    str,
    tuple[str, ...] | None,  # allowed_tools
    tuple[str, ...] | None,  # allowed_skills
    tuple[tuple[str, Any], ...],
    tuple[tuple[str, Any], ...],
    str,  # updated_at
]


def _config_snapshot(config: AgentConfigRecord) -> ConfigSnapshot:
    dumped = config.model_dump(mode="json", exclude={"id", "created_at"})
    return (
        dumped["name"],
        dumped["description"],
        dumped["model_provider"],
        dumped["model_name"],
        dumped["system_prompt"],
        tuple(dumped["allowed_tools"]) if dumped["allowed_tools"] is not None else None,
        tuple(dumped["allowed_skills"])
        if dumped["allowed_skills"] is not None
        else None,
        tuple(sorted(dumped["options"].items())),
        tuple(sorted(dumped["model_params"].items())),
        dumped[
            "updated_at"
        ],  # model_dump(mode='json') already converts datetime to ISO string
    )


@dataclass(slots=True)
class CachedAgent:
    agent: Agent
    config_snapshot: ConfigSnapshot


class AgentRuntimeCache:
    """Cache for runtime Agent instances keyed by session_id."""

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        agent_registry: AgentRegistry,
        console_config: ConsoleConfig,
        session_store: ChannelChatSessionStore,
    ) -> None:
        self._scheduler = scheduler
        self._agent_registry = agent_registry
        self._console_config = console_config
        self._session_store = session_store
        self._cache: dict[str, CachedAgent] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def runtime_agents(self) -> dict[str, Agent]:
        return {key: cached.agent for key, cached in self._cache.items()}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for the given session_id."""
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def _get_or_create_runtime_agent_locked(self, session: Session) -> Agent:
        """Helper method that performs the actual get/create/rebind logic under lock."""
        base_config = await self._agent_registry.get_agent(session.base_agent_id)
        if base_config is None:
            raise BaseAgentNotFoundError(session.base_agent_id)

        snapshot = _config_snapshot(base_config)
        cached = self._cache.get(session.id)
        if cached is not None:
            if cached.config_snapshot == snapshot:
                return cached.agent
            logger.info(
                "runtime_agent_refresh_on_config_change",
                runtime_agent_id=session.id,
                base_agent_id=session.base_agent_id,
            )

        agent = await materialize_agent(
            base_config,
            self._console_config,
            self._agent_registry,
            id=session.id,
        )
        if cached is not None:
            rebound = await self._scheduler.rebind_agent(session.id, agent)
            if not rebound:
                logger.info(
                    "runtime_agent_refresh_deferred",
                    runtime_agent_id=session.id,
                    base_agent_id=session.base_agent_id,
                    reason="state_active",
                )
                await agent.close()
                return cached.agent

        retired = self._cache.pop(session.id, None)
        if retired is not None and retired.agent is not agent:
            await retired.agent.close()

        session.updated_at = datetime.now(timezone.utc)
        await self._session_store.upsert_session(session)

        self._cache[session.id] = CachedAgent(
            agent=agent,
            config_snapshot=snapshot,
        )
        return agent

    async def get_or_create_runtime_agent(self, session: Session) -> Agent:
        """Get or create a runtime agent for the given session, with per-session locking."""
        lock = self._get_lock(session.id)
        async with lock:
            return await self._get_or_create_runtime_agent_locked(session)

    async def close_runtime_agent(self, agent_id: str) -> None:
        retired = self._cache.pop(agent_id, None)
        if retired is not None:
            await retired.agent.close()

    async def close(self) -> None:
        close_tasks = [cached.agent.close() for cached in self._cache.values()]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        self._cache.clear()
