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
from server.services.runtime.agent_factory import build_agent

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
]


def _config_snapshot(config: AgentConfigRecord) -> ConfigSnapshot:
    return (
        config.name,
        config.description,
        config.model_provider,
        config.model_name,
        config.system_prompt,
        tuple(config.allowed_tools) if config.allowed_tools is not None else None,
        tuple(config.allowed_skills) if config.allowed_skills is not None else None,
        tuple(sorted(config.options.items())),
        tuple(sorted(config.model_params.items())),
    )


@dataclass(slots=True)
class CachedAgent:
    agent: Agent
    config_snapshot: ConfigSnapshot


class AgentRuntimeCache:
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

    @property
    def runtime_agents(self) -> dict[str, Agent]:
        return {key: cached.agent for key, cached in self._cache.items()}

    async def get_or_create_runtime_agent(self, session: Session) -> Agent:
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

        agent = await build_agent(
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

    async def close_runtime_agent(self, agent_id: str) -> None:
        retired = self._cache.pop(agent_id, None)
        if retired is not None:
            await retired.agent.close()

    async def close(self) -> None:
        close_tasks = [cached.agent.close() for cached in self._cache.values()]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        self._cache.clear()
