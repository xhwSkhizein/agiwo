"""Runtime agent cache and lifecycle management for session-bound execution."""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from agiwo.agent import Agent
from agiwo.scheduler.engine import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.exceptions import BaseAgentNotFoundError
from server.models.session import ChannelChatSessionStore, Session
from server.config import ConsoleConfig
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.runtime.agent_factory import build_agent

logger = get_logger(__name__)


@dataclass(slots=True)
class CachedAgent:
    agent: Agent
    config_fingerprint: str


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

        expected_fingerprint = self._build_runtime_config_fingerprint(base_config)
        cached = self._cache.get(session.id)
        if cached is not None:
            if cached.config_fingerprint == expected_fingerprint:
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
            if not rebound and cached is not None:
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
            config_fingerprint=expected_fingerprint,
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

    def _build_runtime_config_fingerprint(self, base_config: AgentConfigRecord) -> str:
        payload = {
            "name": base_config.name,
            "description": base_config.description,
            "model_provider": base_config.model_provider,
            "model_name": base_config.model_name,
            "system_prompt": base_config.system_prompt,
            "tools": list(base_config.tools),
            "options": dict(base_config.options),
            "model_params": dict(base_config.model_params),
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()
