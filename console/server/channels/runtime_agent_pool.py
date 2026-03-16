"""
Runtime agent cache and lifecycle management for channel sessions.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone

from agiwo.agent.agent import Agent
from agiwo.scheduler.models import AgentStateStatus
from agiwo.scheduler.scheduler import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.session_binding import assign_runtime_identity
from server.channels.models import ChannelChatSessionStore, Session
from server.config import ConsoleConfig
from server.services.agent_lifecycle import build_agent
from server.services.agent_registry import AgentConfigRecord, AgentRegistry

logger = get_logger(__name__)


class RuntimeAgentPool:
    def __init__(
        self,
        *,
        scheduler: Scheduler,
        agent_registry: AgentRegistry,
        console_config: ConsoleConfig,
        store: ChannelChatSessionStore,
    ) -> None:
        self._scheduler = scheduler
        self._agent_registry = agent_registry
        self._console_config = console_config
        self._store = store
        self._runtime_agents: dict[str, Agent] = {}
        self._runtime_agent_config_fingerprints: dict[str, str] = {}

    @property
    def runtime_agents(self) -> dict[str, Agent]:
        return self._runtime_agents

    async def get_or_create_runtime_agent(self, session: Session) -> Agent:
        base_config = await self._agent_registry.get_agent(session.base_agent_id)
        if base_config is None:
            raise RuntimeError(f"base_agent_not_found: {session.base_agent_id}")

        expected_fingerprint = self._build_runtime_config_fingerprint(base_config)
        existing = self._runtime_agents.get(session.runtime_agent_id)
        if existing is not None:
            current_fingerprint = self._runtime_agent_config_fingerprints.get(
                session.runtime_agent_id
            )
            if current_fingerprint == expected_fingerprint:
                return existing

            state = await self._scheduler.get_state(session.scheduler_state_id)
            if state is not None and state.status == AgentStateStatus.RUNNING:
                logger.info(
                    "runtime_agent_refresh_deferred",
                    runtime_agent_id=session.runtime_agent_id,
                    base_agent_id=session.base_agent_id,
                    reason="state_running",
                )
                return existing

            logger.info(
                "runtime_agent_refresh_on_config_change",
                runtime_agent_id=session.runtime_agent_id,
                base_agent_id=session.base_agent_id,
            )
            await self.close_runtime_agent(session.runtime_agent_id)

        agent = await build_agent(
            base_config,
            self._console_config,
            self._agent_registry,
            id=session.runtime_agent_id or None,
        )
        self._assign_runtime_identity(session, agent)
        session.updated_at = datetime.now(timezone.utc)
        await self._store.upsert_session(session)

        self._runtime_agents[session.runtime_agent_id] = agent
        self._runtime_agent_config_fingerprints[session.runtime_agent_id] = (
            expected_fingerprint
        )
        return agent

    async def close_runtime_agent(self, agent_id: str) -> None:
        cached = self._runtime_agents.pop(agent_id, None)
        self._runtime_agent_config_fingerprints.pop(agent_id, None)
        if cached is not None:
            await cached.close()

    async def close(self) -> None:
        close_tasks = [agent.close() for agent in self._runtime_agents.values()]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        self._runtime_agents.clear()
        self._runtime_agent_config_fingerprints.clear()

    def _assign_runtime_identity(self, session: Session, agent: Agent) -> None:
        assign_runtime_identity(session, agent.id)

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
