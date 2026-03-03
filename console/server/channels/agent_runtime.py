"""
Agent runtime lifecycle management and scheduler bridge.

Manages runtime agent instances (cache, creation, config mapping) and
provides a unified interface for submitting tasks to the Scheduler.
"""

import asyncio
import hashlib
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from agiwo.agent.agent import Agent
from agiwo.scheduler.models import AgentStateStatus, SchedulerOutput
from agiwo.scheduler.scheduler import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.models import BatchContext, SessionRuntime, SessionRuntimeStore
from server.config import ConsoleConfig
from server.services.agent_builder import build_agent
from server.services.agent_registry import AgentConfigRecord, AgentRegistry

logger = get_logger(__name__)


class AgentRuntimeManager:
    def __init__(
        self,
        *,
        scheduler: Scheduler,
        agent_registry: AgentRegistry,
        console_config: ConsoleConfig,
        store: SessionRuntimeStore,
        default_agent_name: str,
        scheduler_wait_timeout: int,
    ) -> None:
        self._scheduler = scheduler
        self._agent_registry = agent_registry
        self._console_config = console_config
        self._store = store
        self._default_agent_name = default_agent_name
        self._scheduler_wait_timeout = scheduler_wait_timeout

        self._runtime_agents: dict[str, Agent] = {}

    @property
    def runtime_agents(self) -> dict[str, Agent]:
        return self._runtime_agents

    async def resolve_default_agent_config(self) -> AgentConfigRecord | None:
        if not self._default_agent_name:
            return None
        return await self._agent_registry.get_agent_by_name(self._default_agent_name)

    async def get_or_create_runtime(self, context: BatchContext) -> SessionRuntime:
        runtime = await self._store.get_session_runtime(context.session_key)
        if runtime is not None:
            return await self._ensure_runtime_base_agent(runtime)

        runtime_agent_id = self._build_runtime_agent_id(
            context.base_agent_id, context.session_key,
        )
        agiwo_session_id = str(uuid5(NAMESPACE_URL, context.session_key))

        runtime = SessionRuntime(
            session_key=context.session_key,
            agiwo_session_id=agiwo_session_id,
            runtime_agent_id=runtime_agent_id,
            scheduler_state_id=runtime_agent_id,
            base_agent_id=context.base_agent_id,
            chat_id=context.chat_id,
            chat_type=context.chat_type,
            trigger_user_id=context.trigger_user_id,
            updated_at=datetime.now(timezone.utc),
        )
        await self._store.upsert_session_runtime(runtime)
        return runtime

    async def get_or_create_runtime_agent(self, runtime: SessionRuntime) -> Agent:
        existing = self._runtime_agents.get(runtime.runtime_agent_id)
        if existing is not None:
            return existing

        base_config = await self._agent_registry.get_agent(runtime.base_agent_id)
        if base_config is None:
            raise RuntimeError(f"base_agent_not_found: {runtime.base_agent_id}")

        runtime_config = self._to_runtime_config(base_config, runtime.runtime_agent_id)
        agent = await build_agent(runtime_config, self._console_config, self._agent_registry)
        self._runtime_agents[runtime.runtime_agent_id] = agent
        return agent

    async def submit_to_scheduler(
        self,
        agent: Agent,
        runtime: SessionRuntime,
        user_input: str,
    ) -> AsyncIterator[SchedulerOutput]:
        current_state = await self._scheduler.get_state(runtime.scheduler_state_id)

        if current_state is None or current_state.status in (
            AgentStateStatus.COMPLETED,
            AgentStateStatus.FAILED,
        ):
            output_stream = self._scheduler.submit_and_subscribe(
                agent,
                user_input,
                session_id=runtime.agiwo_session_id,
                persistent=True,
                timeout=self._scheduler_wait_timeout,
            )
            runtime.scheduler_state_id = agent.id
        elif current_state.status == AgentStateStatus.SLEEPING:
            output_stream = self._scheduler.submit_task_and_subscribe(
                runtime.scheduler_state_id,
                user_input,
                agent=agent,
                timeout=self._scheduler_wait_timeout,
            )
        elif current_state.status in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.PENDING,
        ):
            output_stream = await self._handle_running_state(
                agent, runtime, user_input,
            )
        else:
            output_stream = self._scheduler.submit_and_subscribe(
                agent,
                user_input,
                session_id=runtime.agiwo_session_id,
                persistent=True,
                timeout=self._scheduler_wait_timeout,
            )
            runtime.scheduler_state_id = agent.id

        runtime.updated_at = datetime.now(timezone.utc)
        await self._store.upsert_session_runtime(runtime)
        return output_stream

    async def close_runtime_agent(self, agent_id: str) -> None:
        cached = self._runtime_agents.pop(agent_id, None)
        if cached is not None:
            await cached.close()

    async def close(self) -> None:
        close_tasks = [agent.close() for agent in self._runtime_agents.values()]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        self._runtime_agents.clear()

    # -- Internal ------------------------------------------------------------

    async def _ensure_runtime_base_agent(
        self,
        runtime: SessionRuntime,
    ) -> SessionRuntime:
        base_config = await self._agent_registry.get_agent(runtime.base_agent_id)
        if base_config is not None:
            return runtime

        default_config = await self.resolve_default_agent_config()
        if default_config is None:
            raise RuntimeError(f"base_agent_not_found: {runtime.base_agent_id}")

        old_base_agent_id = runtime.base_agent_id
        old_runtime_agent_id = runtime.runtime_agent_id
        new_runtime_agent_id = self._build_runtime_agent_id(
            default_config.id, runtime.session_key,
        )

        if old_runtime_agent_id != new_runtime_agent_id:
            await self.close_runtime_agent(old_runtime_agent_id)

        runtime.base_agent_id = default_config.id
        runtime.runtime_agent_id = new_runtime_agent_id
        runtime.scheduler_state_id = new_runtime_agent_id
        runtime.updated_at = datetime.now(timezone.utc)
        await self._store.upsert_session_runtime(runtime)

        logger.warning(
            "channel_runtime_rebind_base_agent",
            session_key=runtime.session_key,
            old_base_agent_id=old_base_agent_id,
            new_base_agent_id=runtime.base_agent_id,
            old_runtime_agent_id=old_runtime_agent_id,
            new_runtime_agent_id=new_runtime_agent_id,
        )
        return runtime

    async def _handle_running_state(
        self,
        agent: Agent,
        runtime: SessionRuntime,
        user_input: str,
    ) -> AsyncIterator[SchedulerOutput]:
        await self._scheduler.wait_for(
            runtime.scheduler_state_id,
            timeout=self._scheduler_wait_timeout,
        )
        refreshed = await self._scheduler.get_state(runtime.scheduler_state_id)

        if refreshed is not None and refreshed.status == AgentStateStatus.SLEEPING:
            return self._scheduler.submit_task_and_subscribe(
                runtime.scheduler_state_id,
                user_input,
                agent=agent,
                timeout=self._scheduler_wait_timeout,
            )

        if refreshed is not None and refreshed.status in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.PENDING,
        ):
            raise RuntimeError("previous_task_still_running_after_timeout")

        runtime.scheduler_state_id = agent.id
        return self._scheduler.submit_and_subscribe(
            agent,
            user_input,
            session_id=runtime.agiwo_session_id,
            persistent=True,
            timeout=self._scheduler_wait_timeout,
        )

    def _build_runtime_agent_id(self, base_agent_id: str, session_key: str) -> str:
        digest = hashlib.sha1(session_key.encode("utf-8")).hexdigest()[:12]
        return f"{base_agent_id}--{digest}"

    def _to_runtime_config(
        self,
        base_config: AgentConfigRecord,
        runtime_agent_id: str,
    ) -> AgentConfigRecord:
        return AgentConfigRecord(
            id=runtime_agent_id,
            name=base_config.name,
            description=base_config.description,
            model_provider=base_config.model_provider,
            model_name=base_config.model_name,
            system_prompt=base_config.system_prompt,
            tools=list(base_config.tools),
            options=dict(base_config.options),
            model_params=dict(base_config.model_params),
            created_at=base_config.created_at,
            updated_at=datetime.now(),
        )
