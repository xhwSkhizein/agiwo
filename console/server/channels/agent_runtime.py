"""
Agent runtime lifecycle management and scheduler bridge.

Coordinates session persistence, runtime-agent cache, and scheduler bridging.
"""

from collections.abc import AsyncIterator

from agiwo.agent import Agent, UserInput
from agiwo.scheduler.scheduler import Scheduler

from server.channels.runtime_agent_pool import RuntimeAgentPool
from server.channels.scheduler_session_bridge import SchedulerSessionBridge
from server.channels.session_context_service import SessionContextService
from server.channels.models import (
    BatchContext,
    ChannelChatContext,
    ChannelChatSessionStore,
    Session,
    SessionCreateResult,
    SessionSwitchResult,
    UserSessionItem,
)
from server.config import ConsoleConfig
from server.services.agent_registry import AgentConfigRecord, AgentRegistry


class AgentRuntimeManager:
    def __init__(
        self,
        *,
        scheduler: Scheduler,
        agent_registry: AgentRegistry,
        console_config: ConsoleConfig,
        store: ChannelChatSessionStore,
        default_agent_name: str,
        scheduler_wait_timeout: int,
    ) -> None:
        self._session_context_service = SessionContextService(
            store=store,
            agent_registry=agent_registry,
            default_agent_name=default_agent_name,
        )
        self._runtime_agent_pool = RuntimeAgentPool(
            scheduler=scheduler,
            agent_registry=agent_registry,
            console_config=console_config,
            store=store,
        )
        self._scheduler_bridge = SchedulerSessionBridge(
            scheduler=scheduler,
            store=store,
            scheduler_wait_timeout=scheduler_wait_timeout,
        )

    @property
    def runtime_agents(self) -> dict[str, Agent]:
        return self._runtime_agent_pool.runtime_agents

    async def resolve_default_agent_config(self) -> AgentConfigRecord | None:
        return await self._session_context_service.resolve_default_agent_config()

    async def get_chat_context(
        self,
        chat_context_scope_id: str,
    ) -> ChannelChatContext | None:
        return await self._session_context_service.get_chat_context(chat_context_scope_id)

    async def get_chat_context_and_current_session(
        self,
        chat_context_scope_id: str,
    ) -> tuple[ChannelChatContext | None, Session | None]:
        return await self._session_context_service.get_chat_context_and_current_session(
            chat_context_scope_id
        )

    async def get_or_create_current_session(
        self,
        context: BatchContext,
    ) -> SessionCreateResult:
        resolution = await self._session_context_service.get_or_create_current_session(
            context
        )
        if resolution.retired_runtime_agent_id is not None:
            await self._runtime_agent_pool.close_runtime_agent(
                resolution.retired_runtime_agent_id
            )
        return SessionCreateResult(
            chat_context=resolution.chat_context,
            session=resolution.session,
        )

    async def create_new_session(
        self,
        *,
        chat_context_scope_id: str,
        channel_instance_id: str,
        chat_id: str,
        chat_type: str,
        user_open_id: str,
        base_agent_id: str,
        created_by: str,
    ) -> SessionCreateResult:
        return await self._session_context_service.create_new_session(
            chat_context_scope_id=chat_context_scope_id,
            channel_instance_id=channel_instance_id,
            chat_id=chat_id,
            chat_type=chat_type,
            user_open_id=user_open_id,
            base_agent_id=base_agent_id,
            created_by=created_by,
        )

    async def switch_session(
        self,
        *,
        chat_context_scope_id: str,
        target_session_id: str,
    ) -> SessionSwitchResult:
        return await self._session_context_service.switch_session(
            chat_context_scope_id=chat_context_scope_id,
            target_session_id=target_session_id,
        )

    async def list_user_sessions(
        self,
        *,
        user_open_id: str,
        current_chat_context_scope_id: str | None,
    ) -> list[UserSessionItem]:
        return await self._session_context_service.list_user_sessions(
            user_open_id=user_open_id,
            current_chat_context_scope_id=current_chat_context_scope_id,
        )

    async def terminate_session_runtime(
        self,
        session: Session,
        reason: str,
    ) -> None:
        await self._scheduler_bridge.cancel_session_state_if_active(session, reason)
        if session.runtime_agent_id:
            await self._runtime_agent_pool.close_runtime_agent(session.runtime_agent_id)
        await self._session_context_service.touch_session(session)

    async def get_or_create_runtime_agent(self, session: Session) -> Agent:
        return await self._runtime_agent_pool.get_or_create_runtime_agent(session)

    async def submit_to_scheduler(
        self,
        agent: Agent,
        session: Session,
        user_input: UserInput,
    ) -> AsyncIterator[str]:
        return await self._scheduler_bridge.submit_to_scheduler(
            agent,
            session,
            user_input,
        )

    async def close_runtime_agent(self, agent_id: str) -> None:
        await self._runtime_agent_pool.close_runtime_agent(agent_id)

    async def close(self) -> None:
        await self._runtime_agent_pool.close()
