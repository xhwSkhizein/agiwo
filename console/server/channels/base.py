"""
Base channel service — defines the generic message processing pipeline.

Concrete channel implementations (Feishu, Slack, etc.) inherit from
BaseChannelService and implement the abstract hooks for channel-specific
behaviour (message delivery, prompt rendering, error mapping).
"""

from abc import ABC, abstractmethod

from agiwo.agent import UserMessage, extract_text
from agiwo.utils.logging import get_logger

from server.channels.agent_executor import AgentExecutor
from server.channels.runtime_agent_pool import RuntimeAgentPool
from server.channels.session import SessionContextService, SessionManager
from server.channels.session.models import BatchContext, BatchPayload, InboundMessage

logger = get_logger(__name__)

_MAX_RESPONSE_TEXT_LEN = 6000
_MAX_LOG_TEXT_LEN = 1200


class BaseChannelService(ABC):
    """Generic channel service with batching, agent runtime, and scheduler integration.

    Subclasses must implement the abstract hooks that handle channel-specific
    concerns: message delivery, prompt building, and error display.
    """

    def __init__(
        self,
        *,
        session_service: SessionContextService,
        agent_pool: RuntimeAgentPool,
        executor: AgentExecutor,
        debounce_ms: int,
        max_batch_window_ms: int,
    ) -> None:
        self._session_service = session_service
        self._agent_pool = agent_pool
        self._executor = executor
        self._session_mgr = SessionManager(
            on_batch_ready=self._on_batch_ready,
            debounce_ms=debounce_ms,
            max_batch_window_ms=max_batch_window_ms,
        )

    async def close_base(self) -> None:
        await self._session_mgr.close()
        await self._agent_pool.close()

    @property
    def session_manager(self) -> SessionManager:
        return self._session_mgr

    @property
    def session_service(self) -> SessionContextService:
        return self._session_service

    @property
    def agent_pool(self) -> RuntimeAgentPool:
        return self._agent_pool

    @property
    def executor(self) -> AgentExecutor:
        return self._executor

    # -- Generic pipeline (not overridden by subclasses) ---------------------

    async def _on_batch_ready(
        self,
        chat_context_scope_id: str,
        context: BatchContext,
        messages: list[InboundMessage],
    ) -> None:
        user_message = await self._build_user_message(context, messages)
        batch = BatchPayload(
            context=context,
            messages=messages,
            user_message=user_message,
        )

        logger.info(
            "channel_batch_dispatched",
            chat_context_scope_id=chat_context_scope_id,
            chat_type=batch.context.chat_type,
            chat_id=batch.context.chat_id,
            message_count=len(batch.messages),
            input_preview=self._truncate_for_log(extract_text(user_message)),
        )

        try:
            await self._execute_batch(batch)
        except Exception as e:
            logger.exception(
                "channel_batch_execution_failed",
                chat_context_scope_id=chat_context_scope_id,
                error=str(e),
            )
            failure_text = self._to_user_facing_error(e)
            await self._deliver_reply(batch.context, failure_text)

    async def _execute_batch(self, batch: BatchPayload) -> None:
        resolution = await self._session_service.get_or_create_current_session(
            batch.context,
        )
        if resolution.retired_runtime_agent_id is not None:
            await self._agent_pool.close_runtime_agent(
                resolution.retired_runtime_agent_id,
            )
        session = resolution.session
        agent = await self._agent_pool.get_or_create_runtime_agent(session)

        is_first_output = True
        async for output in self._executor.execute(agent, session, batch.user_message):
            text = self._truncate_text(output)
            if is_first_output:
                await self._deliver_reply(batch.context, text)
                is_first_output = False
            else:
                await self._deliver_message(batch.context, text)

        if is_first_output:
            await self._deliver_reply(batch.context, "执行完成，但未产出可展示内容。")

    # -- Abstract hooks (channel-specific) -----------------------------------

    @abstractmethod
    async def _build_user_message(
        self,
        context: BatchContext,
        messages: list[InboundMessage],
    ) -> UserMessage: ...

    @abstractmethod
    async def _deliver_reply(self, context: BatchContext, text: str) -> None: ...

    @abstractmethod
    async def _deliver_message(self, context: BatchContext, text: str) -> None: ...

    @abstractmethod
    def _to_user_facing_error(self, error: Exception) -> str: ...

    # -- Shared helpers ------------------------------------------------------

    def _truncate_text(self, text: str, max_len: int = _MAX_RESPONSE_TEXT_LEN) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 20] + "\n\n[内容已截断]"

    def _truncate_for_log(self, text: str, max_len: int = _MAX_LOG_TEXT_LEN) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "...[truncated]"
