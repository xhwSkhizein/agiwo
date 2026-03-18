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
from server.channels.deferred_reply import DeferredReplyManager
from server.channels.runtime_agent_pool import RuntimeAgentPool
from server.channels.session import SessionContextService, SessionManager
from server.channels.session.models import BatchContext, BatchPayload, InboundMessage

logger = get_logger(__name__)

_MAX_CHUNK_LEN = 6000
_MAX_LOG_TEXT_LEN = 1200


async def safe_close_all(*closables: object) -> None:
    """Close multiple resources, logging and suppressing individual errors.

    Each *closable* is expected to have an async ``close()`` method.
    If any call raises, the error is logged but does **not** prevent the
    remaining resources from being closed.
    """
    for obj in closables:
        try:
            close_fn = getattr(obj, "close", None)
            if close_fn is not None:
                await close_fn()
        except Exception:  # noqa: BLE001 — must not leak during shutdown
            logger.warning(
                "resource_close_failed",
                resource=type(obj).__name__,
                exc_info=True,
            )


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
        self._deferred_replies = DeferredReplyManager(
            executor=executor,
            session_service=session_service,
            deliver_message=self._deliver_message,
        )
        self._session_mgr = SessionManager(
            on_batch_ready=self._on_batch_ready,
            debounce_ms=debounce_ms,
            max_batch_window_ms=max_batch_window_ms,
        )

    async def close_base(self) -> None:
        await safe_close_all(
            self._deferred_replies,
            self._session_mgr,
            self._agent_pool,
        )

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
            chunks = self._split_text_into_chunks(output)
            for i, chunk in enumerate(chunks):
                if is_first_output and i == 0:
                    await self._deliver_reply(batch.context, chunk)
                    is_first_output = False
                else:
                    await self._deliver_message(batch.context, chunk)

        state = await self._executor.get_state(session.scheduler_state_id)
        if state is not None and state.is_active():
            self._deferred_replies.arm(session=session, context=batch.context)
            return

        if is_first_output:
            if state is not None and state.result_summary:
                chunks = self._split_text_into_chunks(state.result_summary)
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        await self._deliver_reply(batch.context, chunk)
                    else:
                        await self._deliver_message(batch.context, chunk)
                return
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

    def _split_text_into_chunks(
        self, text: str, max_len: int = _MAX_CHUNK_LEN
    ) -> list[str]:
        """将长文本分块，保留完整信息而不是截断。"""
        if len(text) <= max_len:
            return [text]

        raw_chunks: list[str] = []
        current_pos = 0
        total_len = len(text)

        while current_pos < total_len:
            if total_len - current_pos <= max_len:
                raw_chunks.append(text[current_pos:])
                break

            chunk_end = current_pos + max_len
            last_newline = text.rfind("\n", current_pos, chunk_end)
            if last_newline > current_pos:
                chunk_end = last_newline + 1

            raw_chunks.append(text[current_pos:chunk_end])
            current_pos = chunk_end

        total = len(raw_chunks)
        return [
            chunk + f"\n\n[续 {i + 1}/{total}]" if i < total - 1 else chunk
            for i, chunk in enumerate(raw_chunks)
        ]

    def _truncate_for_log(self, text: str, max_len: int = _MAX_LOG_TEXT_LEN) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "...[truncated]"
