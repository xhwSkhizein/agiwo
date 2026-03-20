"""
Base channel service — defines the generic message processing pipeline.

Concrete channel implementations (Feishu, Slack, etc.) inherit from
BaseChannelService and implement the abstract hooks for channel-specific
behaviour (message delivery, prompt rendering, error mapping).
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from agiwo.agent import Agent, UserMessage, extract_text
from agiwo.agent.runtime import AgentStreamItem, RunCompletedEvent, RunFailedEvent
from agiwo.utils.logging import get_logger

from server.channels.agent_executor import AgentExecutor
from server.channels.deferred_reply import DeferredReplyManager
from server.channels.runtime_agent_pool import RuntimeAgentPool
from server.channels.session import SessionContextService, SessionManager
from server.channels.session.models import (
    BatchContext,
    BatchPayload,
    InboundMessage,
    Session,
)

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
            deliver_chunked=self._deliver_message,
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
        session, agent = await self._prepare_batch_runtime(batch)
        dispatch = await self._executor.execute(agent, session, batch.user_message)
        if dispatch.action == "steered":
            await self._handle_steered_dispatch(batch, session)
            return

        had_output = await self._consume_dispatch_stream(
            batch,
            session,
            dispatch.stream,
        )
        await self._finalize_dispatch(batch, session, had_output)

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

    async def _handle_stream_item(
        self,
        batch: BatchPayload,
        session: Session,
        item: AgentStreamItem,
    ) -> bool:
        del batch, session, item
        return False

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

    async def _can_deliver_session(
        self,
        context: BatchContext,
        *,
        session_id: str,
        state_id: str,
    ) -> bool:
        (
            _chat_context,
            current_session,
        ) = await self._session_service.get_chat_context_and_current_session(
            context.chat_context_scope_id
        )
        if current_session is None:
            return False
        return (
            current_session.id == session_id
            and current_session.scheduler_state_id == state_id
        )

    async def _prepare_batch_runtime(
        self, batch: BatchPayload
    ) -> tuple[Session, Agent]:
        resolution = await self._session_service.get_or_create_current_session(
            batch.context,
        )
        if resolution.retired_runtime_agent_id is not None:
            await self._agent_pool.close_runtime_agent(
                resolution.retired_runtime_agent_id,
            )
        session = resolution.session
        agent = await self._agent_pool.get_or_create_runtime_agent(session)
        return session, agent

    async def _handle_steered_dispatch(
        self,
        batch: BatchPayload,
        session: Session,
    ) -> None:
        if await self._can_deliver_target(batch.context, session):
            await self._deliver_reply(batch.context, "消息已收到，正在继续处理。")
        await self._arm_deferred_reply_if_active(batch, session)

    async def _consume_dispatch_stream(
        self,
        batch: BatchPayload,
        session: Session,
        stream: AsyncIterator[AgentStreamItem] | None,
    ) -> bool:
        if stream is None:
            return False

        had_output = False
        async for item in stream:
            if not await self._can_deliver_target(batch.context, session):
                continue
            if await self._handle_stream_item(batch, session, item):
                continue
            text = _extract_stream_text(item)
            if text is None:
                continue
            had_output = await self._deliver_stream_text(
                batch.context,
                text,
                had_output=had_output,
            )
        return had_output

    async def _finalize_dispatch(
        self,
        batch: BatchPayload,
        session: Session,
        had_output: bool,
    ) -> None:
        state = await self._executor.get_state(session.scheduler_state_id)
        if await self._arm_deferred_reply_if_active(
            batch,
            session,
            state=state,
        ):
            return
        if not await self._can_deliver_target(batch.context, session):
            return
        if had_output:
            return
        if state is not None and state.result_summary:
            await self._deliver_stream_text(
                batch.context,
                state.result_summary,
                had_output=False,
            )
            return
        await self._deliver_reply(batch.context, "执行完成，但未产出可展示内容。")

    async def _arm_deferred_reply_if_active(
        self,
        batch: BatchPayload,
        session: Session,
        *,
        state=None,
    ) -> bool:
        resolved_state = state
        if resolved_state is None:
            resolved_state = await self._executor.get_state(session.scheduler_state_id)
        if resolved_state is None or not resolved_state.is_active():
            return False
        if not await self._can_deliver_target(batch.context, session):
            return False
        self._deferred_replies.arm(session=session, context=batch.context)
        return True

    async def _deliver_stream_text(
        self,
        context: BatchContext,
        text: str,
        *,
        had_output: bool,
    ) -> bool:
        chunks = self._split_text_into_chunks(text)
        for index, chunk in enumerate(chunks):
            if not had_output and index == 0:
                await self._deliver_reply(context, chunk)
                had_output = True
                continue
            await self._deliver_message(context, chunk)
            had_output = True
        return had_output

    async def _can_deliver_target(
        self,
        context: BatchContext,
        session: Session,
    ) -> bool:
        return await self._can_deliver_session(
            context,
            session_id=session.id,
            state_id=session.scheduler_state_id,
        )


def _extract_stream_text(item: AgentStreamItem) -> str | None:
    if isinstance(item, RunCompletedEvent):
        if not item.response:
            return None
        if item.depth == 0:
            return item.response
        return (
            f"<notice>agent_id={item.agent_id}, status=completed</notice>\n"
            f"{item.response}"
        )
    if isinstance(item, RunFailedEvent):
        if item.depth == 0:
            return item.error
        return f"<notice>agent_id={item.agent_id}, status=failed</notice>\n{item.error}"
    return None
