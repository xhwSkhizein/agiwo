"""
Feishu channel service — concrete channel implementation.

Handles Feishu-specific concerns: long connection, message parsing, trigger
rules, ack/response delivery, and prompt rendering, as well as the generic
batching, agent runtime, and scheduler message pipeline.
"""

import asyncio
import shutil
from collections.abc import AsyncIterator
from typing import Any, Literal

from agiwo.agent import (
    Agent,
    AgentStreamItem,
    RunCompletedEvent,
    RunFailedEvent,
    StepCompletedEvent,
    UserMessage,
)
from agiwo.scheduler.engine import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.agent_executor import AgentExecutor
from server.channels.base import (
    extract_stream_text,
    safe_close_all,
    split_text_into_chunks,
    truncate_for_log,
)
from server.channels.deferred_reply import DeferredReplyManager
from server.channels.exceptions import (
    BaseAgentNotFoundError,
    DefaultAgentNameNotFoundError,
    PreviousTaskRunningError,
)
from server.channels.feishu.commands import build_feishu_command_registry
from server.channels.feishu.factory import FeishuServiceFactory
from server.channels.feishu.inbound_handler import FeishuInboundHandler
from server.channels.feishu.message_parser import FeishuInboundEnvelope
from server.channels.runtime_agent_pool import RuntimeAgentPool
from server.channels.session import SessionContextService, SessionManager
from server.channels.session.models import (
    BatchContext,
    BatchPayload,
    InboundMessage,
    Session,
)
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry

logger = get_logger(__name__)


class FeishuChannelService:
    def __init__(
        self,
        *,
        config: ConsoleConfig,
        scheduler: Scheduler,
        agent_registry: AgentRegistry,
    ) -> None:
        components = FeishuServiceFactory.create_components(
            config=config,
            scheduler=scheduler,
            agent_registry=agent_registry,
        )

        self._bot_open_id = components.bot_open_id
        self._api = components.api
        self._store = components.store
        self._parser = components.parser
        self._connection = components.connection
        self._tmp_dir = components.tmp_dir
        self._message_builder = components.message_builder
        self._delivery_service = components.delivery_service

        self._session_service = components.session_service
        self._agent_pool = components.agent_pool
        self._executor = components.executor
        self._deferred_replies = DeferredReplyManager(
            executor=components.executor,
            session_service=components.session_service,
            deliver_chunked=self._deliver_message,
        )
        self._session_mgr = SessionManager(
            on_batch_ready=self._on_batch_ready,
            debounce_ms=config.feishu_debounce_ms,
            max_batch_window_ms=config.feishu_max_batch_window_ms,
        )

        command_registry = build_feishu_command_registry(
            session_service=components.session_service,
            agent_pool=components.agent_pool,
            session_manager=self._session_mgr,
            scheduler=scheduler,
            agent_registry=agent_registry,
            console_config=config,
        )
        self._inbound_handler = FeishuInboundHandler(
            channel_instance_id=config.feishu_channel_instance_id,
            default_agent_name=config.feishu_default_agent_name,
            whitelist_open_ids=set(config.feishu_whitelist_open_ids),
            parser=components.parser,
            content_extractor=components.content_extractor,
            group_history_store=components.group_history_store,
            store=components.store,
            session_service=components.session_service,
            session_manager=self._session_mgr,
            command_registry=command_registry,
            delivery_service=components.delivery_service,
            truncate_for_log=truncate_for_log,
        )
        self._verbose_mode: Literal["full", "lite", "off"] = config.feishu_verbose_mode
        self._closed = False

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

    async def initialize(self) -> None:
        await self._store.connect()

        if not self._bot_open_id:
            logger.warning(
                "feishu_bot_open_id_not_configured",
                detail=(
                    "group messages will not trigger without "
                    "AGIWO_CONSOLE_FEISHU_BOT_OPEN_ID"
                ),
            )

        main_loop = asyncio.get_running_loop()
        await self._connection.start(main_loop, self._process_incoming_envelope)

    async def close(self) -> None:
        self._closed = True
        await safe_close_all(
            self._deferred_replies,
            self._session_mgr,
            self._agent_pool,
        )
        try:
            await self._connection.stop()
        except Exception:  # noqa: BLE001
            logger.warning(
                "resource_close_failed", resource="FeishuConnection", exc_info=True
            )
        await safe_close_all(self._api, self._store)
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def get_status(self) -> dict[str, Any]:
        return {
            "mode": "long_connection",
            "long_connection_alive": self._connection.is_alive(),
            "session_count": self._session_mgr.session_count,
        }

    async def _process_incoming_envelope(
        self,
        envelope: FeishuInboundEnvelope,
    ) -> dict[str, Any]:
        if self._closed:
            return {"msg": "feishu_channel_closed"}
        return await self._inbound_handler.process_envelope(envelope)

    # -- Message pipeline -------------------------------------------------------

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
            input_preview=truncate_for_log(user_message.extract_text()),
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
            text = extract_stream_text(item)
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
        chunks = split_text_into_chunks(text)
        for index, chunk in enumerate(chunks):
            if not had_output and index == 0:
                await self._deliver_reply(context, chunk)
                had_output = True
                continue
            await self._deliver_message(context, chunk)
            had_output = True
        return had_output

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

    # -- Feishu-specific hooks --------------------------------------------------

    async def _build_user_message(
        self,
        context: BatchContext,
        messages: list[InboundMessage],
    ) -> UserMessage:
        return await self._message_builder.build_user_message(context, messages)

    async def _deliver_reply(self, context: BatchContext, text: str) -> None:
        await self._delivery_service.deliver_reply(context, text)

    async def _deliver_message(self, context: BatchContext, text: str) -> None:
        await self._delivery_service.deliver_message(context, text)

    def _to_user_facing_error(self, error: Exception) -> str:
        if isinstance(error, PreviousTaskRunningError):
            return "上一条任务仍在处理中，请稍后再试。"
        if isinstance(error, BaseAgentNotFoundError):
            return f"指定的 Agent '{error.base_agent_id}' 不存在或已被删除，请验证该 Agent 是否存在于系统中。"
        if isinstance(error, DefaultAgentNameNotFoundError):
            return (
                f"当前默认 Agent 名称 '{error.agent_name}' 不存在，请检查 "
                "AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME。"
            )
        return f"执行失败: {str(error)}"

    async def _handle_stream_item(
        self,
        batch: BatchPayload,
        session: Session,
        item: AgentStreamItem,
    ) -> bool:
        del batch, session, item
        return False

    # -- Verbose mode hooks ----------------------------------------------------

    def _format_stream_item(self, item: AgentStreamItem) -> str | None:
        if self._verbose_mode == "off":
            return None
        if self._verbose_mode == "lite":
            return self._format_lite(item)
        return self._format_full(item)

    def _format_steer_confirmation(self) -> str | None:
        if self._verbose_mode == "off":
            return None
        return "消息已收到，任务正在处理中。"

    @staticmethod
    def _format_lite(item: AgentStreamItem) -> str | None:
        if isinstance(item, RunCompletedEvent) and item.depth == 0:
            return item.response
        if isinstance(item, RunFailedEvent) and item.depth == 0:
            return item.error
        return None

    @staticmethod
    def _format_full(item: AgentStreamItem) -> str | None:
        if isinstance(item, RunCompletedEvent):
            return _format_completed_run_for_full_mode(item)
        if isinstance(item, RunFailedEvent):
            return _format_failed_run_for_full_mode(item)
        if isinstance(item, StepCompletedEvent):
            return _format_step_for_full_mode(item)
        return None


def _format_completed_run_for_full_mode(event: RunCompletedEvent) -> str | None:
    if not event.response:
        return None
    if event.depth == 0:
        return event.response
    return f"[子Agent: {event.agent_id}] 完成:\n{event.response}"


def _format_failed_run_for_full_mode(event: RunFailedEvent) -> str:
    if event.depth == 0:
        return event.error
    return f"[子Agent: {event.agent_id}] 失败:\n{event.error}"


def _format_step_for_full_mode(event: StepCompletedEvent) -> str | None:
    """Format a StepCompletedEvent for full verbose output."""
    step = event.step
    prefix = f"[子Agent: {event.agent_id}] " if event.depth > 0 else ""

    if step.is_tool_step():
        tool_name = step.name or "tool"
        content = step.content_for_user or step.get_display_text()
        if not content:
            return None
        return f"{prefix}🛠 {tool_name}:\n{content}"

    if step.is_assistant_step():
        if step.tool_calls:
            return None
        content = step.get_display_text()
        if not content:
            return None
        return f"{prefix}{content}" if prefix else None

    return None
