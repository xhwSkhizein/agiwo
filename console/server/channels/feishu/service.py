"""
Feishu channel service — concrete channel implementation.

Handles Feishu-specific concerns: long connection, message parsing, trigger
rules, ack/response delivery, and prompt rendering, as well as the generic
batching, agent runtime, and scheduler message pipeline.
"""

import asyncio
import shutil
from collections.abc import AsyncIterator
from typing import Any

from agiwo.agent import (
    Agent,
    AgentStreamItem,
    UserMessage,
)
from agiwo.scheduler.engine import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.utils import (
    extract_stream_text,
    safe_close_all,
    split_text_into_chunks,
    truncate_for_log,
)
from server.channels.exceptions import (
    BaseAgentNotFoundError,
    DefaultAgentNameNotFoundError,
    PreviousTaskRunningError,
)
from server.channels.feishu.commands import build_feishu_command_registry
from server.channels.feishu.factory import FeishuServiceFactory
from server.channels.feishu.inbound_handler import FeishuInboundHandler
from server.channels.feishu.message_parser import FeishuInboundEnvelope
from server.channels.session import SessionManager
from server.config import ConsoleConfig
from server.models.session import (
    BatchContext,
    BatchPayload,
    InboundMessage,
    Session,
)
from server.services.agent_registry import AgentRegistry
from server.services.runtime import (
    AgentRuntimeCache,
    SessionContextService,
    SessionRuntimeService,
)

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
        feishu = config.channels.feishu
        self._session_mgr = SessionManager(
            on_batch_ready=self._on_batch_ready,
            debounce_ms=feishu.debounce_ms,
            max_batch_window_ms=feishu.max_batch_window_ms,
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
            channel_instance_id=feishu.channel_instance_id,
            default_agent_name=feishu.default_agent_name,
            whitelist_open_ids=set(feishu.whitelist_open_ids),
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
        self._closed = False

    @property
    def session_manager(self) -> SessionManager:
        return self._session_mgr

    @property
    def session_service(self) -> SessionContextService:
        return self._session_service

    @property
    def agent_pool(self) -> AgentRuntimeCache:
        return self._agent_pool

    @property
    def executor(self) -> SessionRuntimeService:
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

        had_output = await self._consume_dispatch_stream(
            batch,
            session,
            dispatch.stream,
        )
        if had_output:
            return

        if not await self._can_deliver_session(batch.context, session):
            return

        if dispatch.stream is None:
            await self._deliver_reply(batch.context, "消息已收到，正在继续处理。")
            return

        state = await self._executor.get_state(session.scheduler_state_id)
        if state is not None and state.result_summary:
            await self._deliver_stream_text(
                batch.context,
                state.result_summary,
                had_output=False,
            )
            return
        await self._deliver_reply(batch.context, "执行完成，但未产出可展示内容。")

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
            if not await self._can_deliver_session(batch.context, session):
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
        session: Session,
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
            current_session.id == session.id
            and current_session.scheduler_state_id == session.scheduler_state_id
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
