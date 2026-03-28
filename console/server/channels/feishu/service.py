"""
Feishu channel service — concrete channel implementation.

Handles Feishu-specific concerns: long connection, message parsing, trigger
rules, ack/response delivery, and prompt rendering. Generic batching, agent
runtime, and scheduler logic live in the base channel infrastructure.
"""

import asyncio
import shutil
from typing import Any, Literal

from agiwo.agent import (
    AgentStreamItem,
    RunCompletedEvent,
    RunFailedEvent,
    StepCompletedEvent,
    UserMessage,
)
from agiwo.scheduler.scheduler import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.base import BaseChannelService, safe_close_all
from server.channels.exceptions import (
    BaseAgentNotFoundError,
    DefaultAgentNameNotFoundError,
    PreviousTaskRunningError,
)
from server.channels.feishu.factory import FeishuServiceFactory
from server.channels.feishu.message_parser import FeishuInboundEnvelope
from server.channels.session.models import BatchContext, InboundMessage
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry

logger = get_logger(__name__)


class FeishuChannelService(BaseChannelService):
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
        self._inbound_handler = components.inbound_handler

        super().__init__(
            session_service=components.session_service,
            agent_pool=components.agent_pool,
            executor=components.executor,
            workspace_conversation=components.workspace_conversation,
            debounce_ms=config.feishu_debounce_ms,
            max_batch_window_ms=config.feishu_max_batch_window_ms,
        )

        self._inbound_handler._session_mgr = self._session_mgr
        self._verbose_mode: Literal["full", "lite", "off"] = config.feishu_verbose_mode
        self._closed = False

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
        await self.close_base()
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
