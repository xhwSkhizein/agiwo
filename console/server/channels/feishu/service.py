"""
Feishu channel service — concrete channel implementation.

Handles Feishu-specific concerns: long connection, message parsing, trigger
rules, ack/response delivery, and prompt rendering. Generic batching, agent
runtime, and scheduler logic live in the base channel infrastructure.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any

from agiwo.agent import UserMessage
from agiwo.scheduler.scheduler import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.base import BaseChannelService
from server.channels.exceptions import (
    BaseAgentNotFoundError,
    DefaultAgentNameNotFoundError,
    PreviousTaskRunningError,
)
from server.channels.agent_executor import AgentExecutor
from server.channels.feishu.api_client import FeishuApiClient
from server.channels.feishu.commands import build_feishu_command_registry
from server.channels.feishu.content_extractor import FeishuContentExtractor
from server.channels.feishu.connection import FeishuConnection
from server.channels.feishu.delivery_service import FeishuDeliveryService
from server.channels.feishu.group_history_store import FeishuGroupHistoryStore
from server.channels.feishu.inbound_handler import FeishuInboundHandler
from server.channels.feishu.message_builder import (
    FeishuAttachmentResolver,
    FeishuUserMessageBuilder,
)
from server.channels.feishu.message_parser import (
    FeishuInboundEnvelope,
    FeishuMessageParser,
    FeishuSenderResolver,
)
from server.channels.feishu.store import create_feishu_channel_store
from server.channels.runtime_agent_pool import RuntimeAgentPool
from server.channels.session import SessionContextService
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
        self._bot_open_id = config.feishu_bot_open_id
        self._api = FeishuApiClient(
            app_id=config.feishu_app_id,
            app_secret=config.feishu_app_secret,
            api_base_url=config.feishu_api_base_url,
        )
        self._store = create_feishu_channel_store(
            db_path=config.sqlite_db_path,
            use_persistent_store=config.metadata_storage_type == "sqlite",
        )
        content_extractor = FeishuContentExtractor()
        group_history_store = FeishuGroupHistoryStore()
        sender_resolver = FeishuSenderResolver(api=self._api)
        self._parser = FeishuMessageParser(
            content_extractor=content_extractor,
            sender_resolver=sender_resolver,
            channel_instance_id=config.feishu_channel_instance_id,
            bot_open_id=self._bot_open_id,
        )
        self._connection = FeishuConnection(
            app_id=config.feishu_app_id,
            app_secret=config.feishu_app_secret,
            encrypt_key=config.feishu_encrypt_key,
            verification_token=config.feishu_verification_token,
            sdk_log_level=config.feishu_sdk_log_level,
        )

        session_service = SessionContextService(
            store=self._store,
            agent_registry=agent_registry,
            default_agent_name=config.feishu_default_agent_name,
        )
        agent_pool = RuntimeAgentPool(
            scheduler=scheduler,
            agent_registry=agent_registry,
            console_config=config,
            store=self._store,
        )
        executor = AgentExecutor(
            scheduler=scheduler,
            store=self._store,
            timeout=config.feishu_scheduler_wait_timeout,
        )
        super().__init__(
            session_service=session_service,
            agent_pool=agent_pool,
            executor=executor,
            debounce_ms=config.feishu_debounce_ms,
            max_batch_window_ms=config.feishu_max_batch_window_ms,
        )

        self._tmp_dir = Path(tempfile.mkdtemp(prefix="feishu_attachments_"))
        attachment_resolver = FeishuAttachmentResolver(
            api=self._api,
            tmp_dir=self._tmp_dir,
        )
        self._message_builder = FeishuUserMessageBuilder(
            content_extractor=content_extractor,
            group_history_store=group_history_store,
            attachment_resolver=attachment_resolver,
        )
        self._delivery_service = FeishuDeliveryService(
            api=self._api,
            config=config,
            truncate_for_log=self._truncate_for_log,
        )
        command_registry = build_feishu_command_registry(
            session_service=session_service,
            agent_pool=agent_pool,
            executor=executor,
            session_manager=self._session_mgr,
            scheduler=scheduler,
            agent_registry=agent_registry,
            console_config=config,
        )
        self._inbound_handler = FeishuInboundHandler(
            channel_instance_id=config.feishu_channel_instance_id,
            default_agent_name=config.feishu_default_agent_name,
            whitelist_open_ids=set(config.feishu_whitelist_open_ids),
            parser=self._parser,
            content_extractor=content_extractor,
            group_history_store=group_history_store,
            store=self._store,
            session_service=session_service,
            session_manager=self._session_mgr,
            command_registry=command_registry,
            delivery_service=self._delivery_service,
            truncate_for_log=self._truncate_for_log,
        )
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
        await self._connection.stop()
        await self._api.close()
        await self._store.close()
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
            return (
                f"默认 Agent '{error.agent_name}' 不存在或已被删除，请检查 "
                "AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME。"
            )
        if isinstance(error, DefaultAgentNameNotFoundError):
            return (
                f"当前默认 Agent 名称 '{error.agent_name}' 不存在，请检查 "
                "AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME。"
            )
        return f"执行失败: {str(error)}"
