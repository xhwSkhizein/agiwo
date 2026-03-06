"""
Feishu channel service — concrete channel implementation.

Handles Feishu-specific concerns: long connection, message parsing, trigger
rules, ack/response delivery, and prompt rendering. Generic batching, agent
runtime, and scheduler logic live in the base channel infrastructure.
"""

import asyncio
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from agiwo.agent.schema import ChannelContext, ContentPart, ContentType, UserMessage
from agiwo.scheduler.scheduler import Scheduler
from agiwo.utils.logging import get_logger

from server.channels.agent_runtime import AgentRuntimeManager
from server.channels.base import BaseChannelService
from server.channels.feishu.api_client import FeishuApiClient
from server.channels.feishu.commands import (
    CommandContext,
    CommandRegistry,
    ContextCommand,
    HelpCommand,
    ListSessionsCommand,
    NewSessionCommand,
    StatusCommand,
)
from server.channels.feishu.connection import FeishuConnection
from server.channels.feishu.message_parser import FeishuMessageParser
from server.channels.feishu.store import FeishuChannelStore
from server.channels.models import Attachment, BatchContext, InboundMessage
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry

logger = get_logger(__name__)

_MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024  # 20 MB

_ATTACHMENT_CONTENT_TYPE_MAP: dict[str, ContentType] = {
    "image": ContentType.IMAGE,
    "audio": ContentType.AUDIO,
    "media": ContentType.VIDEO,
    "file": ContentType.FILE,
}

_ATTACHMENT_PLACEHOLDER_RE = re.compile(r'\[(?:图片|文件|语音消息|视频|表情)[^\]]*\]')


def _detect_mime(data: bytes) -> str:
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if data[:2] == b'\xff\xd8':
        return "image/jpeg"
    if data[:4] == b'GIF8':
        return "image/gif"
    if data[:4] == b'RIFF' and len(data) >= 12 and data[8:12] == b'WEBP':
        return "image/webp"
    if data[:4] == b'%PDF':
        return "application/pdf"
    if len(data) > 8 and data[4:8] == b'ftyp':
        return "video/mp4"
    return "application/octet-stream"


def _mime_to_ext(mime: str, original_name: str = "") -> str:
    if original_name:
        suffix = Path(original_name).suffix
        if suffix:
            return suffix
    _map = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "application/pdf": ".pdf",
        "video/mp4": ".mp4",
        "audio/mp4": ".m4a",
        "audio/mpeg": ".mp3",
        "audio/ogg": ".ogg",
    }
    return _map.get(mime, ".bin")


def _clean_attachment_placeholders(text: str) -> str:
    return _ATTACHMENT_PLACEHOLDER_RE.sub("", text).strip()


class FeishuChannelService(BaseChannelService):
    def __init__(
        self,
        *,
        config: ConsoleConfig,
        scheduler: Scheduler,
        agent_registry: AgentRegistry,
    ) -> None:
        self._config = config
        self._channel_instance_id = config.feishu_channel_instance_id
        self._default_agent_name = config.feishu_default_agent_name
        self._bot_open_id = config.feishu_bot_open_id
        self._whitelist_open_ids = set(config.feishu_whitelist_open_ids)

        self._api = FeishuApiClient(
            app_id=config.feishu_app_id,
            app_secret=config.feishu_app_secret,
            api_base_url=config.feishu_api_base_url,
        )
        self._store = FeishuChannelStore(
            db_path=config.sqlite_db_path,
            use_persistent_store=config.metadata_storage_type == "sqlite",
        )
        self._parser = FeishuMessageParser(
            api=self._api,
            channel_instance_id=self._channel_instance_id,
            bot_open_id=self._bot_open_id,
        )
        self._connection = FeishuConnection(
            app_id=config.feishu_app_id,
            app_secret=config.feishu_app_secret,
            encrypt_key=config.feishu_encrypt_key,
            verification_token=config.feishu_verification_token,
            sdk_log_level=config.feishu_sdk_log_level,
        )

        runtime_mgr = AgentRuntimeManager(
            scheduler=scheduler,
            agent_registry=agent_registry,
            console_config=config,
            store=self._store,
            default_agent_name=self._default_agent_name,
            scheduler_wait_timeout=config.feishu_scheduler_wait_timeout,
        )

        super().__init__(
            runtime_mgr=runtime_mgr,
            debounce_ms=config.feishu_debounce_ms,
            max_batch_window_ms=config.feishu_max_batch_window_ms,
        )

        self._command_registry = self._build_command_registry(scheduler)
        self._closed = False
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="feishu_attachments_"))

    async def initialize(self) -> None:
        await self._store.connect()

        if not self._bot_open_id:
            logger.warning(
                "feishu_bot_open_id_not_configured",
                detail="group messages will not trigger without AGIWO_CONSOLE_FEISHU_BOT_OPEN_ID",
            )

        main_loop = asyncio.get_running_loop()
        await self._connection.start(main_loop, self._process_incoming_payload)

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

    # -- Incoming message pipeline -------------------------------------------

    async def _process_incoming_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if self._closed:
            return {"msg": "feishu_channel_closed"}

        event_type = self._parser.extract_event_type(payload)
        if event_type != "im.message.receive_v1":
            return {"msg": "ignored_non_message_event"}

        inbound = await self._parser.parse_inbound_message(payload)
        if inbound is None:
            return {"msg": "ignored_invalid_payload"}

        logger.info(
            "feishu_message_received",
            channel="feishu",
            chat_type=inbound.chat_type,
            chat_id=inbound.chat_id,
            message_id=inbound.message_id,
            message_type=inbound.message_type,
            sender=inbound.sender_name,
            is_at_bot=inbound.is_at_bot,
            text=self._truncate_for_log(inbound.text),
        )

        claimed = await self._store.claim_event(
            inbound.channel_instance_id, inbound.event_id,
        )
        if not claimed:
            logger.info(
                "feishu_message_ignored",
                channel="feishu",
                reason="duplicate",
                chat_type=inbound.chat_type,
                chat_id=inbound.chat_id,
                message_id=inbound.message_id,
            )
            return {"msg": "ignored_duplicate"}

        self._parser.record_group_message(inbound)

        if not self._should_trigger(inbound):
            logger.info(
                "feishu_message_ignored",
                channel="feishu",
                reason="not_trigger",
                chat_type=inbound.chat_type,
                chat_id=inbound.chat_id,
                message_id=inbound.message_id,
                sender=inbound.sender_name,
            )
            return {"msg": "ignored_not_trigger"}

        command_result = await self._try_handle_command(inbound)
        if command_result is not None:
            return command_result

        await self._send_ack(inbound)
        await self._enqueue_message(inbound)
        return {"msg": "ok"}

    # -- Trigger rules -------------------------------------------------------

    def _should_trigger(self, inbound: InboundMessage) -> bool:
        if not self._default_agent_name:
            return False
        if not self._is_whitelisted(inbound.sender_id):
            return False
        if inbound.chat_type == "group":
            return inbound.is_at_bot
        if inbound.chat_type == "p2p":
            return True
        return False

    def _is_whitelisted(self, sender_id: str) -> bool:
        if not self._whitelist_open_ids:
            return True
        return sender_id in self._whitelist_open_ids

    # -- Ack -----------------------------------------------------------------

    async def _send_ack(self, inbound: InboundMessage) -> None:
        try:
            await self._api.add_message_reaction(
                inbound.message_id,
                self._config.feishu_ack_reaction_emoji,
            )
            logger.info(
                "feishu_ack_sent",
                channel="feishu",
                mode="reaction",
                chat_type=inbound.chat_type,
                chat_id=inbound.chat_id,
                message_id=inbound.message_id,
            )
            return
        except Exception as reaction_error:
            logger.warning(
                "feishu_ack_reaction_failed",
                message_id=inbound.message_id,
                error=str(reaction_error),
            )

        try:
            await self._api.reply_text(
                inbound.message_id,
                self._config.feishu_ack_fallback_text,
            )
            logger.info(
                "feishu_ack_sent",
                channel="feishu",
                mode="reply_fallback",
                chat_type=inbound.chat_type,
                chat_id=inbound.chat_id,
                message_id=inbound.message_id,
                text=self._truncate_for_log(self._config.feishu_ack_fallback_text),
            )
        except Exception as fallback_error:
            logger.warning(
                "feishu_ack_fallback_failed",
                message_id=inbound.message_id,
                error=str(fallback_error),
            )

    # -- Enqueue into session manager ----------------------------------------

    async def _enqueue_message(self, inbound: InboundMessage) -> None:
        default_agent = await self._runtime_mgr.resolve_default_agent_config()
        if default_agent is None:
            raise RuntimeError(
                f"default_agent_name_not_found: {self._default_agent_name}"
            )

        session_key = self._build_session_key(inbound)
        context = BatchContext(
            session_key=session_key,
            chat_id=inbound.chat_id,
            chat_type=inbound.chat_type,
            trigger_user_id=inbound.sender_id,
            trigger_message_id=inbound.message_id,
            base_agent_id=default_agent.id,
        )

        await self._session_mgr.enqueue(session_key, inbound, context)

    def _build_session_key(self, inbound: InboundMessage) -> str:
        if inbound.chat_type == "p2p":
            return f"feishu:{self._channel_instance_id}:dm:{inbound.sender_id}"
        return (
            f"feishu:{self._channel_instance_id}:group:{inbound.chat_id}:"
            f"user:{inbound.sender_id}"
        )

    # -- BaseChannelService hooks --------------------------------------------

    async def _build_user_message(
        self,
        context: BatchContext,
        messages: list[InboundMessage],
    ) -> UserMessage:
        latest = messages[-1]

        # Resolve attachments: download to local tmp dir and build ContentParts
        resolved_parts = await self._resolve_attachments(latest)

        # Build text content, cleaning placeholder tokens for resolved attachments
        text = self._parser.normalize_message_text(latest.text)
        if resolved_parts:
            text = _clean_attachment_placeholders(text)
        if not text:
            text = latest.text.strip()
        if not text and not resolved_parts:
            text = "请根据上下文处理用户请求。"

        content_parts: list[ContentPart] = []
        if text:
            content_parts.append(ContentPart(type=ContentType.TEXT, text=text))
        content_parts.extend(resolved_parts)

        # Build channel context (replaces <system_notice> string concatenation)
        metadata: dict[str, Any] = {
            "chat_type": context.chat_type,
            "chat_id": context.chat_id,
            "trigger_user": latest.sender_name,
            "batch_message_count": len(messages),
        }

        if context.chat_type == "p2p":
            dm_history = [
                self._parser.normalize_message_text(msg.text)
                for msg in messages[:-1]
            ]
            dm_history = [line for line in dm_history if line]
            if dm_history:
                metadata["recent_dm_messages"] = dm_history[-5:]
        else:
            current_batch_message_ids = {msg.message_id for msg in messages}
            group_history = self._parser.get_group_history_lines(
                context.chat_id,
                exclude_message_ids=current_batch_message_ids,
            )
            if group_history:
                metadata["recent_group_messages"] = group_history

        channel_context = ChannelContext(source="feishu", metadata=metadata)
        return UserMessage(content=content_parts, context=channel_context)

    # -- Attachment resolution -----------------------------------------------

    async def _resolve_attachments(
        self, message: InboundMessage,
    ) -> list[ContentPart]:
        parts: list[ContentPart] = []
        for attachment in message.attachments:
            try:
                part = await self._resolve_single_attachment(message, attachment)
                if part is not None:
                    parts.append(part)
            except Exception as e:
                logger.warning(
                    "feishu_attachment_resolve_failed",
                    key=attachment.key,
                    attachment_type=attachment.type,
                    error=str(e),
                )
        return parts

    async def _resolve_single_attachment(
        self,
        message: InboundMessage,
        attachment: Attachment,
    ) -> ContentPart | None:
        if attachment.type == "sticker":
            return None

        if attachment.type == "image":
            data = await self._api.download_image(attachment.key)
        else:
            data = await self._api.download_message_resource(
                message.message_id, attachment.key, "file"
            )

        if len(data) > _MAX_DOWNLOAD_SIZE:
            logger.warning(
                "feishu_attachment_too_large",
                key=attachment.key,
                size=len(data),
                limit=_MAX_DOWNLOAD_SIZE,
            )
            return None

        mime = _detect_mime(data)
        ext = _mime_to_ext(mime, attachment.name)
        filename = f"{attachment.key}{ext}"
        local_path = self._tmp_dir / filename
        local_path.write_bytes(data)

        content_type = _ATTACHMENT_CONTENT_TYPE_MAP.get(attachment.type, ContentType.FILE)
        return ContentPart(
            type=content_type,
            url=str(local_path),
            mime_type=mime,
            metadata={
                "name": attachment.name or filename,
                "size": len(data),
                "source": "feishu",
            },
        )

    async def _deliver_reply(self, context: BatchContext, text: str) -> None:
        final_text = self._prepend_at_tag(context, text)

        try:
            await self._api.reply_text(context.trigger_message_id, final_text)
            logger.info(
                "feishu_response_sent",
                channel="feishu",
                delivery="reply",
                chat_type=context.chat_type,
                chat_id=context.chat_id,
                trigger_message_id=context.trigger_message_id,
                text=self._truncate_for_log(final_text),
            )
        except Exception as reply_error:
            logger.warning(
                "feishu_reply_failed_fallback_to_create",
                chat_id=context.chat_id,
                message_id=context.trigger_message_id,
                error=str(reply_error),
            )
            await self._api.create_text_message(context.chat_id, final_text)
            logger.info(
                "feishu_response_sent",
                channel="feishu",
                delivery="create_message",
                chat_type=context.chat_type,
                chat_id=context.chat_id,
                trigger_message_id=context.trigger_message_id,
                text=self._truncate_for_log(final_text),
            )

    async def _deliver_message(self, context: BatchContext, text: str) -> None:
        final_text = self._prepend_at_tag(context, text)

        try:
            await self._api.create_text_message(context.chat_id, final_text)
            logger.info(
                "feishu_followup_sent",
                channel="feishu",
                chat_type=context.chat_type,
                chat_id=context.chat_id,
                text=self._truncate_for_log(final_text),
            )
        except Exception as e:
            logger.warning(
                "feishu_followup_send_failed",
                chat_id=context.chat_id,
                error=str(e),
            )

    def _to_user_facing_error(self, error: Exception) -> str:
        raw = str(error)
        if raw == "previous_task_still_running_after_timeout":
            return "上一条任务仍在处理中，请稍后再试。"
        if raw.startswith("base_agent_not_found:"):
            return "默认 Agent 不存在或已被删除，请检查 AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME。"
        if raw.startswith("default_agent_name_not_found:"):
            return "当前默认 Agent 名称不存在，请检查 AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME。"
        return f"执行失败: {raw}"

    # -- Commands ------------------------------------------------------------

    def _build_command_registry(self, scheduler: Scheduler) -> CommandRegistry:
        registry = CommandRegistry()
        registry.register(
            NewSessionCommand(
                self._store,
                scheduler,
                self._runtime_mgr.runtime_agents,
                self._session_mgr,
            )
        )
        registry.register(ListSessionsCommand(self._store, scheduler))
        registry.register(ContextCommand(self._runtime_mgr.runtime_agents))
        registry.register(StatusCommand(self._runtime_mgr.runtime_agents, scheduler))
        registry.register(HelpCommand(registry))
        return registry

    async def _try_handle_command(
        self,
        inbound: InboundMessage,
    ) -> dict[str, Any] | None:
        parsed = self._command_registry.try_parse(inbound.text)
        if parsed is None:
            return None

        handler, args = parsed
        logger.info(
            "feishu_command_received",
            channel="feishu",
            command=handler.name,
            args=args,
            chat_id=inbound.chat_id,
            sender=inbound.sender_name,
        )

        default_agent = await self._runtime_mgr.resolve_default_agent_config()
        base_agent_id = default_agent.id if default_agent else ""
        session_key = self._build_session_key(inbound)
        runtime = await self._store.get_session_runtime(session_key)

        ctx = CommandContext(
            session_key=session_key,
            chat_id=inbound.chat_id,
            chat_type=inbound.chat_type,
            trigger_user_open_id=inbound.sender_id,
            trigger_message_id=inbound.message_id,
            base_agent_id=base_agent_id,
            runtime=runtime,
        )

        try:
            result = await handler.execute(ctx, args)
        except Exception as e:
            logger.exception(
                "feishu_command_failed",
                command=handler.name,
                error=str(e),
            )
            result_text = f"命令执行失败: {e}"
        else:
            result_text = result.text

        await self._send_command_response(inbound, result_text)
        return {"msg": "command_executed"}

    async def _send_command_response(
        self,
        inbound: InboundMessage,
        text: str,
    ) -> None:
        reply_text = text
        if inbound.chat_type == "group":
            reply_text = (
                f"<at user_id=\"{inbound.sender_id}\">发起人</at> " + text
            )
        try:
            await self._api.reply_text(inbound.message_id, reply_text)
            logger.info(
                "feishu_command_response_sent",
                channel="feishu",
                chat_id=inbound.chat_id,
                message_id=inbound.message_id,
                text=self._truncate_for_log(reply_text),
            )
        except Exception as e:
            logger.warning(
                "feishu_command_response_failed",
                message_id=inbound.message_id,
                error=str(e),
            )
            await self._api.create_text_message(inbound.chat_id, reply_text)

    # -- Helpers -------------------------------------------------------------

    def _prepend_at_tag(self, context: BatchContext, text: str) -> str:
        if context.chat_type == "group":
            return (
                f"<at user_id=\"{context.trigger_user_id}\">发起人</at> " + text
            )
        return text
