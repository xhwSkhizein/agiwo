"""Feishu inbound payload handling, trigger rules, and command dispatch."""

from collections.abc import Callable

from agiwo.utils.logging import get_logger

from server.channels.exceptions import DefaultAgentNameNotFoundError
from server.channels.feishu.commands import CommandContext, CommandRegistry
from server.channels.feishu.content_extractor import FeishuContentExtractor
from server.channels.feishu.delivery_service import FeishuDeliveryService
from server.channels.feishu.group_history_store import FeishuGroupHistoryStore
from server.channels.feishu.message_parser import (
    FeishuInboundEnvelope,
    FeishuMessageParser,
)
from server.channels.feishu.store import FeishuChannelStoreBackend
from server.channels.batch_manager import ChannelBatchManager
from server.models.session import BatchContext, InboundMessage
from server.services.agent_registry import AgentConfigRecord
from server.services.runtime import SessionContextService

logger = get_logger(__name__)


class FeishuInboundHandler:
    def __init__(
        self,
        *,
        channel_instance_id: str,
        default_agent_name: str,
        whitelist_open_ids: set[str],
        parser: FeishuMessageParser,
        content_extractor: FeishuContentExtractor,
        group_history_store: FeishuGroupHistoryStore,
        store: FeishuChannelStoreBackend,
        session_service: SessionContextService,
        session_manager: ChannelBatchManager,
        command_registry: CommandRegistry,
        delivery_service: FeishuDeliveryService,
        truncate_for_log: Callable[[str], str],
    ) -> None:
        self._channel_instance_id = channel_instance_id
        self._default_agent_name = default_agent_name
        self._whitelist_open_ids = whitelist_open_ids
        self._parser = parser
        self._content_extractor = content_extractor
        self._group_history_store = group_history_store
        self._store = store
        self._session_service = session_service
        self._session_mgr = session_manager
        self._command_registry = command_registry
        self._delivery_service = delivery_service
        self._truncate_for_log = truncate_for_log

    async def process_envelope(  # noqa: PLR0911
        self, envelope: FeishuInboundEnvelope
    ) -> dict[str, object]:
        if envelope.event_type != "im.message.receive_v1":
            return {"msg": "ignored_non_message_event"}

        inbound = await self._parser.parse_inbound_message(envelope)
        if inbound is None:
            return {"msg": "ignored_invalid_payload"}

        self._log_message_received(inbound)

        claimed = await self._store.claim_event(
            inbound.channel_instance_id,
            inbound.event_id,
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

        self._group_history_store.record_message(
            inbound,
            normalized_text=self._content_extractor.normalize_message_text(
                inbound.text
            ),
        )
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

        # Resolve default agent before ACK/command to catch config errors early
        try:
            default_agent = await self._session_service.resolve_default_agent_config()
            if default_agent is None and self._default_agent_name:
                raise DefaultAgentNameNotFoundError(self._default_agent_name)
        except DefaultAgentNameNotFoundError:
            # Return user-facing error without ACKing
            return {
                "msg": "default_agent_not_found",
                "error": (
                    f"默认 Agent '{self._default_agent_name}' 不存在，"
                    "请检查 AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME 配置。"
                ),
            }

        command_result = await self._try_handle_command(inbound, default_agent)
        if command_result is not None:
            return command_result

        await self._delivery_service.send_ack(inbound)
        await self._enqueue_message(inbound, default_agent)
        return {"msg": "ok"}

    async def _enqueue_message(
        self,
        inbound: InboundMessage,
        default_agent: AgentConfigRecord | None,
    ) -> None:
        if default_agent is None:
            raise DefaultAgentNameNotFoundError(self._default_agent_name)
        chat_context_scope_id = self._build_chat_context_scope_id(inbound)
        context = BatchContext(
            chat_context_scope_id=chat_context_scope_id,
            channel_instance_id=self._channel_instance_id,
            chat_id=inbound.chat_id,
            chat_type=inbound.chat_type,
            trigger_user_id=inbound.sender_id,
            trigger_message_id=inbound.message_id,
            base_agent_id=default_agent.id,
        )
        await self._session_mgr.enqueue(chat_context_scope_id, inbound, context)

    async def _try_handle_command(
        self,
        inbound: InboundMessage,
        default_agent: AgentConfigRecord | None,
    ) -> dict[str, object] | None:
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

        ctx = await self._build_command_context(inbound, default_agent)
        try:
            result = await handler.execute(ctx, args)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "feishu_command_failed",
                command=handler.name,
                error=str(exc),
            )
            result_text = f"命令执行失败: {exc}"
        else:
            if result.is_post() and result.post_content:
                await self._delivery_service.send_command_response_post(
                    inbound, result.post_content
                )
                return {"msg": "command_executed"}
            result_text = result.text

        await self._delivery_service.send_command_response(inbound, result_text)
        return {"msg": "command_executed"}

    async def _build_command_context(
        self,
        inbound: InboundMessage,
        default_agent: AgentConfigRecord | None,
    ) -> CommandContext:
        chat_context_scope_id = self._build_chat_context_scope_id(inbound)
        (
            chat_context,
            current_session,
        ) = await self._session_service.get_chat_context_and_current_session(
            chat_context_scope_id
        )
        base_agent_id = default_agent.id if default_agent is not None else ""
        return CommandContext(
            chat_context_scope_id=chat_context_scope_id,
            channel_instance_id=self._channel_instance_id,
            chat_id=inbound.chat_id,
            chat_type=inbound.chat_type,
            trigger_user_open_id=inbound.sender_id,
            trigger_message_id=inbound.message_id,
            base_agent_id=base_agent_id,
            chat_context=chat_context,
            current_session=current_session,
        )

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

    def _build_chat_context_scope_id(self, inbound: InboundMessage) -> str:
        if inbound.chat_type == "p2p":
            return f"feishu:{self._channel_instance_id}:p2p:{inbound.sender_id}"
        return (
            f"feishu:{self._channel_instance_id}:group:{inbound.chat_id}:"
            f"user:{inbound.sender_id}"
        )

    def _log_message_received(self, inbound: InboundMessage) -> None:
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


__all__ = ["FeishuInboundHandler"]
