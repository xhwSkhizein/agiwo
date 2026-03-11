"""Feishu outbound delivery helpers for ack, reply, and follow-up messages."""

from collections.abc import Callable

from agiwo.utils.logging import get_logger

from server.channels.feishu.api_client import FeishuApiClient
from server.channels.models import BatchContext, InboundMessage
from server.config import ConsoleConfig

logger = get_logger(__name__)


class FeishuDeliveryService:
    def __init__(
        self,
        *,
        api: FeishuApiClient,
        config: ConsoleConfig,
        truncate_for_log: Callable[[str], str],
    ) -> None:
        self._api = api
        self._config = config
        self._truncate_for_log = truncate_for_log

    async def send_ack(self, inbound: InboundMessage) -> None:
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
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu_ack_reaction_failed",
                message_id=inbound.message_id,
                error=str(exc),
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
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu_ack_fallback_failed",
                message_id=inbound.message_id,
                error=str(exc),
            )

    async def deliver_reply(self, context: BatchContext, text: str) -> None:
        final_text = self._format_group_reply(
            context.chat_type,
            context.trigger_user_id,
            text,
        )
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
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu_reply_failed_fallback_to_create",
                chat_id=context.chat_id,
                message_id=context.trigger_message_id,
                error=str(exc),
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

    async def deliver_message(self, context: BatchContext, text: str) -> None:
        final_text = self._format_group_reply(
            context.chat_type,
            context.trigger_user_id,
            text,
        )
        try:
            await self._api.create_text_message(context.chat_id, final_text)
            logger.info(
                "feishu_followup_sent",
                channel="feishu",
                chat_type=context.chat_type,
                chat_id=context.chat_id,
                text=self._truncate_for_log(final_text),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu_followup_send_failed",
                chat_id=context.chat_id,
                error=str(exc),
            )

    async def send_command_response(
        self,
        inbound: InboundMessage,
        text: str,
    ) -> None:
        reply_text = self._format_group_reply(
            inbound.chat_type,
            inbound.sender_id,
            text,
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
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu_command_response_failed",
                message_id=inbound.message_id,
                error=str(exc),
            )
            await self._api.create_text_message(inbound.chat_id, reply_text)

    def _format_group_reply(
        self,
        chat_type: str,
        user_id: str,
        text: str,
    ) -> str:
        if chat_type == "group":
            return f"<at user_id=\"{user_id}\">发起人</at> " + text
        return text


__all__ = ["FeishuDeliveryService"]
