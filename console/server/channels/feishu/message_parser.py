"""Feishu inbound message parsing facade."""

from server.channels.feishu.content_extractor import FeishuContentExtractor
from server.channels.feishu.inbound_envelope import FeishuInboundEnvelope
from server.channels.feishu.sender_resolver import FeishuSenderResolver
from server.channels.models import InboundMessage


class FeishuMessageParser:
    def __init__(
        self,
        *,
        content_extractor: FeishuContentExtractor,
        sender_resolver: FeishuSenderResolver,
        channel_instance_id: str,
        bot_open_id: str,
    ) -> None:
        self._content_extractor = content_extractor
        self._sender_resolver = sender_resolver
        self._channel_instance_id = channel_instance_id
        self._bot_open_id = bot_open_id

    async def parse_inbound_message(
        self,
        envelope: FeishuInboundEnvelope,
    ) -> InboundMessage:
        parsed_content = self._content_extractor.extract(
            message_type=envelope.message_type,
            raw_content=envelope.content,
        )
        mentions = [mention.open_id for mention in envelope.mentions]
        is_at_bot = bool(self._bot_open_id) and self._bot_open_id in mentions

        return InboundMessage(
            channel_instance_id=self._channel_instance_id,
            event_id=envelope.event_id,
            message_id=envelope.message_id,
            chat_id=envelope.chat_id,
            chat_type=envelope.chat_type,
            sender_id=envelope.sender_open_id,
            sender_name=await self._sender_resolver.resolve_sender_name(
                envelope.sender_open_id
            ),
            text=parsed_content.text.strip(),
            event_time_ms=envelope.event_time_ms,
            raw_payload=envelope.to_payload_dict(),
            message_type=envelope.message_type,
            thread_id=envelope.thread_id,
            mentions=mentions,
            is_at_bot=is_at_bot,
            attachments=parsed_content.attachments,
        )


__all__ = ["FeishuMessageParser"]
