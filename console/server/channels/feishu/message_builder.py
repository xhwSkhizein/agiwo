"""Feishu UserMessage construction and channel context enrichment."""

import re
from typing import Any

from agiwo.agent.schema import ChannelContext, ContentPart, ContentType, UserMessage

from server.channels.feishu.attachment_resolver import FeishuAttachmentResolver
from server.channels.feishu.content_extractor import FeishuContentExtractor
from server.channels.feishu.group_history_store import FeishuGroupHistoryStore
from server.channels.models import BatchContext, InboundMessage

_ATTACHMENT_PLACEHOLDER_RE = re.compile(r"\[(?:图片|文件|语音消息|视频|表情)[^\]]*\]")


class FeishuUserMessageBuilder:
    def __init__(
        self,
        *,
        content_extractor: FeishuContentExtractor,
        group_history_store: FeishuGroupHistoryStore,
        attachment_resolver: FeishuAttachmentResolver,
    ) -> None:
        self._content_extractor = content_extractor
        self._group_history_store = group_history_store
        self._attachment_resolver = attachment_resolver

    async def build_user_message(
        self,
        context: BatchContext,
        messages: list[InboundMessage],
    ) -> UserMessage:
        latest = messages[-1]
        resolved_parts = await self._attachment_resolver.resolve_attachments(latest)

        text = self._content_extractor.normalize_message_text(latest.text)
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

        channel_context = ChannelContext(
            source="feishu",
            metadata=self._build_channel_metadata(context, messages, latest),
        )
        return UserMessage(content=content_parts, context=channel_context)

    def _build_channel_metadata(
        self,
        context: BatchContext,
        messages: list[InboundMessage],
        latest: InboundMessage,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "chat_type": context.chat_type,
            "chat_id": context.chat_id,
            "trigger_user": latest.sender_name,
            "batch_message_count": len(messages),
        }

        if context.chat_type == "p2p":
            dm_history = [
                self._content_extractor.normalize_message_text(message.text)
                for message in messages[:-1]
            ]
            dm_history = [line for line in dm_history if line]
            if dm_history:
                metadata["recent_dm_messages"] = dm_history[-5:]
            return metadata

        current_batch_message_ids = {message.message_id for message in messages}
        group_history = self._group_history_store.get_history_lines(
            context.chat_id,
            exclude_message_ids=current_batch_message_ids,
        )
        if group_history:
            metadata["recent_group_messages"] = group_history
        return metadata


def _clean_attachment_placeholders(text: str) -> str:
    return _ATTACHMENT_PLACEHOLDER_RE.sub("", text).strip()


__all__ = ["FeishuUserMessageBuilder"]
