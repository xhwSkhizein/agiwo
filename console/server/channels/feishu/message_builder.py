"""Feishu UserMessage construction: attachment resolution and channel context enrichment."""

import re
from pathlib import Path
from typing import Any

from agiwo.agent import ChannelContext, ContentPart, ContentType, UserMessage
from agiwo.utils.logging import get_logger

from server.channels.feishu.api_client import FeishuApiClient
from server.channels.feishu.content_extractor import FeishuContentExtractor
from server.channels.feishu.group_history_store import FeishuGroupHistoryStore
from server.channels.session.models import Attachment, BatchContext, InboundMessage

logger = get_logger(__name__)

_ATTACHMENT_PLACEHOLDER_RE = re.compile(r"\[(?:图片|文件|语音消息|视频|表情)[^\]]*\]")
_MAX_DOWNLOAD_SIZE = 20 * 1024 * 1024  # 20 MB

_ATTACHMENT_CONTENT_TYPE_MAP: dict[str, ContentType] = {
    "image": ContentType.IMAGE,
    "audio": ContentType.AUDIO,
    "media": ContentType.VIDEO,
    "file": ContentType.FILE,
}


# ── Attachment resolution ────────────────────────────────────────────────────


class FeishuAttachmentResolver:
    def __init__(
        self,
        *,
        api: FeishuApiClient,
        tmp_dir: Path,
    ) -> None:
        self._api = api
        self._tmp_dir = tmp_dir

    async def resolve_attachments(
        self,
        message: InboundMessage,
    ) -> list[ContentPart]:
        parts: list[ContentPart] = []
        for attachment in message.attachments:
            try:
                part = await self._resolve_single_attachment(message, attachment)
                if part is not None:
                    parts.append(part)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "feishu_attachment_resolve_failed",
                    key=attachment.key,
                    attachment_type=attachment.type,
                    error=str(exc),
                )
        return parts

    async def _resolve_single_attachment(
        self,
        message: InboundMessage,
        attachment: Attachment,
    ) -> ContentPart | None:
        if attachment.type == "sticker":
            return None

        data = await self._download_attachment(message, attachment)
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

        content_type = _ATTACHMENT_CONTENT_TYPE_MAP.get(
            attachment.type,
            ContentType.FILE,
        )
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

    async def _download_attachment(
        self,
        message: InboundMessage,
        attachment: Attachment,
    ) -> bytes:
        if attachment.type == "image":
            return await self._api.download_image(attachment.key)
        return await self._api.download_message_resource(
            message.message_id,
            attachment.key,
            "file",
        )


def _detect_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif data[:2] == b"\xff\xd8":
        mime = "image/jpeg"
    elif data[:4] == b"GIF8":
        mime = "image/gif"
    elif data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WEBP":
        mime = "image/webp"
    elif data[:4] == b"%PDF":
        mime = "application/pdf"
    elif len(data) > 8 and data[4:8] == b"ftyp":
        mime = "video/mp4"
    else:
        mime = "application/octet-stream"
    return mime


def _mime_to_ext(mime: str, original_name: str = "") -> str:
    if original_name:
        suffix = Path(original_name).suffix
        if suffix:
            return suffix

    ext_map = {
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
    return ext_map.get(mime, ".bin")


# ── UserMessage builder ──────────────────────────────────────────────────────


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
            "trigger_user_id": context.trigger_user_id,
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


__all__ = [
    "FeishuAttachmentResolver",
    "FeishuUserMessageBuilder",
]
