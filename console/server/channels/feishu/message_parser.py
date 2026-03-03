"""
Feishu message parsing, sender name resolution, and group history tracking.
"""

import json
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agiwo.utils.logging import get_logger

from server.channels.feishu.api_client import FeishuApiClient
from server.channels.models import Attachment, InboundMessage

logger = get_logger(__name__)

_TEXT_MENTION_PATTERN = re.compile(r"<at[^>]*>.*?</at>", re.IGNORECASE)
_GROUP_HISTORY_MAX_ITEMS = 20
_GROUP_HISTORY_WINDOW_MS = 15 * 60 * 1000
_SENDER_NAME_CACHE_TTL_SECONDS = 3600


# ---------------------------------------------------------------------------
# Content extraction — one extractor per message_type
# ---------------------------------------------------------------------------

@dataclass
class ParsedContent:
    text: str
    attachments: list[Attachment] = field(default_factory=list)


def _extract_text(content: dict[str, Any]) -> ParsedContent:
    return ParsedContent(text=content.get("text", ""))


def _extract_post(content: dict[str, Any]) -> ParsedContent:
    """Extract plain text and attachments from Feishu post (rich text).

    Structure: {"zh_cn": {"title": "...", "content": [[{tag, ...}, ...]]}}
    """
    post: dict[str, Any] | None = None
    for locale in ("zh_cn", "en_us", "ja_jp"):
        post = content.get(locale)
        if post is not None:
            break
    if post is None:
        for val in content.values():
            if isinstance(val, dict) and "content" in val:
                post = val
                break
    if post is None and "content" in content and isinstance(content["content"], list):
        post = content
    if post is None:
        return ParsedContent(text="")

    parts: list[str] = []
    attachments: list[Attachment] = []

    title = post.get("title", "")
    if title:
        parts.append(title)

    for paragraph in post.get("content", []):
        if not isinstance(paragraph, list):
            continue
        line_parts: list[str] = []
        for element in paragraph:
            if not isinstance(element, dict):
                continue
            tag = element.get("tag")
            if tag == "text":
                line_parts.append(element.get("text", ""))
            elif tag == "a":
                href = element.get("href", "")
                link_text = element.get("text", "")
                if link_text and href:
                    line_parts.append(f"{link_text}({href})")
                else:
                    line_parts.append(href or link_text)
            elif tag == "at":
                user_name = element.get("user_name", "")
                if user_name:
                    line_parts.append(f"@{user_name}")
            elif tag == "img":
                image_key = element.get("image_key", "")
                attachments.append(Attachment(type="image", key=image_key))
                line_parts.append(f"[图片 image_key={image_key}]")
            elif tag == "media":
                file_key = element.get("file_key", "")
                attachments.append(Attachment(type="media", key=file_key))
                line_parts.append(f"[视频 file_key={file_key}]")
            elif tag == "emotion":
                emoji_type = element.get("emoji_type", "")
                if emoji_type:
                    line_parts.append(f"[{emoji_type}]")
        if line_parts:
            parts.append("".join(line_parts))

    return ParsedContent(text="\n".join(parts), attachments=attachments)


def _extract_image(content: dict[str, Any]) -> ParsedContent:
    image_key = content.get("image_key", "")
    return ParsedContent(
        text=f"[图片 image_key={image_key}]",
        attachments=[Attachment(type="image", key=image_key)],
    )


def _extract_file(content: dict[str, Any]) -> ParsedContent:
    file_key = content.get("file_key", "")
    file_name = content.get("file_name", "")
    return ParsedContent(
        text=f"[文件: {file_name or file_key} (file_key={file_key})]",
        attachments=[Attachment(type="file", key=file_key, name=file_name)],
    )


def _extract_audio(content: dict[str, Any]) -> ParsedContent:
    file_key = content.get("file_key", "")
    return ParsedContent(
        text=f"[语音消息 file_key={file_key}]",
        attachments=[Attachment(type="audio", key=file_key)],
    )


def _extract_media(content: dict[str, Any]) -> ParsedContent:
    file_key = content.get("file_key", "")
    return ParsedContent(
        text=f"[视频 file_key={file_key}]",
        attachments=[Attachment(type="media", key=file_key)],
    )


def _extract_sticker(content: dict[str, Any]) -> ParsedContent:
    file_key = content.get("file_key", "")
    return ParsedContent(
        text=f"[表情 file_key={file_key}]",
        attachments=[Attachment(type="sticker", key=file_key)],
    )


_CONTENT_EXTRACTORS: dict[str, Callable[[dict[str, Any]], ParsedContent]] = {
    "text": _extract_text,
    "post": _extract_post,
    "image": _extract_image,
    "file": _extract_file,
    "audio": _extract_audio,
    "media": _extract_media,
    "sticker": _extract_sticker,
    "interactive": lambda _: ParsedContent(text="[卡片消息]"),
    "share_chat": lambda _: ParsedContent(text="[分享群组]"),
    "share_user": lambda _: ParsedContent(text="[分享用户]"),
}


def extract_content(message_type: str, content: dict[str, Any]) -> ParsedContent:
    extractor = _CONTENT_EXTRACTORS.get(message_type)
    if extractor is not None:
        return extractor(content)
    return ParsedContent(text=f"[{message_type}]")


@dataclass
class _GroupRecentMessage:
    message_id: str
    event_time_ms: int
    sender_name: str
    text: str


@dataclass
class _SenderNameCacheEntry:
    display_name: str
    expire_at: float


class FeishuMessageParser:
    def __init__(
        self,
        *,
        api: FeishuApiClient,
        channel_instance_id: str,
        bot_open_id: str,
    ) -> None:
        self._api = api
        self._channel_instance_id = channel_instance_id
        self._bot_open_id = bot_open_id

        self._sender_name_cache: dict[str, _SenderNameCacheEntry] = {}
        self._group_recent_messages: dict[str, deque[_GroupRecentMessage]] = {}

    def extract_event_type(self, payload: dict[str, Any]) -> str:
        header = payload.get("header") or {}
        event_type = header.get("event_type")
        if isinstance(event_type, str) and event_type:
            return event_type

        raw_type = payload.get("type")
        if isinstance(raw_type, str):
            return raw_type
        return ""

    async def parse_inbound_message(
        self,
        payload: dict[str, Any],
    ) -> InboundMessage | None:
        event = payload.get("event") or {}
        message = event.get("message") or {}
        sender = event.get("sender") or {}

        message_id = message.get("message_id")
        chat_id = message.get("chat_id")
        chat_type = message.get("chat_type")
        if not isinstance(message_id, str) or not isinstance(chat_id, str) or not isinstance(chat_type, str):
            return None

        message_type = message.get("message_type") or "text"

        sender_id_obj = sender.get("sender_id") or {}
        sender_open_id = sender_id_obj.get("open_id")
        if not isinstance(sender_open_id, str) or not sender_open_id:
            return None

        raw_content = message.get("content")
        parsed_content = ParsedContent(text="")
        if isinstance(raw_content, str) and raw_content:
            try:
                content_json = json.loads(raw_content)
                parsed_content = extract_content(message_type, content_json)
            except json.JSONDecodeError:
                parsed_content = ParsedContent(text=raw_content)

        mentions: list[str] = []
        raw_mentions = message.get("mentions") or []
        if isinstance(raw_mentions, list):
            for item in raw_mentions:
                if not isinstance(item, dict):
                    continue
                mention_id = item.get("id") or {}
                if not isinstance(mention_id, dict):
                    continue
                mention_open_id = mention_id.get("open_id")
                if isinstance(mention_open_id, str) and mention_open_id:
                    mentions.append(mention_open_id)

        is_at_bot = bool(self._bot_open_id) and self._bot_open_id in mentions

        header = payload.get("header") or {}
        event_id = header.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            event_id = message_id

        create_time = message.get("create_time")
        try:
            if isinstance(create_time, int | str):
                event_time_ms = int(create_time)
            else:
                event_time_ms = int(time.time() * 1000)
        except (TypeError, ValueError):
            event_time_ms = int(time.time() * 1000)

        return InboundMessage(
            channel_instance_id=self._channel_instance_id,
            event_id=event_id,
            message_id=message_id,
            chat_id=chat_id,
            chat_type=chat_type,
            sender_id=sender_open_id,
            sender_name=await self.resolve_sender_name(sender_open_id),
            text=parsed_content.text.strip(),
            event_time_ms=event_time_ms,
            raw_payload=payload,
            message_type=message_type,
            thread_id=message.get("thread_id"),
            mentions=mentions,
            is_at_bot=is_at_bot,
            attachments=parsed_content.attachments,
        )

    async def resolve_sender_name(self, sender_open_id: str) -> str:
        now = time.time()
        cached = self._sender_name_cache.get(sender_open_id)
        if cached is not None and now < cached.expire_at:
            return cached.display_name

        fallback = self._format_sender_name(sender_open_id)
        try:
            display_name = await self._api.get_user_display_name(sender_open_id)
            if display_name is None:
                return fallback

            self._sender_name_cache[sender_open_id] = _SenderNameCacheEntry(
                display_name=display_name,
                expire_at=now + _SENDER_NAME_CACHE_TTL_SECONDS,
            )
            return display_name
        except Exception as e:
            logger.warning(
                "feishu_resolve_sender_name_failed",
                sender_open_id=sender_open_id,
                error=str(e),
            )
            return fallback

    def normalize_message_text(self, text: str) -> str:
        clean = _TEXT_MENTION_PATTERN.sub("", text or "").strip()
        clean = re.sub(r"\s+", " ", clean)
        return clean

    def record_group_message(self, message: InboundMessage) -> None:
        if message.chat_type != "group":
            return

        clean_text = self.normalize_message_text(message.text)
        if not clean_text:
            return

        history = self._group_recent_messages.setdefault(
            message.chat_id, deque(),
        )
        history.append(
            _GroupRecentMessage(
                message_id=message.message_id,
                event_time_ms=message.event_time_ms,
                sender_name=message.sender_name,
                text=clean_text,
            )
        )

        cutoff_ms = int(time.time() * 1000) - _GROUP_HISTORY_WINDOW_MS
        while history and (
            len(history) > _GROUP_HISTORY_MAX_ITEMS
            or history[0].event_time_ms < cutoff_ms
        ):
            history.popleft()

    def get_group_history_lines(
        self,
        chat_id: str,
        *,
        exclude_message_ids: set[str],
    ) -> list[str]:
        history = self._group_recent_messages.get(chat_id)
        if not history:
            return []

        cutoff_ms = int(time.time() * 1000) - _GROUP_HISTORY_WINDOW_MS
        lines: list[str] = []
        for item in history:
            if item.event_time_ms < cutoff_ms:
                continue
            if item.message_id in exclude_message_ids:
                continue
            lines.append(f"{item.sender_name}: {item.text}")

        return lines[-_GROUP_HISTORY_MAX_ITEMS:]

    def _format_sender_name(self, sender_open_id: str) -> str:
        normalized = sender_open_id.strip()
        if not normalized:
            return "user"
        suffix = normalized[-6:] if len(normalized) >= 6 else normalized
        return f"user_{suffix}"
