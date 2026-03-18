"""Feishu inbound message parsing: envelope types, sender resolution, and parsing facade."""

import time
from dataclasses import dataclass, field

from agiwo.utils.logging import get_logger

from server.channels.feishu.api_client import FeishuApiClient
from server.channels.feishu.content_extractor import FeishuContentExtractor
from server.channels.session.models import InboundMessage

logger = get_logger(__name__)

# ── Inbound envelope types ──────────────────────────────────────────────────


@dataclass(frozen=True)
class FeishuMention:
    open_id: str


@dataclass(frozen=True)
class FeishuInboundEnvelope:
    event_type: str
    event_id: str
    message_id: str
    chat_id: str
    chat_type: str
    sender_open_id: str
    message_type: str
    content: str
    event_time_ms: int
    thread_id: str | None = None
    mentions: tuple[FeishuMention, ...] = field(default_factory=tuple)

    def to_payload_dict(self) -> dict[str, object]:
        return {
            "header": {
                "event_id": self.event_id,
                "event_type": self.event_type,
                "token": "",
            },
            "event": {
                "message": {
                    "message_id": self.message_id,
                    "chat_id": self.chat_id,
                    "chat_type": self.chat_type,
                    "thread_id": self.thread_id,
                    "message_type": self.message_type,
                    "content": self.content,
                    "mentions": [
                        {"id": {"open_id": mention.open_id}}
                        for mention in self.mentions
                    ],
                    "create_time": self.event_time_ms,
                },
                "sender": {
                    "sender_id": {
                        "open_id": self.sender_open_id,
                    }
                },
            },
        }


# ── Sender name resolution ──────────────────────────────────────────────────

_SENDER_NAME_CACHE_TTL_SECONDS = 3600


@dataclass
class _SenderNameCacheEntry:
    display_name: str
    expire_at: float


class FeishuSenderResolver:
    def __init__(
        self,
        *,
        api: FeishuApiClient,
        cache_ttl_seconds: int = _SENDER_NAME_CACHE_TTL_SECONDS,
    ) -> None:
        self._api = api
        self._cache_ttl_seconds = cache_ttl_seconds
        self._sender_name_cache: dict[str, _SenderNameCacheEntry] = {}

    async def resolve_sender_name(self, sender_open_id: str) -> str:
        now = time.time()
        cached = self._sender_name_cache.get(sender_open_id)
        if cached is not None and now < cached.expire_at:
            return cached.display_name

        fallback = self._format_sender_name(sender_open_id)
        try:
            display_name = await self._api.get_user_display_name(sender_open_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu_resolve_sender_name_failed",
                sender_open_id=sender_open_id,
                error=str(exc),
            )
            return fallback

        if display_name is None:
            return fallback

        self._sender_name_cache[sender_open_id] = _SenderNameCacheEntry(
            display_name=display_name,
            expire_at=now + self._cache_ttl_seconds,
        )
        return display_name

    def _format_sender_name(self, sender_open_id: str) -> str:
        normalized = sender_open_id.strip()
        if not normalized:
            return "user"
        suffix = normalized[-6:] if len(normalized) >= 6 else normalized
        return f"user_{suffix}"


# ── Message parser ───────────────────────────────────────────────────────────


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


__all__ = [
    "FeishuInboundEnvelope",
    "FeishuMention",
    "FeishuMessageParser",
    "FeishuSenderResolver",
]
