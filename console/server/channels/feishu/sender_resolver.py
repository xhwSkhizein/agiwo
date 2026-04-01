"""Feishu sender display-name resolution with a small in-memory cache."""

import time
from dataclasses import dataclass

from agiwo.utils.logging import get_logger

from server.channels.feishu.api_client import FeishuApiClient

logger = get_logger(__name__)

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


__all__ = ["FeishuSenderResolver"]
