"""Recent Feishu group-message history used for prompt context."""

import time
from collections import deque
from dataclasses import dataclass

from server.models.session import InboundMessage

_GROUP_HISTORY_MAX_ITEMS = 20
_GROUP_HISTORY_WINDOW_MS = 15 * 60 * 1000
_GROUP_HISTORY_MAX_GROUPS = 200


@dataclass
class _GroupRecentMessage:
    message_id: str
    event_time_ms: int
    sender_name: str
    text: str


class FeishuGroupHistoryStore:
    def __init__(
        self,
        *,
        max_items: int = _GROUP_HISTORY_MAX_ITEMS,
        window_ms: int = _GROUP_HISTORY_WINDOW_MS,
        max_groups: int = _GROUP_HISTORY_MAX_GROUPS,
    ) -> None:
        self._max_items = max_items
        self._window_ms = window_ms
        self._max_groups = max_groups
        self._group_recent_messages: dict[str, deque[_GroupRecentMessage]] = {}

    def record_message(
        self,
        message: InboundMessage,
        *,
        normalized_text: str,
    ) -> None:
        if message.chat_type != "group" or not normalized_text:
            return

        history = self._group_recent_messages.setdefault(message.chat_id, deque())
        history.append(
            _GroupRecentMessage(
                message_id=message.message_id,
                event_time_ms=message.event_time_ms,
                sender_name=message.sender_name,
                text=normalized_text,
            )
        )
        self._trim_history(history)
        self._evict_stale_groups()

    def get_history_lines(
        self,
        chat_id: str,
        *,
        exclude_message_ids: set[str],
    ) -> list[str]:
        history = self._group_recent_messages.get(chat_id)
        if not history:
            return []

        cutoff_ms = self._cutoff_ms()
        lines: list[str] = []
        for item in history:
            if item.event_time_ms < cutoff_ms:
                continue
            if item.message_id in exclude_message_ids:
                continue
            lines.append(f"{item.sender_name}: {item.text}")

        return lines[-self._max_items :]

    def _trim_history(self, history: deque[_GroupRecentMessage]) -> None:
        cutoff_ms = self._cutoff_ms()
        while history and (
            len(history) > self._max_items or history[0].event_time_ms < cutoff_ms
        ):
            history.popleft()

    def _evict_stale_groups(self) -> None:
        if len(self._group_recent_messages) <= self._max_groups:
            return
        groups = sorted(
            self._group_recent_messages.items(),
            key=lambda kv: kv[1][-1].event_time_ms if kv[1] else 0,
        )
        while len(self._group_recent_messages) > self._max_groups and groups:
            chat_id, _ = groups.pop(0)
            del self._group_recent_messages[chat_id]

    def _cutoff_ms(self) -> int:
        return int(time.time() * 1000) - self._window_ms


__all__ = ["FeishuGroupHistoryStore"]
