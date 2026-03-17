"""Recent Feishu group-message history used for prompt context."""

import time
from collections import deque
from dataclasses import dataclass

from server.channels.session.models import InboundMessage

_GROUP_HISTORY_MAX_ITEMS = 20
_GROUP_HISTORY_WINDOW_MS = 15 * 60 * 1000


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
    ) -> None:
        self._max_items = max_items
        self._window_ms = window_ms
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

        return lines[-self._max_items:]

    def _trim_history(self, history: deque[_GroupRecentMessage]) -> None:
        cutoff_ms = self._cutoff_ms()
        while history and (
            len(history) > self._max_items or history[0].event_time_ms < cutoff_ms
        ):
            history.popleft()

    def _cutoff_ms(self) -> int:
        return int(time.time() * 1000) - self._window_ms


__all__ = ["FeishuGroupHistoryStore"]
