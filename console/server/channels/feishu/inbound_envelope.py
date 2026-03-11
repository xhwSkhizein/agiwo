"""Typed Feishu inbound envelope used across the channel pipeline."""

from dataclasses import dataclass, field


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


__all__ = ["FeishuInboundEnvelope", "FeishuMention"]
