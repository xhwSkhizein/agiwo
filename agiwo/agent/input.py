"""Input-domain models for agent user messages."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


@dataclass
class ContentPart:
    type: ContentType
    text: str | None = None
    url: str | None = None
    mime_type: str | None = None
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type.value}
        if self.text is not None:
            result["text"] = self.text
        if self.url is not None:
            result["url"] = self.url
        if self.mime_type is not None:
            result["mime_type"] = self.mime_type
        if self.detail is not None:
            result["detail"] = self.detail
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentPart":
        return cls(
            type=ContentType(data["type"]),
            text=data.get("text"),
            url=data.get("url"),
            mime_type=data.get("mime_type"),
            detail=data.get("detail"),
            metadata=data.get("metadata") or {},
        )


@dataclass
class ChannelContext:
    """Metadata about the channel/environment from which input originates."""

    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelContext":
        return cls(source=data["source"], metadata=data.get("metadata") or {})


@dataclass
class UserMessage:
    """Structured user input: multimodal content plus optional channel context."""

    content: list[ContentPart]
    context: ChannelContext | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "__type": "user_message",
            "content": [part.to_dict() for part in self.content],
            "context": self.context.to_dict() if self.context else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserMessage":
        return cls(
            content=[ContentPart.from_dict(part) for part in data.get("content") or []],
            context=(
                ChannelContext.from_dict(data["context"])
                if data.get("context")
                else None
            ),
        )


UserInput = str | list[ContentPart] | UserMessage
MessageContent = str | list[dict[str, Any]]


__all__ = [
    "ChannelContext",
    "ContentPart",
    "ContentType",
    "MessageContent",
    "UserInput",
    "UserMessage",
]
