"""Input-domain models for agent user messages."""

import json
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

    @classmethod
    def from_value(cls, value: "UserInput") -> "UserMessage":
        if isinstance(value, str):
            return cls(content=[ContentPart(type=ContentType.TEXT, text=value)])
        if isinstance(value, list):
            return cls(content=value)
        return value

    def extract_text(self) -> str:
        texts = [
            part.text
            for part in self.content
            if part.type == ContentType.TEXT and part.text
        ]
        return "\n".join(texts)

    def has_content(self) -> bool:
        for part in self.content:
            if part.type == ContentType.TEXT and part.text and part.text.strip():
                return True
            if part.url:
                return True
        return False

    def to_message_content(self) -> "MessageContent":
        if len(self.content) == 1 and self.content[0].type == ContentType.TEXT:
            return self.content[0].text or ""

        result: list[dict[str, Any]] = []
        for part in self.content:
            if part.type == ContentType.TEXT:
                result.append({"type": "text", "text": part.text or ""})
                continue

            url = part.url or ""
            if not url:
                continue
            if _is_local_path(url):
                result.append({"type": "text", "text": _render_local_resource(part)})
                continue
            result.append(_build_media_block(part, url))
        return result

    @staticmethod
    def _serialize_content_parts(parts: list[Any]) -> list[Any]:
        payload: list[Any] = []
        for item in parts:
            payload.append(item.to_dict() if isinstance(item, ContentPart) else item)
        return payload

    @classmethod
    def _serialize_message_payload(cls, message: "UserMessage") -> dict[str, Any]:
        return {
            "content": cls._serialize_content_parts(message.content),
            "context": message.context.to_dict() if message.context else None,
        }

    @classmethod
    def serialize(cls, value: "UserInput") -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return json.dumps(
                {
                    "__type": "content_parts",
                    "parts": [part.to_dict() for part in value],
                }
            )
        return json.dumps(value.to_dict())

    @classmethod
    def deserialize(cls, value: str) -> "UserInput":
        if not value or not value.startswith("{"):
            return value
        try:
            data = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
        input_type = data.get("__type")
        if input_type == "content_parts":
            return [ContentPart.from_dict(part) for part in data.get("parts") or []]
        if input_type == "user_message":
            return cls.from_dict(data)
        return value

    @classmethod
    def to_storage_value(cls, value: "UserInput | None") -> str | None:
        if value is None:
            return None
        return cls.serialize(value)

    @classmethod
    def from_storage_value(cls, value: "UserInput | None") -> "UserInput | None":
        if isinstance(value, str):
            return cls.deserialize(value)
        return value

    @classmethod
    def to_transport_payload(
        cls,
        value: "UserInput | dict[str, Any] | None",
    ) -> str | dict[str, Any] | list[Any] | None:
        if value is None:
            return None

        normalized = cls.from_storage_value(value)
        if isinstance(normalized, dict):
            input_type = normalized.get("__type")
            if input_type in {"content_parts", "user_message"}:
                normalized = cls.deserialize(json.dumps(normalized))

        if isinstance(normalized, UserMessage):
            return cls._serialize_message_payload(normalized)
        if isinstance(normalized, list):
            return cls._serialize_content_parts(normalized)
        return normalized

    @classmethod
    def to_structured_payload(
        cls,
        value: "UserInput | None",
    ) -> str | dict[str, Any] | list[Any] | None:
        return cls.to_transport_payload(value)


UserInput = str | list[ContentPart] | UserMessage
MessageContent = str | list[dict[str, Any]]


def _is_local_path(url: str) -> bool:
    return not (
        url.startswith("http://")
        or url.startswith("https://")
        or url.startswith("data:")
    )


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    return f"{size_bytes / (1024 * 1024):.1f}MB"


_CONTENT_TYPE_LABELS: dict[ContentType, str] = {
    ContentType.IMAGE: "图片",
    ContentType.AUDIO: "语音",
    ContentType.VIDEO: "视频",
    ContentType.FILE: "文件",
}
_DEFAULT_CONTENT_LABEL = "附件"
_LOCAL_PATH_PREFIX = "本地路径"
_FILE_PROCESSING_HINT = "如需处理此文件内容，请使用文件读取工具。"


def _render_local_resource(part: ContentPart) -> str:
    name = (part.metadata or {}).get("name", "")
    size = (part.metadata or {}).get("size", 0)
    mime_type = part.mime_type or ""
    meta_parts = [
        value for value in [mime_type, _format_file_size(size) if size else ""] if value
    ]
    meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
    label = _CONTENT_TYPE_LABELS.get(part.type, _DEFAULT_CONTENT_LABEL)
    lines = [f"[{label}: {name}{meta_str}]"]
    if part.url:
        lines.append(f"{_LOCAL_PATH_PREFIX}: {part.url}")
    if part.type in (ContentType.AUDIO, ContentType.VIDEO, ContentType.FILE):
        lines.append(_FILE_PROCESSING_HINT)
    return "\n".join(lines)


def _build_media_block(part: ContentPart, url: str) -> dict[str, Any]:
    if part.type == ContentType.IMAGE:
        block: dict[str, Any] = {
            "type": "image_url",
            "image_url": {"url": url},
        }
        if part.detail:
            block["image_url"]["detail"] = part.detail
        return block
    if part.type == ContentType.AUDIO:
        return {
            "type": "input_audio",
            "input_audio": {"url": url, "format": part.mime_type or ""},
        }
    if part.type == ContentType.VIDEO:
        return {
            "type": "video_url",
            "video_url": {"url": url},
        }
    return {
        "type": "file",
        "file": {"url": url, "mime_type": part.mime_type or ""},
    }


__all__ = [
    "ChannelContext",
    "ContentPart",
    "ContentType",
    "MessageContent",
    "UserInput",
    "UserMessage",
]
