"""Helpers for normalizing and serializing agent user input."""

import json
from typing import Any

from agiwo.agent.input import (
    ContentPart,
    ContentType,
    MessageContent,
    UserInput,
    UserMessage,
)


def normalize_to_message(user_input: UserInput) -> UserMessage:
    """Normalize any UserInput form into a UserMessage."""
    if isinstance(user_input, str):
        return UserMessage(
            content=[ContentPart(type=ContentType.TEXT, text=user_input)]
        )
    if isinstance(user_input, list):
        return UserMessage(content=user_input)
    return user_input


def extract_text(user_input: UserInput) -> str:
    if isinstance(user_input, str):
        return user_input
    parts = user_input.content if isinstance(user_input, UserMessage) else user_input
    texts = [part.text for part in parts if part.type == ContentType.TEXT and part.text]
    return "\n".join(texts)


def serialize_user_input(user_input: UserInput) -> str:
    """Serialize UserInput to a string suitable for storage."""
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        return json.dumps(
            {
                "__type": "content_parts",
                "parts": [part.to_dict() for part in user_input],
            }
        )
    return json.dumps(user_input.to_dict())


def deserialize_user_input(value: str) -> UserInput:
    """Deserialize UserInput from a storage string."""
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
        return UserMessage.from_dict(data)
    return value


def to_message_content(parts: list[ContentPart]) -> MessageContent:
    if len(parts) == 1 and parts[0].type == ContentType.TEXT:
        return parts[0].text or ""

    result: list[dict[str, Any]] = []
    for part in parts:
        if part.type == ContentType.TEXT:
            result.append({"type": "text", "text": part.text or ""})
            continue

        url = part.url or ""
        if url and _is_local_path(url):
            result.append({"type": "text", "text": _render_local_resource(part)})
            continue
        result.append(_build_media_block(part, url))
    return result


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


def _render_local_resource(part: ContentPart) -> str:
    name = (part.metadata or {}).get("name", "")
    size = (part.metadata or {}).get("size", 0)
    mime_type = part.mime_type or ""
    meta_parts = [
        value for value in [mime_type, _format_file_size(size) if size else ""] if value
    ]
    meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
    type_labels = {
        ContentType.IMAGE: "图片",
        ContentType.AUDIO: "语音",
        ContentType.VIDEO: "视频",
        ContentType.FILE: "文件",
    }
    label = type_labels.get(part.type, "附件")
    lines = [f"[{label}: {name}{meta_str}]"]
    if part.url:
        lines.append(f"本地路径: {part.url}")
    if part.type in (ContentType.AUDIO, ContentType.VIDEO, ContentType.FILE):
        lines.append("如需处理此文件内容，请使用文件读取工具。")
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
    "deserialize_user_input",
    "extract_text",
    "normalize_to_message",
    "serialize_user_input",
    "to_message_content",
]
