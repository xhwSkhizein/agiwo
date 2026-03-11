"""Feishu message content extraction and text normalization."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from server.channels.models import Attachment

_TEXT_MENTION_PATTERN = re.compile(r"<at[^>]*>.*?</at>", re.IGNORECASE)


@dataclass
class ParsedContent:
    text: str
    attachments: list[Attachment] = field(default_factory=list)


def _extract_text(content: dict[str, Any]) -> ParsedContent:
    return ParsedContent(text=content.get("text", ""))


def _resolve_post_payload(content: dict[str, Any]) -> dict[str, Any] | None:
    for locale in ("zh_cn", "en_us", "ja_jp"):
        post = content.get(locale)
        if post is not None:
            return post
    for value in content.values():
        if isinstance(value, dict) and "content" in value:
            return value
    if "content" in content and isinstance(content["content"], list):
        return content
    return None


def _render_post_link(element: dict[str, Any]) -> str:
    href = element.get("href", "")
    link_text = element.get("text", "")
    if link_text and href:
        return f"{link_text}({href})"
    return href or link_text


def _append_post_element(
    element: dict[str, Any],
    *,
    line_parts: list[str],
    attachments: list[Attachment],
) -> None:
    tag = element.get("tag")
    if tag == "text":
        line_parts.append(element.get("text", ""))
        return
    if tag == "a":
        line_parts.append(_render_post_link(element))
        return
    if tag == "at":
        user_name = element.get("user_name", "")
        if user_name:
            line_parts.append(f"@{user_name}")
        return
    if tag == "img":
        image_key = element.get("image_key", "")
        attachments.append(Attachment(type="image", key=image_key))
        line_parts.append(f"[图片 image_key={image_key}]")
        return
    if tag == "media":
        file_key = element.get("file_key", "")
        attachments.append(Attachment(type="media", key=file_key))
        line_parts.append(f"[视频 file_key={file_key}]")
        return
    if tag == "emotion":
        emoji_type = element.get("emoji_type", "")
        if emoji_type:
            line_parts.append(f"[{emoji_type}]")


def _extract_post_paragraph(
    paragraph: object,
    *,
    attachments: list[Attachment],
) -> str:
    if not isinstance(paragraph, list):
        return ""
    line_parts: list[str] = []
    for element in paragraph:
        if not isinstance(element, dict):
            continue
        _append_post_element(
            element,
            line_parts=line_parts,
            attachments=attachments,
        )
    return "".join(line_parts)


def _extract_post(content: dict[str, Any]) -> ParsedContent:
    """Extract plain text and attachments from Feishu post (rich text)."""
    post = _resolve_post_payload(content)
    if post is None:
        return ParsedContent(text="")

    parts: list[str] = []
    attachments: list[Attachment] = []

    title = post.get("title", "")
    if title:
        parts.append(title)

    for paragraph in post.get("content", []):
        line = _extract_post_paragraph(paragraph, attachments=attachments)
        if line:
            parts.append(line)

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


class FeishuContentExtractor:
    def extract(
        self,
        *,
        message_type: str,
        raw_content: str,
    ) -> ParsedContent:
        if not raw_content:
            return ParsedContent(text="")
        try:
            content_json = json.loads(raw_content)
        except json.JSONDecodeError:
            return ParsedContent(text=raw_content)
        if not isinstance(content_json, dict):
            return ParsedContent(text=raw_content)
        extractor = _CONTENT_EXTRACTORS.get(message_type)
        if extractor is None:
            return ParsedContent(text=f"[{message_type}]")
        return extractor(content_json)

    def normalize_message_text(self, text: str) -> str:
        clean = _TEXT_MENTION_PATTERN.sub("", text or "").strip()
        clean = re.sub(r"\s+", " ", clean)
        return clean


__all__ = ["FeishuContentExtractor", "ParsedContent"]
