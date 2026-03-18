"""Feishu rich-text (post) message content builders."""

from typing import Any


def build_post_content(
    title: str,
    content: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Build Feishu post message content.

    Args:
        title: Message title
        content: List of lines, each line is a list of elements.
                Each element has tag (text, at, a, img, etc.)

    Example:
        >>> build_post_content("会话列表", [
        ...     [{"tag": "text", "text": "1. "}, {"tag": "at", "user_id": "xxx"}],
        ...     [{"tag": "a", "text": "查看详情", "href": "https://..."}],
        ... ])
    """
    return {
        "zh_cn": {
            "title": title,
            "content": content,
        }
    }


def text_element(text: str, style: list[str] | None = None) -> dict[str, Any]:
    """Create a text element."""
    elem: dict[str, Any] = {"tag": "text", "text": text}
    if style:
        elem["style"] = style
    return elem


def bold(text: str) -> dict[str, Any]:
    """Create a bold text element."""
    return {"tag": "text", "text": text, "style": ["bold"]}


def code(text: str) -> dict[str, Any]:
    """Create a code-style text element."""
    return {"tag": "text", "text": f"`{text}`", "style": ["code"]}


def link(text: str, href: str) -> dict[str, Any]:
    """Create a link element."""
    return {"tag": "a", "text": text, "href": href}


def at_element(user_id: str, user_name: str = "") -> dict[str, Any]:
    """Create an @mention element."""
    elem: dict[str, Any] = {"tag": "at", "user_id": user_id}
    if user_name:
        elem["user_name"] = user_name
    return elem


def image_element(image_key: str) -> dict[str, Any]:
    """Create an image element."""
    return {"tag": "img", "image_key": image_key}


def new_line() -> list[dict[str, Any]]:
    """Create an empty line (line break)."""
    return []


def separator_line() -> list[dict[str, Any]]:
    """Create a separator line with dashes."""
    return [text_element("─" * 30)]


__all__ = [
    "build_post_content",
    "text_element",
    "bold",
    "code",
    "link",
    "at_element",
    "image_element",
    "new_line",
    "separator_line",
]
