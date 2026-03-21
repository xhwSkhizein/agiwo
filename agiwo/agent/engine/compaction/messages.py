from typing import Any

from agiwo.agent.engine.compaction.prompt import DEFAULT_ASSISTANT_RESPONSE
from agiwo.config.settings import settings


def build_compacted_messages(
    system_prompt: str,
    summary: str,
    transcript_path: str,
    latest_user_message: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the compacted message list after compaction."""
    assistant_response = (
        settings.compact_assistant_response or DEFAULT_ASSISTANT_RESPONSE
    )

    compacted_messages: list[dict[str, Any]] = []

    if system_prompt:
        compacted_messages.append({"role": "system", "content": system_prompt})

    compact_user_content = (
        f"[Conversation compressed. original source: {transcript_path}]\n\n"
        f"# Summary\n{summary}"
    )
    compacted_messages.append({"role": "user", "content": compact_user_content})
    compacted_messages.append({"role": "assistant", "content": assistant_response})

    if latest_user_message and latest_user_message.get("role") == "user":
        compacted_messages.append(latest_user_message)

    return compacted_messages


__all__ = ["build_compacted_messages"]
