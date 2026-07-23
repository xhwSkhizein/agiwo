"""Review window helpers for trajectory introspection."""

from typing import Any


def build_review_window(
    messages: list[dict[str, Any]],
    *,
    last_boundary_seq: int,
    exclude_tool_call_id: str | None = None,
) -> list[dict[str, str]]:
    """Return prompt-visible tool calls in the current review window.

    The window contains committed tool results after ``last_boundary_seq``,
    excluding ``review_trajectory`` itself. Each entry exposes only
    ``tool_call_id`` and ``tool_name`` for the model protocol.
    """

    window: list[dict[str, str]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        sequence = message.get("_sequence", 0)
        if not isinstance(sequence, int) or sequence <= last_boundary_seq:
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        if exclude_tool_call_id is not None and tool_call_id == exclude_tool_call_id:
            continue
        tool_name = message.get("name")
        if tool_name == "review_trajectory":
            continue
        window.append(
            {
                "tool_call_id": tool_call_id,
                "tool_name": tool_name if isinstance(tool_name, str) else "tool",
            }
        )
    return window


__all__ = ["build_review_window"]
