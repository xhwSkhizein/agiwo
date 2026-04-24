"""StepBack Executor — KV-cache-safe content condensation."""

from dataclasses import dataclass, field
from typing import Any

from agiwo.agent.storage.base import RunLogStorage
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StepBackOutcome:
    """Result of a step-back execution.

    When ``applied`` is True, messages contains the updated message list.
    The caller applies targeted content updates in-place — no
    rebuild_messages, no message reordering.
    """

    applied: bool = False
    affected_count: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_seq: int = 0
    experience: str | None = None


async def execute_step_back(
    *,
    messages: list[dict[str, Any]],
    checkpoint_seq: int,
    experience: str,
    review_tool_call_id: str | None = None,
    step_lookup: dict[str, dict[str, Any]],
    storage: RunLogStorage,
    session_id: str,
    run_id: str,
    agent_id: str,
) -> StepBackOutcome:
    """Condense tool results after *checkpoint_seq* to *experience*.

    Core invariants (KV-cache-safe):
    - Tool call assistant messages are NEVER removed or reordered
    - Only tool result content is replaced in-place
    - review_trajectory's own call + result are removed (they're at the tail)
    - No rebuild_messages call — caller applies targeted content updates
    """
    working = list(messages)  # shallow copy
    affected_count = 0

    # 1. Remove review_trajectory tool call and result first (they're at the tail)
    _remove_review_tool_call(working, review_tool_call_id=review_tool_call_id)

    # 2. Find and condense tool results after checkpoint
    for i, msg in enumerate(working):
        if msg.get("role") != "tool":
            continue
        seq = msg.get("_sequence", 0)
        if seq <= checkpoint_seq:
            continue

        tool_call_id = msg.get("tool_call_id", "")
        if not tool_call_id:
            continue

        original_content = msg.get("content", "")
        if not original_content:
            continue

        condensed = f"[EXPERIENCE] {experience}"
        msg["content"] = condensed
        affected_count += 1

        # Persist original content to storage
        step_info = step_lookup.get(tool_call_id)
        step_id = step_info.get("id", "") if step_info is not None else ""
        if not step_id:
            step = await storage.get_step_by_tool_call_id(session_id, tool_call_id)
            step_id = step.id if step is not None else ""
        if step_id:
            await storage.append_step_condensed_content(
                session_id,
                run_id,
                agent_id,
                step_id,
                condensed,
            )

    logger.info(
        "step_back_executed",
        session_id=session_id,
        affected_count=affected_count,
        checkpoint_seq=checkpoint_seq,
    )

    return StepBackOutcome(
        applied=True,
        affected_count=affected_count,
        messages=working,
        checkpoint_seq=checkpoint_seq,
        experience=experience,
    )


def _remove_review_tool_call(
    messages: list[dict[str, Any]],
    *,
    review_tool_call_id: str | None,
) -> None:
    """Remove only the current review_trajectory tool call and its result."""
    indices_to_remove: list[int] = []
    review_call_ids = (
        {review_tool_call_id}
        if review_tool_call_id
        else _find_latest_review_tool_call_id(messages)
    )
    if not review_call_ids:
        return

    confirmed_review_call_ids: set[str] = set()
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            remaining: list[dict[str, Any]] = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                if (
                    tc.get("id") not in review_call_ids
                    or fn.get("name") != "review_trajectory"
                ):
                    remaining.append(tc)
                else:
                    confirmed_review_call_ids.add(tc.get("id", ""))
            msg["tool_calls"] = remaining
            if not remaining:
                indices_to_remove.append(i)

    if not confirmed_review_call_ids:
        return

    for i, msg in enumerate(messages):
        if (
            msg.get("role") == "tool"
            and msg.get("tool_call_id", "") in confirmed_review_call_ids
        ):
            indices_to_remove.append(i)

    for i in sorted(indices_to_remove, reverse=True):
        messages.pop(i)


def _find_latest_review_tool_call_id(messages: list[dict[str, Any]]) -> set[str]:
    for msg in reversed(messages):
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in reversed(msg["tool_calls"]):
            fn = tc.get("function", {})
            if fn.get("name") == "review_trajectory":
                tool_call_id = tc.get("id")
                return {tool_call_id} if tool_call_id else set()
    return set()


__all__ = [
    "StepBackOutcome",
    "execute_step_back",
]
