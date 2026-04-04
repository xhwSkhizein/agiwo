"""Retrospect execution — offload, replace, and persist condensed content."""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agiwo.agent.models.run import RunLedger
from agiwo.agent.storage.base import RunStepStorage
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrospectOutcome:
    """Result of a retrospect execution.

    When ``applied`` is True, ``messages`` contains the updated message list.
    The caller must explicitly apply it via ``replace_messages(state, outcome.messages)``.
    """

    applied: bool = False
    offloaded_count: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)


def offload_to_disk(content: str, path: Path) -> str:
    """Write *content* to *path* and return a placeholder string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"[ToolResult offloaded to {path}]"


async def execute_retrospect(
    *,
    feedback: str,
    messages: list[dict[str, Any]],
    ledger: RunLedger,
    storage: RunStepStorage,
    session_id: str,
    offload_dir: Path,
    step_lookup: dict[str, dict[str, Any]],
) -> RetrospectOutcome:
    """Execute a retrospect pass and return the outcome.

    Works on a *copy* of the message list so the caller can decide when to
    apply the change via ``replace_messages``.  Disk offload and storage
    ``condensed_content`` updates are performed as internal side effects.
    """
    working = copy.deepcopy(messages)

    last_retrospect_seq = ledger.last_retrospect_seq
    offloaded_count = 0
    last_tool_call_id: str | None = None

    for msg in working:
        if msg.get("role") != "tool":
            continue
        seq = msg.get("_sequence", 0)
        if seq <= last_retrospect_seq:
            continue

        tool_call_id = msg.get("tool_call_id", "")
        original_content = msg.get("content", "")
        if not original_content:
            continue

        offload_path = offload_dir / f"{tool_call_id}.txt"
        placeholder = offload_to_disk(original_content, offload_path)
        msg["content"] = placeholder
        offloaded_count += 1
        last_tool_call_id = tool_call_id

        step_info = step_lookup.get(tool_call_id)
        if step_info is not None:
            step_id = step_info.get("id", "")
            if step_id:
                await storage.update_step_condensed_content(
                    session_id, step_id, placeholder
                )

    if last_tool_call_id is not None and feedback:
        await _persist_feedback(
            working,
            last_retrospect_seq,
            feedback,
            last_tool_call_id,
            step_lookup,
            storage,
            session_id,
        )

    _remove_retrospect_tool_call(working)

    max_seq = 0
    for msg in working:
        s = msg.get("_sequence", 0)
        if s > max_seq:
            max_seq = s
    ledger.last_retrospect_seq = max_seq
    ledger.retrospect_pending_tokens = 0
    ledger.retrospect_pending_rounds = 0

    logger.info(
        "retrospect_executed",
        session_id=session_id,
        offloaded_count=offloaded_count,
    )

    return RetrospectOutcome(
        applied=True,
        offloaded_count=offloaded_count,
        messages=working,
    )


async def _persist_feedback(
    messages: list[dict[str, Any]],
    last_retrospect_seq: int,
    feedback: str,
    target_tool_call_id: str,
    step_lookup: dict[str, Any],
    storage: RunStepStorage,
    session_id: str,
) -> None:
    """Append retrospect feedback to the last tool message and persist."""
    for msg in reversed(messages):
        if msg.get("role") != "tool":
            continue
        seq = msg.get("_sequence", 0)
        if seq <= last_retrospect_seq:
            break
        combined = (msg.get("content", "") or "") + f"\n---\nRetrospect: {feedback}"
        msg["content"] = combined
        step_info = step_lookup.get(target_tool_call_id)
        if step_info is not None:
            step_id = step_info.get("id", "")
            if step_id:
                await storage.update_step_condensed_content(
                    session_id, step_id, combined
                )
        break


def _remove_retrospect_tool_call(messages: list[dict[str, Any]]) -> None:
    """Remove the retrospect_tool_result tool call and its result."""
    indices_to_remove: list[int] = []

    retrospect_call_ids: set[str] = set()
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                if fn.get("name") == "retrospect_tool_result":
                    retrospect_call_ids.add(tc.get("id", ""))
                    if len(msg["tool_calls"]) == 1:
                        indices_to_remove.append(i)

    for i, msg in enumerate(messages):
        if (
            msg.get("role") == "tool"
            and msg.get("tool_call_id", "") in retrospect_call_ids
        ):
            indices_to_remove.append(i)

    for i in sorted(indices_to_remove, reverse=True):
        messages.pop(i)


__all__ = [
    "RetrospectOutcome",
    "execute_retrospect",
    "offload_to_disk",
]
