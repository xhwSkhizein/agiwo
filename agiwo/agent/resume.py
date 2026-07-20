"""Rebuild next-step plan from a paused RunLog (message-tail rules)."""

from dataclasses import dataclass
from typing import Any

from agiwo.agent.models.log import (
    ContextAssembled,
    MessagesRebuilt,
    RunCheckpoint,
    RunLogEntry,
    RunPaused,
)
from agiwo.agent.models.input import UserMessage


@dataclass(frozen=True, slots=True)
class ResumePlan:
    checkpoint_id: str
    last_committed_sequence: int
    messages: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, Any]] | None
    continue_user_message: UserMessage | None


def build_resume_plan(entries: list[RunLogEntry]) -> ResumePlan:
    """Deterministic resume from committed facts (ADR 0033 message-tail rules)."""
    paused = next(
        (e for e in reversed(entries) if isinstance(e, RunPaused)),
        None,
    )
    if paused is None:
        raise ValueError("run is not paused (missing RunPaused)")
    checkpoint = next(
        (
            e
            for e in reversed(entries)
            if isinstance(e, RunCheckpoint) and e.checkpoint_id == paused.checkpoint_id
        ),
        None,
    )
    if checkpoint is None:
        raise ValueError("paused run missing RunCheckpoint")

    replay = [e for e in entries if e.sequence <= checkpoint.last_committed_sequence]
    messages = _latest_messages(replay)
    if not messages:
        raise ValueError("paused run has no rebuildable messages")

    last = messages[-1]
    role = last.get("role")
    tool_calls = last.get("tool_calls")
    pending: list[dict[str, Any]] | None = None
    continue_user: UserMessage | None = None
    if role == "assistant" and tool_calls:
        pending = list(tool_calls)
    else:
        # assistant without tools, or tool result → continue with system user turn
        continue_user = UserMessage.from_system(
            "Continue from the recoverable pause checkpoint."
        )
    return ResumePlan(
        checkpoint_id=checkpoint.checkpoint_id,
        last_committed_sequence=checkpoint.last_committed_sequence,
        messages=messages,
        pending_tool_calls=pending,
        continue_user_message=continue_user,
    )


def _latest_messages(entries: list[RunLogEntry]) -> list[dict[str, Any]]:
    for entry in reversed(entries):
        if isinstance(entry, MessagesRebuilt):
            return list(entry.messages)
        if isinstance(entry, ContextAssembled):
            return list(entry.messages)
    return []


__all__ = ["ResumePlan", "build_resume_plan"]
