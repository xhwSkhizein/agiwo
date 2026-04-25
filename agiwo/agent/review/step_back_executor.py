"""Step-back execution — build targeted tool-result content updates."""

from dataclasses import dataclass, field
from typing import Any, Literal

from agiwo.agent.storage.base import RunLogStorage
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ContentUpdate:
    step_id: str
    tool_call_id: str
    content: str


@dataclass
class StepBackOutcome:
    """Structured cleanup outcome used by review finalization."""

    mode: Literal["none", "metadata_only", "step_back"] = "none"
    review_tool_call_id: str | None = None
    hidden_step_ids: list[str] = field(default_factory=list)
    content_updates: list[ContentUpdate] = field(default_factory=list)
    affected_count: int = 0
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
    """Condense tool results after *checkpoint_seq* into targeted updates."""

    content_updates: list[ContentUpdate] = []

    for message in messages:
        if message.get("role") != "tool":
            continue

        sequence = message.get("_sequence", 0)
        if sequence <= checkpoint_seq:
            continue

        tool_call_id = message.get("tool_call_id", "")
        if not tool_call_id:
            continue
        if review_tool_call_id and tool_call_id == review_tool_call_id:
            continue

        original_content = message.get("content", "")
        if not original_content:
            continue

        step_info = step_lookup.get(tool_call_id)
        step_id = step_info.get("id", "") if step_info is not None else ""
        if not step_id:
            step = await storage.get_step_by_tool_call_id(session_id, tool_call_id)
            step_id = step.id if step is not None else ""
        if not step_id:
            continue

        condensed_content = f"[EXPERIENCE] {experience}"
        await storage.append_step_condensed_content(
            session_id,
            run_id,
            agent_id,
            step_id,
            condensed_content,
        )
        content_updates.append(
            ContentUpdate(
                step_id=step_id,
                tool_call_id=tool_call_id,
                content=condensed_content,
            )
        )

    logger.info(
        "step_back_executed",
        session_id=session_id,
        affected_count=len(content_updates),
        checkpoint_seq=checkpoint_seq,
    )

    return StepBackOutcome(
        mode="step_back",
        review_tool_call_id=review_tool_call_id,
        content_updates=content_updates,
        affected_count=len(content_updates),
        checkpoint_seq=checkpoint_seq,
        experience=experience,
    )


__all__ = [
    "ContentUpdate",
    "StepBackOutcome",
    "execute_step_back",
]
