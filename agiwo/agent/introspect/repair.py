"""Pure context repair planning for introspection outcomes."""

from typing import Any

from agiwo.agent.introspect.models import (
    ContentUpdate,
    ContextRepairPlan,
    IntrospectionOutcome,
)
from agiwo.agent.introspect.trajectory import (
    has_system_review_notice,
    strip_system_review_notices,
)


def _step_id_for_tool_call(
    tool_call_id: str,
    step_lookup: dict[str, dict[str, Any]],
) -> str:
    info = step_lookup.get(tool_call_id)
    step_id = info.get("id", "") if info is not None else ""
    return step_id if isinstance(step_id, str) else ""


def build_context_repair_plan(
    messages: list[dict[str, Any]],
    outcome: IntrospectionOutcome,
    *,
    previous_boundary_seq: int,
    step_lookup: dict[str, dict[str, Any]],
) -> ContextRepairPlan | None:
    updates: list[ContentUpdate] = []
    cleaned_step_ids: list[str] = []
    start_seq = previous_boundary_seq + 1
    end_seq = max(outcome.boundary_seq - 1, previous_boundary_seq)

    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        if outcome.review_tool_call_id and tool_call_id == outcome.review_tool_call_id:
            continue
        sequence = message.get("_sequence", 0)
        if not isinstance(sequence, int):
            sequence = 0
        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue
        step_id = _step_id_for_tool_call(tool_call_id, step_lookup)
        if not step_id:
            continue
        if (
            outcome.mode == "step_back"
            and previous_boundary_seq < sequence < outcome.boundary_seq
        ):
            updates.append(
                ContentUpdate(
                    step_id=step_id,
                    tool_call_id=tool_call_id,
                    content=f"[EXPERIENCE] {outcome.experience or ''}",
                )
            )
            continue
        if has_system_review_notice(content):
            updates.append(
                ContentUpdate(
                    step_id=step_id,
                    tool_call_id=tool_call_id,
                    content=strip_system_review_notices(content),
                )
            )
            cleaned_step_ids.append(step_id)

    if not updates:
        return None
    return ContextRepairPlan(
        mode=outcome.mode,
        start_seq=start_seq,
        end_seq=end_seq,
        experience=outcome.experience or "",
        content_updates=updates,
        notice_cleaned_step_ids=cleaned_step_ids,
    )


__all__ = ["build_context_repair_plan"]
