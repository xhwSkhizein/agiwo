"""Apply introspection outcomes to runtime state and committed facts."""

from typing import Any

from agiwo.agent.introspect.models import (
    IntrospectionCheckpoint,
    IntrospectionOutcome,
    ToolUsefulnessEntry,
)
from agiwo.agent.models.step import StepView
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_writer import RunStateWriter


async def build_tool_step_lookup(
    context: RunContext,
    batch_lookup: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup = dict(batch_lookup)
    steps = await context.session_runtime.run_log_storage.list_step_views(
        session_id=context.session_id,
        agent_id=context.agent_id,
        limit=100_000,
    )
    for step in steps:
        if step.tool_call_id is None:
            continue
        lookup.setdefault(
            step.tool_call_id,
            {
                "id": step.id,
                "sequence": step.sequence,
            },
        )
    return lookup


async def apply_introspection_outcome(
    context: RunContext,
    outcome: IntrospectionOutcome,
    *,
    writer: RunStateWriter,
    step_lookup: dict[str, dict[str, Any]],
) -> None:
    del step_lookup

    if outcome.aligned is True:
        context.ledger.introspection.latest_aligned_checkpoint = (
            IntrospectionCheckpoint(
                seq=outcome.boundary_seq,
                milestone_id=outcome.active_milestone_id or "",
            )
        )
        await writer.record_introspection_checkpoint_recorded(
            checkpoint_seq=outcome.boundary_seq,
            milestone_id=outcome.active_milestone_id,
            review_tool_call_id=outcome.review_tool_call_id,
            review_step_id=outcome.review_step_id,
        )

    context.ledger.introspection.latest_tool_usefulness = list(outcome.tool_usefulness)

    await writer.record_introspection_outcome_recorded(
        aligned=outcome.aligned,
        experience=outcome.experience,
        tool_usefulness=[
            {
                "tool_call_id": entry.tool_call_id,
                "tool_name": entry.tool_name,
                "score": entry.score,
            }
            for entry in outcome.tool_usefulness
        ],
        active_milestone_id=outcome.active_milestone_id,
        review_tool_call_id=outcome.review_tool_call_id,
        review_step_id=outcome.review_step_id,
        boundary_seq=outcome.boundary_seq,
    )

    context.ledger.introspection.pending_trigger = None
    context.ledger.introspection.notice_requested = False
    context.ledger.introspection.pending_milestone_switch = False
    context.ledger.introspection.review_count_since_boundary = 0
    context.ledger.introspection.last_boundary_seq = outcome.boundary_seq


def register_committed_tool_step(
    lookup: dict[str, dict[str, Any]],
    *,
    tool_call_id: str,
    step: StepView,
) -> None:
    lookup[tool_call_id] = {
        "id": step.id,
        "sequence": step.sequence,
    }


def parse_tool_usefulness_output(
    raw_entries: object,
) -> list[ToolUsefulnessEntry]:
    if not isinstance(raw_entries, list):
        return []
    entries: list[ToolUsefulnessEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        tool_call_id = item.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        tool_name = item.get("tool_name")
        score = item.get("score")
        if score is not None and not isinstance(score, int):
            continue
        entries.append(
            ToolUsefulnessEntry(
                tool_call_id=tool_call_id,
                tool_name=tool_name if isinstance(tool_name, str) else None,
                score=score,
            )
        )
    return entries


__all__ = [
    "apply_introspection_outcome",
    "build_tool_step_lookup",
    "parse_tool_usefulness_output",
    "register_committed_tool_step",
]
