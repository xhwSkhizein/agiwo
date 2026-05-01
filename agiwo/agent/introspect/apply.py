"""Apply introspection outcomes to runtime state and committed facts."""

from typing import Any

from agiwo.agent.introspect.models import IntrospectionCheckpoint, IntrospectionOutcome
from agiwo.agent.introspect.repair import build_context_repair_plan
from agiwo.agent.models.step import StepView
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_ops import (
    apply_tool_message_content_updates,
    remove_tool_call_from_messages,
)
from agiwo.agent.runtime.state_writer import RunStateWriter

_HIDDEN_CONTEXT_REASON = "introspection_metadata"


async def build_tool_step_lookup(
    context: RunContext,
    batch_lookup: dict[str, dict[str, Any]],
    *,
    start_seq: int | None = None,
    end_seq: int | None = None,
) -> dict[str, dict[str, Any]]:
    lookup = {
        tool_call_id: info
        for tool_call_id, info in batch_lookup.items()
        if _sequence_in_window(
            info.get("sequence"), start_seq=start_seq, end_seq=end_seq
        )
    }
    steps = await context.session_runtime.run_log_storage.list_step_views(
        session_id=context.session_id,
        start_seq=start_seq,
        end_seq=end_seq,
        agent_id=context.agent_id,
        include_hidden_from_context=True,
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


def _sequence_in_window(
    sequence: object,
    *,
    start_seq: int | None,
    end_seq: int | None,
) -> bool:
    if not isinstance(sequence, int):
        return False
    if start_seq is not None and sequence < start_seq:
        return False
    if end_seq is not None and sequence > end_seq:
        return False
    return True


async def apply_introspection_outcome(
    context: RunContext,
    outcome: IntrospectionOutcome,
    *,
    writer: RunStateWriter,
    step_lookup: dict[str, dict[str, Any]],
) -> None:
    projectable_entries: list[object] = []
    previous_boundary_seq = context.ledger.introspection.last_boundary_seq
    repair_plan = build_context_repair_plan(
        context.ledger.messages,
        outcome,
        previous_boundary_seq=previous_boundary_seq,
        step_lookup=step_lookup,
    )
    outcome.repair_plan = repair_plan

    if outcome.hidden_step_ids:
        projectable_entries.extend(
            await writer.record_context_steps_hidden(
                step_ids=outcome.hidden_step_ids,
                reason=_HIDDEN_CONTEXT_REASON,
            )
        )

    if repair_plan is not None:
        apply_tool_message_content_updates(
            context,
            [
                (update.tool_call_id, update.content)
                for update in repair_plan.content_updates
            ],
        )
        for update in repair_plan.content_updates:
            projectable_entries.extend(
                await writer.record_step_condensed_content_updated(
                    step_id=update.step_id,
                    condensed_content=update.content,
                )
            )

    remove_tool_call_from_messages(
        context,
        tool_call_id=outcome.review_tool_call_id,
    )

    if outcome.aligned is True:
        context.ledger.introspection.latest_aligned_checkpoint = (
            IntrospectionCheckpoint(
                seq=outcome.boundary_seq,
                milestone_id=outcome.active_milestone_id or "",
            )
        )
        projectable_entries.extend(
            await writer.record_introspection_checkpoint_recorded(
                checkpoint_seq=outcome.boundary_seq,
                milestone_id=outcome.active_milestone_id,
                review_tool_call_id=outcome.review_tool_call_id,
                review_step_id=outcome.review_step_id,
            )
        )

    projectable_entries.extend(
        await writer.record_introspection_outcome_recorded(
            aligned=outcome.aligned,
            mode=outcome.mode,
            experience=outcome.experience,
            active_milestone_id=outcome.active_milestone_id,
            review_tool_call_id=outcome.review_tool_call_id,
            review_step_id=outcome.review_step_id,
            hidden_step_ids=outcome.hidden_step_ids,
            notice_cleaned_step_ids=(
                repair_plan.notice_cleaned_step_ids if repair_plan is not None else []
            ),
            condensed_step_ids=repair_plan.condensed_step_ids
            if repair_plan is not None and outcome.mode == "step_back"
            else [],
            boundary_seq=outcome.boundary_seq,
            repair_start_seq=repair_plan.start_seq if repair_plan is not None else None,
            repair_end_seq=repair_plan.end_seq if repair_plan is not None else None,
        )
    )

    if outcome.mode == "step_back" and repair_plan is not None:
        projectable_entries.extend(
            await writer.record_context_repair_applied(
                mode="step_back",
                affected_count=repair_plan.affected_count,
                start_seq=repair_plan.start_seq,
                end_seq=repair_plan.end_seq,
                experience=repair_plan.experience,
            )
        )
        await writer.project_entries(projectable_entries)
        await context.hooks.after_step_back(outcome, context)
    else:
        await writer.project_entries(projectable_entries)

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


__all__ = [
    "apply_introspection_outcome",
    "build_tool_step_lookup",
    "register_committed_tool_step",
]
