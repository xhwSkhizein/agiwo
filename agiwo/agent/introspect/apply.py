"""Apply introspection outcomes to runtime state and committed facts."""

from typing import Any

from agiwo.agent.introspect.models import (
    ContentUpdate,
    IntrospectionCheckpoint,
    IntrospectionOutcome,
)
from agiwo.agent.introspect.repair import build_context_repair_plan
from agiwo.agent.models.step import StepView
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_writer import RunStateWriter

_HIDDEN_CONTEXT_REASON = "introspection_metadata"


async def build_tool_step_lookup(
    context: RunContext,
    batch_lookup: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    lookup = dict(batch_lookup)
    steps = await context.session_runtime.run_log_storage.list_step_views(
        session_id=context.session_id,
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


async def apply_introspection_outcome(
    context: RunContext,
    outcome: IntrospectionOutcome,
    *,
    writer: RunStateWriter,
    step_lookup: dict[str, dict[str, Any]],
) -> None:
    previous_boundary_seq = context.ledger.introspection.last_boundary_seq
    repair_plan = build_context_repair_plan(
        context.ledger.messages,
        outcome,
        previous_boundary_seq=previous_boundary_seq,
        step_lookup=step_lookup,
    )
    outcome.repair_plan = repair_plan

    if outcome.hidden_step_ids:
        await writer.record_context_steps_hidden(
            step_ids=outcome.hidden_step_ids,
            reason=_HIDDEN_CONTEXT_REASON,
        )

    if repair_plan is not None:
        _apply_content_updates(context.ledger.messages, repair_plan.content_updates)
        for update in repair_plan.content_updates:
            await writer.record_step_condensed_content_updated(
                step_id=update.step_id,
                condensed_content=update.content,
            )

    _remove_review_tool_call(
        context.ledger.messages,
        review_tool_call_id=outcome.review_tool_call_id,
    )

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

    if outcome.mode == "step_back" and repair_plan is not None:
        repair_entries = await writer.record_context_repair_applied(
            mode="step_back",
            affected_count=repair_plan.affected_count,
            start_seq=repair_plan.start_seq,
            end_seq=repair_plan.end_seq,
            experience=repair_plan.experience,
        )
        await context.session_runtime.project_run_log_entries(
            repair_entries,
            run_id=context.run_id,
            agent_id=context.agent_id,
            parent_run_id=context.parent_run_id,
            depth=context.depth,
        )
        await context.hooks.after_step_back(outcome, context)

    context.ledger.introspection.pending_trigger = None
    context.ledger.introspection.notice_requested = False
    context.ledger.introspection.pending_milestone_switch = False
    context.ledger.introspection.review_count_since_boundary = 0
    context.ledger.introspection.last_boundary_seq = outcome.boundary_seq


def _apply_content_updates(
    messages: list[dict[str, Any]],
    updates: list[ContentUpdate],
) -> None:
    for update in updates:
        _replace_tool_message_content(
            messages,
            tool_call_id=update.tool_call_id,
            content=update.content,
        )


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


def _replace_tool_message_content(
    messages: list[dict[str, Any]],
    *,
    tool_call_id: str,
    content: str,
) -> None:
    for message in messages:
        if message.get("role") != "tool":
            continue
        if message.get("tool_call_id") != tool_call_id:
            continue
        message["content"] = content
        break


def _remove_review_tool_call(
    messages: list[dict[str, Any]],
    *,
    review_tool_call_id: str | None,
) -> None:
    if not review_tool_call_id:
        return

    kept_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "tool":
            if message.get("tool_call_id") != review_tool_call_id:
                kept_messages.append(message)
            continue

        if message.get("role") != "assistant":
            kept_messages.append(message)
            continue

        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            kept_messages.append(message)
            continue

        remaining_tool_calls = [
            tool_call
            for tool_call in tool_calls
            if tool_call.get("id") != review_tool_call_id
        ]
        if remaining_tool_calls:
            message["tool_calls"] = remaining_tool_calls
        else:
            message.pop("tool_calls", None)
        content = message.get("content")
        if remaining_tool_calls or _has_preservable_assistant_content(content):
            kept_messages.append(message)

    messages[:] = kept_messages


def _has_preservable_assistant_content(content: object) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, (list, dict)):
        return len(content) > 0
    return False


__all__ = [
    "apply_introspection_outcome",
    "build_tool_step_lookup",
    "register_committed_tool_step",
]
