"""Helpers for executing tool batches and applying review outcomes."""

from collections.abc import Awaitable, Callable
from typing import Any

from agiwo.agent.models.run import TerminationReason
from agiwo.agent.models.step import StepView
from agiwo.agent.review import ReviewBatch, inject_system_review
from agiwo.agent.runtime.context import RunContext, RunRuntime
from agiwo.agent.runtime.step_commit import StepCommitter
from agiwo.agent.runtime.state_writer import RunStateWriter
from agiwo.agent.tool_executor import execute_tool_batch


ToolTerminationWriter = Callable[[TerminationReason, str], Awaitable[None]]


async def execute_tool_batch_cycle(
    *,
    context: RunContext,
    runtime: RunRuntime,
    tool_calls: list[dict[str, Any]],
    assistant_step_id: str | None,
    set_termination_reason: ToolTerminationWriter,
    commit_step: StepCommitter,
) -> bool:
    """Execute one tool batch and apply review/step-back outcomes."""
    writer = RunStateWriter(context)
    tool_results = await execute_tool_batch(
        tool_calls,
        tools_map=runtime.tools_map,
        context=context,
        abort_signal=runtime.abort_signal,
    )
    terminated = False
    batch = ReviewBatch(context.config, context.ledger, runtime.tools_map)

    for result in tool_results:
        call_id = result.tool_call_id or ""
        await context.hooks.after_tool_call(
            call_id,
            result.tool_name,
            result.input_args or {},
            result,
            context,
        )

        seq = await context.session_runtime.allocate_sequence()
        tool_step = StepView.tool(
            context,
            sequence=seq,
            tool_call_id=call_id,
            name=result.tool_name,
            content=result.content or "",
            content_for_user=result.content_for_user,
            is_error=not result.is_success,
        )
        tool_step.content = batch.process_result(
            result,
            current_seq=seq,
            assistant_step_id=assistant_step_id,
            tool_step_id=tool_step.id,
        )
        milestones_update = batch.consume_milestones_update_request()
        review_notice = batch.consume_review_notice_request()
        if review_notice is not None:
            review_advice = await context.hooks.before_review(
                trigger_reason=review_notice.trigger.value,
                milestone=review_notice.milestone,
                step_count=review_notice.step_count,
                context=context,
            )
            tool_step.content = inject_system_review(
                review_notice.content,
                review_notice.milestone,
                review_notice.step_count,
                trigger_reason=review_notice.trigger.value,
                review_advice=review_advice,
            )
        committed_step = await commit_step(tool_step)
        batch.register_step(call_id, committed_step.id, committed_step.sequence)
        if milestones_update is not None:
            await writer.record_review_milestones_updated(
                milestones=milestones_update.milestones,
                active_milestone_id=milestones_update.active_milestone_id,
                source_tool_call_id=milestones_update.source_tool_call_id,
                source_step_id=committed_step.id,
                reason=milestones_update.reason,
            )
        if review_notice is not None:
            await writer.record_review_trigger_decided(
                trigger_reason=review_notice.trigger.value,
                active_milestone_id=review_notice.milestone.id
                if review_notice.milestone is not None
                else None,
                review_count_since_checkpoint=review_notice.step_count,
                trigger_tool_call_id=result.tool_call_id or None,
                trigger_tool_step_id=committed_step.id,
                notice_step_id=committed_step.id,
            )

        if not terminated and result.termination_reason is not None:
            await set_termination_reason(result.termination_reason, result.tool_name)
            terminated = True

    await _apply_review_outcome(context, batch, writer=writer)
    return terminated or context.is_terminal


async def _apply_review_outcome(
    context: RunContext,
    batch: ReviewBatch,
    *,
    writer: RunStateWriter,
) -> None:
    outcome = await batch.finalize(
        storage=context.session_runtime.run_log_storage,
        session_id=context.session_id,
        run_id=context.run_id,
        agent_id=context.agent_id,
    )
    if outcome.mode == "none":
        return

    if outcome.hidden_step_ids:
        await writer.record_context_steps_hidden(step_ids=outcome.hidden_step_ids)

    for update in outcome.content_updates:
        _replace_tool_message_content(
            context.ledger.messages,
            tool_call_id=update.tool_call_id,
            content=update.content,
        )

    _remove_review_tool_call(
        context.ledger.messages,
        review_tool_call_id=outcome.review_tool_call_id,
    )

    if outcome.aligned is True:
        await writer.record_review_checkpoint_recorded(
            checkpoint_seq=outcome.checkpoint_seq,
            milestone_id=outcome.active_milestone_id,
            review_tool_call_id=outcome.review_tool_call_id,
            review_step_id=outcome.review_step_id,
        )

    await writer.record_review_outcome_recorded(
        aligned=outcome.aligned,
        mode=outcome.mode,
        experience=outcome.experience,
        active_milestone_id=outcome.active_milestone_id,
        review_tool_call_id=outcome.review_tool_call_id,
        review_step_id=outcome.review_step_id,
        hidden_step_ids=outcome.hidden_step_ids,
        notice_cleaned_step_ids=[
            update.step_id
            for update in outcome.content_updates
            if update.step_id and update.step_id not in outcome.condensed_step_ids
        ],
        condensed_step_ids=outcome.condensed_step_ids,
    )

    if outcome.mode == "step_back":
        step_back_entries = await writer.record_step_back_applied(
            affected_count=outcome.affected_count,
            checkpoint_seq=outcome.checkpoint_seq,
            experience=outcome.experience or "",
        )
        await context.session_runtime.project_run_log_entries(
            step_back_entries,
            run_id=context.run_id,
            agent_id=context.agent_id,
            parent_run_id=context.parent_run_id,
            depth=context.depth,
        )
        await context.hooks.after_step_back(outcome, context)


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
        if remaining_tool_calls or (isinstance(content, str) and content.strip()):
            kept_messages.append(message)

    messages[:] = kept_messages


__all__ = ["execute_tool_batch_cycle"]
