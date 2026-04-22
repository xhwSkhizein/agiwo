"""Helpers for executing tool batches and applying retrospect outcomes."""

from collections.abc import Awaitable, Callable
from typing import Any

from agiwo.agent.models.stream import MessagesRebuiltEvent, RetrospectAppliedEvent
from agiwo.agent.models.run import TerminationReason
from agiwo.agent.models.step import StepView
from agiwo.agent.retrospect import RetrospectBatch
from agiwo.agent.runtime.context import RunContext, RunRuntime
from agiwo.agent.runtime.state_ops import replace_messages
from agiwo.agent.runtime.state_writer import (
    build_messages_rebuilt_entry,
    build_retrospect_applied_entry,
)
from agiwo.agent.runtime.step_committer import commit_step
from agiwo.agent.tool_executor import execute_tool_batch


ToolTerminationWriter = Callable[[TerminationReason, str], Awaitable[None]]


async def execute_tool_batch_cycle(
    *,
    context: RunContext,
    runtime: RunRuntime,
    tool_calls: list[dict[str, Any]],
    set_termination_reason: ToolTerminationWriter,
) -> bool:
    """Execute one tool batch and apply retrospect-driven message rebuilds."""
    tool_results = await execute_tool_batch(
        tool_calls,
        tools_map=runtime.tools_map,
        context=context,
        abort_signal=runtime.abort_signal,
    )
    terminated = False
    batch = RetrospectBatch(context, runtime.tools_map)

    for result in tool_results:
        call_id = result.tool_call_id or ""
        await context.hooks.after_tool_call(
            call_id,
            result.tool_name,
            result.input_args or {},
            result,
            context,
        )

        tool_step = StepView.tool(
            context,
            sequence=await context.session_runtime.allocate_sequence(),
            tool_call_id=call_id,
            name=result.tool_name,
            content=batch.process_result(result),
            content_for_user=result.content_for_user,
            is_error=not result.is_success,
        )
        batch.register_step(call_id, tool_step.id, tool_step.sequence)
        await commit_step(context, tool_step)

        if not terminated and result.termination_reason is not None:
            await set_termination_reason(result.termination_reason, result.tool_name)
            terminated = True

    await _apply_retrospect_outcome(context, batch)
    return terminated or context.is_terminal


async def _apply_retrospect_outcome(
    context: RunContext,
    batch: RetrospectBatch,
) -> None:
    outcome = await batch.finalize()
    if not outcome.applied:
        return

    replace_messages(context, outcome.messages)
    rebuilt_entry = build_messages_rebuilt_entry(
        context,
        sequence=await context.session_runtime.allocate_sequence(),
        reason="retrospect",
        messages=context.snapshot_messages(),
    )
    retrospect_entry = build_retrospect_applied_entry(
        context,
        sequence=await context.session_runtime.allocate_sequence(),
        affected_sequences=outcome.affected_sequences,
        affected_step_ids=outcome.affected_step_ids,
        feedback=outcome.feedback,
        replacement=outcome.replacement,
        trigger=outcome.trigger.value if outcome.trigger is not None else None,
    )
    await context.session_runtime.append_run_log_entries(
        [rebuilt_entry, retrospect_entry]
    )
    await context.session_runtime.publish(
        MessagesRebuiltEvent.from_context(
            context,
            reason="retrospect",
            message_count=len(context.snapshot_messages()),
        )
    )
    await context.session_runtime.publish(
        RetrospectAppliedEvent.from_context(
            context,
            affected_sequences=outcome.affected_sequences,
            affected_step_ids=outcome.affected_step_ids,
            feedback=outcome.feedback,
            replacement=outcome.replacement,
            trigger=outcome.trigger.value if outcome.trigger is not None else None,
        )
    )


__all__ = ["execute_tool_batch_cycle"]
