"""Helpers for executing tool batches and applying review outcomes."""

from collections.abc import Awaitable, Callable
from typing import Any

from agiwo.agent.models.run import TerminationReason
from agiwo.agent.models.step import StepView
from agiwo.agent.review import ReviewBatch
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
            content=batch.process_result(result, current_seq=seq),
            content_for_user=result.content_for_user,
            is_error=not result.is_success,
        )
        batch.register_step(call_id, tool_step.id, tool_step.sequence)
        await commit_step(tool_step)

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
    if not outcome.applied:
        return

    # Step-back already updates surviving message dicts in-place. Sync the
    # ledger list to drop temporary review_trajectory metadata without a full
    # messages rebuild.
    context.ledger.messages[:] = outcome.messages

    # Record step_back event to run log
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


__all__ = ["execute_tool_batch_cycle"]
