"""Helpers for executing tool batches and applying introspection outcomes."""

from collections.abc import Awaitable, Callable
from typing import Any

from agiwo.agent.introspect.apply import (
    apply_introspection_outcome,
    build_tool_step_lookup,
    register_committed_tool_step,
)
from agiwo.agent.introspect.goal import handle_goal_tool_result
from agiwo.agent.introspect.models import IntrospectionOutcome
from agiwo.agent.introspect.trajectory import (
    append_system_review_notice,
    has_prompt_visible_system_review,
    maybe_build_introspection_notice,
    parse_introspection_outcome,
)
from agiwo.agent.models.run import TerminationReason
from agiwo.agent.models.step import StepView
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
    """Execute one tool batch and apply introspection/context-repair outcomes."""
    writer = RunStateWriter(context)
    tool_results = await execute_tool_batch(
        tool_calls,
        tools_map=runtime.tools_map,
        context=context,
        abort_signal=runtime.abort_signal,
    )
    terminated = False
    introspection_enabled = _introspection_enabled(context, runtime)
    pending_outcome: IntrospectionOutcome | None = None
    step_lookup: dict[str, dict[str, Any]] = {}

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
        goal_update = None
        introspection_notice = None
        if introspection_enabled:
            goal_update = handle_goal_tool_result(
                result,
                context.ledger.goal,
                current_seq=seq,
            )
            if goal_update is not None and goal_update.milestone_switch:
                context.ledger.introspection.pending_milestone_switch = True

            pending_outcome = (
                parse_introspection_outcome(
                    result,
                    context.ledger.goal,
                    current_seq=seq,
                    assistant_step_id=assistant_step_id,
                    tool_step_id=tool_step.id,
                )
                or pending_outcome
            )

            introspection_notice = maybe_build_introspection_notice(
                result,
                context.ledger.goal,
                context.ledger.introspection,
                step_interval=context.config.review_step_interval,
                review_on_error=context.config.review_on_error,
                has_visible_notice=has_prompt_visible_system_review(
                    context.ledger.messages
                ),
            )
            if introspection_notice is not None:
                review_advice = await context.hooks.before_review(
                    trigger_reason=introspection_notice.trigger_reason,
                    milestone=introspection_notice.active_milestone,
                    step_count=introspection_notice.step_count,
                    context=context,
                )
                tool_step.content = append_system_review_notice(
                    result.content or "",
                    introspection_notice.active_milestone,
                    introspection_notice.step_count,
                    trigger_reason=introspection_notice.trigger_reason,
                    review_advice=review_advice,
                )

        committed_step = await commit_step(tool_step)
        register_committed_tool_step(
            step_lookup,
            tool_call_id=call_id,
            step=committed_step,
        )
        if goal_update is not None:
            await writer.record_goal_milestones_updated(
                milestones=goal_update.milestones,
                active_milestone_id=goal_update.active_milestone_id,
                source_tool_call_id=goal_update.source_tool_call_id,
                source_step_id=committed_step.id,
                reason=goal_update.reason,
            )
        if introspection_notice is not None:
            await writer.record_introspection_triggered(
                trigger_reason=introspection_notice.trigger_reason,
                active_milestone_id=introspection_notice.active_milestone.id
                if introspection_notice.active_milestone is not None
                else None,
                review_count_since_boundary=introspection_notice.step_count,
                trigger_tool_call_id=result.tool_call_id or None,
                trigger_tool_step_id=committed_step.id,
                notice_step_id=committed_step.id,
            )

        if not terminated and result.termination_reason is not None:
            await set_termination_reason(result.termination_reason, result.tool_name)
            terminated = True

    if pending_outcome is not None:
        previous_boundary_seq = context.ledger.introspection.last_boundary_seq
        full_lookup = await build_tool_step_lookup(
            context,
            step_lookup,
            start_seq=previous_boundary_seq + 1,
            end_seq=pending_outcome.boundary_seq,
        )
        await apply_introspection_outcome(
            context,
            pending_outcome,
            writer=writer,
            step_lookup=full_lookup,
        )
    return terminated or context.is_terminal


def _introspection_enabled(context: RunContext, runtime: RunRuntime) -> bool:
    return (
        context.config.enable_goal_directed_review
        and "review_trajectory" in runtime.tools_map
        and "declare_milestones" in runtime.tools_map
    )


__all__ = ["execute_tool_batch_cycle"]
