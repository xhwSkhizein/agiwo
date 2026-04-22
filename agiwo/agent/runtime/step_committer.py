"""Step commit pipeline for persistence, hooks, trace, and stream publication."""

from agiwo.agent.models.step import LLMCallContext, StepView
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_ops import track_step_state
from agiwo.agent.runtime.state_writer import build_step_log_entry
from agiwo.agent.models.stream import StepCompletedEvent


async def commit_step(
    state: RunContext,
    step: StepView,
    *,
    llm: LLMCallContext | None = None,
    append_message: bool = True,
    track_state: bool = True,
) -> StepView:
    if track_state:
        track_step_state(state, step, append_message=append_message)
    await state.session_runtime.append_run_log_entries([build_step_log_entry(step)])
    if state.session_runtime.trace_runtime is not None:
        await state.session_runtime.trace_runtime.on_step(step, llm)
    await state.hooks.on_step(step, state)
    await state.session_runtime.publish(
        StepCompletedEvent.from_context(state, step=step),
    )
    return step


__all__ = ["commit_step"]
