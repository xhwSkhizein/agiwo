"""Step commit pipeline for persistence, hooks, trace, and stream publication."""

from agiwo.agent.models.step import LLMCallContext, StepRecord
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_ops import track_step_state
from agiwo.agent.models.stream import StepCompletedEvent


async def commit_step(
    state: RunContext,
    step: StepRecord,
    *,
    llm: LLMCallContext | None = None,
    append_message: bool = True,
    track_state: bool = True,
) -> StepRecord:
    if track_state:
        track_step_state(state, step, append_message=append_message)
    await state.session_runtime.run_step_storage.save_step(step)
    if state.session_runtime.trace_runtime is not None:
        await state.session_runtime.trace_runtime.on_step(step, llm)
    if state.hooks.on_step is not None:
        await state.hooks.on_step(step)
    await state.session_runtime.publish(
        StepCompletedEvent.from_context(state, step=step),
    )
    return step


__all__ = ["commit_step"]
