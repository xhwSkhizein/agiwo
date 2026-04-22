"""Step commit pipeline for persistence, hooks, trace, and stream publication."""

from agiwo.agent.models.step import LLMCallContext, StepView
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_writer import RunStateWriter
from agiwo.agent.models.stream import StepCompletedEvent


async def commit_step(
    state: RunContext,
    step: StepView,
    *,
    llm: LLMCallContext | None = None,
    append_message: bool = True,
    track_state: bool = True,
) -> StepView:
    writer = RunStateWriter(state)
    await writer.commit_step(
        step,
        append_message=append_message,
        track_state=track_state,
    )
    if state.session_runtime.trace_runtime is not None:
        await state.session_runtime.trace_runtime.on_step(step, llm)
    await state.hooks.on_step(step, state)
    await state.session_runtime.publish(
        StepCompletedEvent.from_context(state, step=step),
    )
    return step


__all__ = ["commit_step"]
