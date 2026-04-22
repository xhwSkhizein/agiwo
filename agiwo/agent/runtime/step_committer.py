"""Step commit pipeline for persistence, hooks, trace, and stream publication."""

from agiwo.agent.models.step import LLMCallContext, StepView
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_writer import RunStateWriter


async def commit_step(
    state: RunContext,
    step: StepView,
    *,
    llm: LLMCallContext | None = None,
    append_message: bool = True,
    track_state: bool = True,
) -> StepView:
    writer = RunStateWriter(state)
    entries = await writer.commit_step(
        step,
        append_message=append_message,
        track_state=track_state,
    )
    await state.session_runtime.project_run_log_entries(
        entries,
        run_id=state.run_id,
        agent_id=state.agent_id,
        parent_run_id=state.parent_run_id,
        depth=state.depth,
    )
    await state.hooks.on_step(step, state)
    return step


__all__ = ["commit_step"]
