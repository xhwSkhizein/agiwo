from collections.abc import Sequence
from typing import Any

from agiwo.agent.compact_types import CompactMetadata
from agiwo.agent.run_state import RunContext
from agiwo.agent.state_tracking import track_step_state
from agiwo.agent.types import LLMCallContext, StepCompletedEvent, StepRecord


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


def replace_messages(
    state: RunContext,
    messages: Sequence[dict[str, Any]],
    *,
    compact_metadata: CompactMetadata | None = None,
) -> None:
    state.messages = list(messages)
    if compact_metadata is not None:
        state.last_compact_metadata = compact_metadata
