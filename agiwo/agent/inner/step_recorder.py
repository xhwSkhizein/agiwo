from collections.abc import Sequence
from datetime import datetime, timezone

from agiwo.agent.hooks import AgentHooks
from agiwo.agent.inner.context import AgentRunContext
from agiwo.agent.inner.event_emitter import EventEmitter
from agiwo.agent.inner.run_state import RunState
from agiwo.agent.runtime import LLMCallContext, StepMetrics, StepRecord
from agiwo.agent.scheduler_port import StepObserver


class StepRecorder:
    """Single owner for sequence allocation and committed step side effects."""

    def __init__(
        self,
        *,
        context: AgentRunContext,
        emitter: EventEmitter,
        hooks: AgentHooks,
        step_observers: Sequence[StepObserver],
        state: RunState | None = None,
    ) -> None:
        self.context = context
        self.emitter = emitter
        self.hooks = hooks
        self._step_observers = list(step_observers)
        self._state = state

    def attach_state(self, state: RunState) -> "StepRecorder":
        return StepRecorder(
            context=self.context,
            emitter=self.emitter,
            hooks=self.hooks,
            step_observers=self._step_observers,
            state=state,
        )

    async def next_sequence(self) -> int:
        return await self.context.next_sequence()

    async def create_user_step(
        self,
        *,
        user_input=None,
        content=None,
        name: str | None = None,
    ) -> StepRecord:
        sequence = await self.next_sequence()
        return StepRecord.user(
            self.context,
            sequence=sequence,
            user_input=user_input,
            content=content,
            name=name,
        )

    async def create_assistant_step(self) -> StepRecord:
        sequence = await self.next_sequence()
        return StepRecord.assistant(
            self.context,
            sequence=sequence,
            content="",
            tool_calls=None,
            metrics=StepMetrics(
                start_at=datetime.now(timezone.utc),
            ),
        )

    async def create_tool_step(
        self,
        *,
        tool_call_id: str,
        name: str,
        content: str,
        content_for_user: str | None = None,
    ) -> StepRecord:
        sequence = await self.next_sequence()
        return StepRecord.tool(
            self.context,
            sequence=sequence,
            tool_call_id=tool_call_id,
            name=name,
            content=content,
            content_for_user=content_for_user,
        )

    async def record(
        self,
        step: StepRecord,
        *,
        llm: LLMCallContext | None = None,
        append_message: bool = True,
    ) -> StepRecord:
        await self.emitter.emit_step_completed(step, llm=llm)
        if self._state is not None:
            self._state.track_step(step, append_message=append_message)
        if self.hooks.on_step:
            await self.hooks.on_step(step)
        for observer in list(self._step_observers):
            await observer(step)
        return step


__all__ = ["StepRecorder"]
