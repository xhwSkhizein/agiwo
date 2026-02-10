from agiwo.agent.schema import (
    EventType,
    LLMCallContext,
    StepDelta,
    StepRecord,
    StreamEvent,
)
from agiwo.agent.execution_context import ExecutionContext


class EventEmitter:
    """
    Pure event emitter for Agent execution.

    Only responsibility: emit StreamEvents to the StreamChannel.
    No storage, no sequence allocation, no side effects beyond event emission.
    """

    def __init__(self, context: ExecutionContext) -> None:
        self.context = context

    def _make_event(
        self,
        *,
        event_type: EventType,
        step_id: str | None = None,
        delta: StepDelta | None = None,
        step: StepRecord | None = None,
        llm: LLMCallContext | None = None,
        data: dict | None = None,
    ) -> StreamEvent:
        return StreamEvent(
            type=event_type,
            run_id=self.context.run_id,
            step_id=step_id,
            delta=delta,
            step=step,
            llm=llm,
            data=data,
            depth=self.context.depth,
            parent_run_id=self.context.parent_run_id,
            agent_id=self.context.agent_id,
        )

    async def emit_run_started(self, data: dict) -> None:
        event = self._make_event(event_type=EventType.RUN_STARTED, data=data)
        await self.context.channel.write(event)

    async def emit_step_delta(self, step_id: str, delta: StepDelta) -> None:
        event = self._make_event(
            event_type=EventType.STEP_DELTA,
            step_id=step_id,
            delta=delta,
        )
        await self.context.channel.write(event)

    async def emit_step_completed(
        self, step: StepRecord, llm: LLMCallContext | None = None
    ) -> None:
        event = self._make_event(
            event_type=EventType.STEP_COMPLETED,
            step_id=step.id,
            step=step,
            llm=llm,
        )
        await self.context.channel.write(event)

    async def emit_run_completed(self, data: dict) -> None:
        event = self._make_event(event_type=EventType.RUN_COMPLETED, data=data)
        await self.context.channel.write(event)

    async def emit_run_failed(self, error: Exception) -> None:
        event = self._make_event(
            event_type=EventType.RUN_FAILED,
            data={"error": str(error)},
        )
        await self.context.channel.write(event)


__all__ = ["EventEmitter"]
