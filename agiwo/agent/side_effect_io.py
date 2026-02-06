from agiwo.agent.schema import (
    EventType,
    LLMCallContext,
    Run,
    RunOutput,
    StepDelta,
    StepRecord,
    StreamEvent,
)
from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.session.base import SessionStore


class SideEffectIO:
    """
    Side-effect handler for Agent execution.

    Responsibilities:
    1. Event emission
    2. Persistence
    3. Sequence allocation
    """

    def __init__(
        self,
        context: ExecutionContext,
        session_store: SessionStore,
    ) -> None:
        self.context = context
        self.session_store = session_store

    async def allocate_sequence(self) -> int:
        """Allocate next sequence number for the current session."""
        if "seq_start" in self.context.metadata:
            return int(self.context.metadata.pop("seq_start"))
        return await self.session_store.allocate_sequence(self.context.session_id)

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

    async def _write_event(self, event: StreamEvent) -> None:
        await self.context.wire.write(event)

    async def emit_run_started(self, run: Run) -> None:
        """Emit RUN_STARTED event."""
        event = self._make_event(
            event_type=EventType.RUN_STARTED,
            data={"query": run.input_query, "session_id": run.session_id},
        )
        await self._write_event(event)
        await self.session_store.save_run(run)

    async def emit_step_delta(self, step_id: str, delta: StepDelta) -> None:
        """Emit STEP_DELTA event."""
        event = self._make_event(
            event_type=EventType.STEP_DELTA,
            step_id=step_id,
            delta=delta,
        )
        await self._write_event(event)

    async def commit_step(
        self, step: StepRecord, llm: LLMCallContext | None = None
    ) -> None:
        """Emit STEP_COMPLETED event and persist step."""
        event = self._make_event(
            event_type=EventType.STEP_COMPLETED,
            step_id=step.id,
            step=step,
            llm=llm,
        )
        await self._write_event(event)
        await self.session_store.save_step(step)

    async def emit_run_completed(self, run: Run, output: RunOutput) -> None:
        """Emit RUN_COMPLETED event."""
        data = {
            "response": output.response or "",
            "metrics": {
                "duration": run.metrics.duration_ms,
                "total_tokens": run.metrics.total_tokens,
            },
        }
        if output.termination_reason:
            data["termination_reason"] = output.termination_reason
        event = self._make_event(
            event_type=EventType.RUN_COMPLETED,
            data=data,
        )
        await self._write_event(event)
        await self.session_store.save_run(run)

    async def emit_run_failed(self, run: Run, error: Exception) -> None:
        """Emit RUN_FAILED event."""
        event = self._make_event(
            event_type=EventType.RUN_FAILED,
            data={"error": str(error)},
        )
        await self._write_event(event)
        await self.session_store.save_run(run)


__all__ = ["SideEffectIO"]
