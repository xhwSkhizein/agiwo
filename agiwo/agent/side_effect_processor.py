from agiwo.agent.schema import Run, RunOutput, Step, StepDelta
from agiwo.observability.collector import TraceCollector
from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.event_factory import EventFactory
from agiwo.agent.sequence_manager import SequenceManager
from agiwo.agent.session.base import SessionStore
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SideEffectProcessor:
    """
    Side effect processor for Agent execution.

    Responsible for:
    1. Event emission (using EventFactory)
    2. Direct side effects: event streaming, persistence, observability
    3. Sequence number allocation
    """

    def __init__(
        self,
        context: ExecutionContext,
        session_store: SessionStore | None = None,
        trace_collector: TraceCollector | None = None,
    ) -> None:
        self.context = context
        self.event_factory = EventFactory(context)
        self.sequence_manager = (
            SequenceManager(session_store) if session_store else None
        )
        self.session_store = session_store
        self.trace_collector = trace_collector

    async def allocate_sequence(self) -> int:
        """Allocate next sequence number for the current session."""
        if self.sequence_manager:
            return await self.sequence_manager.allocate(
                self.context.session_id, self.context
            )
        return 1

    async def emit_run_started(self, run: Run) -> None:
        """Emit RUN_STARTED event."""
        event = self.event_factory.run_started(run.input_query)

        # Event streaming
        await self.context.wire.write(event)

        # Persistence
        if self.session_store:
            await self.session_store.save_run(run)

        # Observability
        if self.trace_collector:
            self.trace_collector.start(
                trace_id=event.trace_id,
                agent_id=run.agent_id,
                session_id=run.session_id,
                user_id=event.data.get("user_id") if event.data else None,
                input_query=run.input_query,
            )
            await self.trace_collector.collect(event)

    async def emit_step_delta(self, step_id: str, delta: StepDelta) -> None:
        """Emit STEP_DELTA event."""
        event = self.event_factory.step_delta(step_id, delta)

        # Event streaming
        await self.context.wire.write(event)

        # Observability
        if self.trace_collector:
            await self.trace_collector.collect(event)

    async def commit_step(self, step: Step) -> None:
        """Emit STEP_COMPLETED event."""
        event = self.event_factory.step_completed(step.id, step)

        # Event streaming
        await self.context.wire.write(event)

        # Persistence
        if self.session_store:
            await self.session_store.save_step(step)

        # Observability
        if self.trace_collector:
            await self.trace_collector.collect(event)

    async def emit_run_completed(self, run: Run, output: RunOutput) -> None:
        """Emit RUN_COMPLETED event."""
        event = self.event_factory.run_completed(
            response=output.response or "",
            metrics={
                "duration": run.metrics.duration_ms,
                "total_tokens": run.metrics.total_tokens,
            },
            termination_reason=output.termination_reason,
        )

        # Event streaming
        await self.context.wire.write(event)

        # Persistence
        if self.session_store:
            await self.session_store.save_run(run)

        # Observability
        if self.trace_collector:
            await self.trace_collector.collect(event)
            await self.trace_collector.stop()

    async def emit_run_failed(self, run: Run, error: Exception) -> None:
        """Emit RUN_FAILED event."""
        event = self.event_factory.run_failed(str(error))

        # Event streaming
        await self.context.wire.write(event)

        # Persistence
        if self.session_store:
            await self.session_store.save_run(run)

        # Observability
        if self.trace_collector:
            await self.trace_collector.collect(event)
            self.trace_collector.fail(error)
            await self.trace_collector.stop()
