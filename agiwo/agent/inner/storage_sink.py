import time
from typing import AsyncIterator

from agiwo.agent.schema import EventType, Run, RunStatus, StreamEvent
from agiwo.agent.storage.base import RunStepStorage
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class StorageSink:
    """
    Event stream middleware that persists Run/Step data.

    Consumes StreamEvents and writes to RunStepStorage,
    then yields events unchanged for downstream consumers.

    Handles events from both root and nested agents (which share
    the same StreamChannel). Tracks multiple Run objects.
    """

    def __init__(self, storage: RunStepStorage, root_run: Run) -> None:
        self.storage = storage
        self._runs: dict[str, Run] = {root_run.id: root_run}

    async def wrap_stream(
        self, stream: AsyncIterator[StreamEvent]
    ) -> AsyncIterator[StreamEvent]:
        async for event in stream:
            await self._persist(event)
            yield event

    async def _persist(self, event: StreamEvent) -> None:
        try:
            if event.type == EventType.RUN_STARTED:
                await self._handle_run_started(event)

            elif event.type == EventType.STEP_COMPLETED and event.step:
                await self.storage.save_step(event.step)

            elif event.type == EventType.RUN_COMPLETED:
                await self._handle_run_completed(event)

            elif event.type == EventType.RUN_FAILED:
                await self._handle_run_failed(event)

        except Exception as e:
            logger.warning(
                "storage_sink_persist_failed",
                event_type=event.type.value,
                run_id=event.run_id,
                error=str(e),
            )

    async def _handle_run_started(self, event: StreamEvent) -> None:
        run = self._runs.get(event.run_id)
        if run is None:
            data = event.data or {}
            run = Run(
                id=event.run_id,
                agent_id=event.agent_id or "",
                session_id=data.get("session_id", ""),
                user_input=data.get("query", ""),
                status=RunStatus.RUNNING,
                parent_run_id=event.parent_run_id,
            )
            run.metrics.start_at = time.time()
            self._runs[event.run_id] = run
        await self.storage.save_run(run)

    async def _handle_run_completed(self, event: StreamEvent) -> None:
        run = self._runs.get(event.run_id)
        if run is None:
            return
        data = event.data or {}
        run.status = RunStatus.COMPLETED
        run.response_content = data.get("response")
        run.metrics.end_at = time.time()
        metrics: dict = data.get("metrics")
        if metrics:
            run.metrics.duration_ms = metrics.get("duration", 0)
            run.metrics.total_tokens = metrics.get("total_tokens", 0)
            run.metrics.input_tokens = metrics.get("input_tokens", 0)
            run.metrics.output_tokens = metrics.get("output_tokens", 0)
            run.metrics.tool_calls_count = metrics.get("tool_calls_count", 0)
        await self.storage.save_run(run)

    async def _handle_run_failed(self, event: StreamEvent) -> None:
        run = self._runs.get(event.run_id)
        if run is None:
            return
        run.status = RunStatus.FAILED
        run.metrics.end_at = time.time()
        await self.storage.save_run(run)


__all__ = ["StorageSink"]
