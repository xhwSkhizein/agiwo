from collections.abc import Sequence
from datetime import datetime, timezone

from agiwo.agent.hooks import AgentHooks
from agiwo.agent.engine.context import AgentRunContext
from agiwo.agent.engine.state import RunState
from agiwo.agent.runtime import (
    LLMCallContext,
    Run,
    RunCompletedEvent,
    RunFailedEvent,
    RunOutput,
    RunStartedEvent,
    RunStatus,
    StepCompletedEvent,
    StepDelta,
    StepDeltaEvent,
    StepMetrics,
    StepRecord,
    TerminationReason,
)
from agiwo.agent.scheduler_port import StepObserver


class RunRecorder:
    """Single owner for run lifecycle, step commits, storage, and fanout."""

    def __init__(
        self,
        *,
        context: AgentRunContext,
        hooks: AgentHooks,
        step_observers: Sequence[StepObserver],
        state: RunState | None = None,
    ) -> None:
        self.context = context
        self.hooks = hooks
        self._step_observers = list(step_observers)
        self._state = state
        self._run: Run | None = None

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
            metrics=StepMetrics(start_at=datetime.now(timezone.utc)),
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

    async def start_run(self, run: Run) -> None:
        self._run = run
        run.trace_id = self.context.trace_id
        await self.context.session_runtime.run_step_storage.save_run(run)
        if self.context.session_runtime.trace_runtime is not None:
            self.context.session_runtime.trace_runtime.on_run_started(run)
        await self.context.session_runtime.publish(
            RunStartedEvent(
                session_id=self.context.session_id,
                run_id=self.context.run_id,
                agent_id=self.context.agent_id,
                parent_run_id=self.context.parent_run_id,
                depth=self.context.depth,
            )
        )

    async def publish_delta(self, step_id: str, delta: StepDelta) -> None:
        await self.context.session_runtime.publish(
            StepDeltaEvent(
                session_id=self.context.session_id,
                run_id=self.context.run_id,
                agent_id=self.context.agent_id,
                parent_run_id=self.context.parent_run_id,
                depth=self.context.depth,
                step_id=step_id,
                delta=delta,
            )
        )

    async def commit_step(
        self,
        step: StepRecord,
        *,
        llm: LLMCallContext | None = None,
        append_message: bool = True,
        track_state: bool = True,
    ) -> StepRecord:
        if self._state is not None and track_state:
            self._state.track_step(step, append_message=append_message)
        await self.context.session_runtime.run_step_storage.save_step(step)
        if self.context.session_runtime.trace_runtime is not None:
            await self.context.session_runtime.trace_runtime.on_step(step, llm)
        if self.hooks.on_step is not None:
            await self.hooks.on_step(step)
        for observer in list(self._step_observers):
            await observer(step)
        await self.context.session_runtime.publish(
            StepCompletedEvent(
                session_id=self.context.session_id,
                run_id=self.context.run_id,
                agent_id=self.context.agent_id,
                parent_run_id=self.context.parent_run_id,
                depth=self.context.depth,
                step=step,
            )
        )
        return step

    async def complete_run(self, result: RunOutput) -> None:
        run = self._require_run()
        run.status = (
            RunStatus.CANCELLED
            if result.termination_reason == TerminationReason.CANCELLED
            else RunStatus.COMPLETED
        )
        run.response_content = result.response
        run.updated_at = datetime.now()
        _apply_run_output_metrics(run, result)
        await self.context.session_runtime.run_step_storage.save_run(run)
        if self.context.session_runtime.trace_runtime is not None:
            await self.context.session_runtime.trace_runtime.on_run_completed(
                result,
                run_id=run.id,
            )
        await self.context.session_runtime.publish(
            RunCompletedEvent(
                session_id=self.context.session_id,
                run_id=self.context.run_id,
                agent_id=self.context.agent_id,
                parent_run_id=self.context.parent_run_id,
                depth=self.context.depth,
                response=result.response,
                metrics=result.metrics,
                termination_reason=result.termination_reason,
            )
        )

    async def fail_run(self, error: Exception) -> None:
        run = self._require_run()
        run.status = RunStatus.FAILED
        run.updated_at = datetime.now()
        run.metrics.end_at = datetime.now(timezone.utc).timestamp()
        await self.context.session_runtime.run_step_storage.save_run(run)
        if self.context.session_runtime.trace_runtime is not None:
            await self.context.session_runtime.trace_runtime.on_run_failed(
                error,
                run_id=run.id,
            )
        await self.context.session_runtime.publish(
            RunFailedEvent(
                session_id=self.context.session_id,
                run_id=self.context.run_id,
                agent_id=self.context.agent_id,
                parent_run_id=self.context.parent_run_id,
                depth=self.context.depth,
                error=str(error),
            )
        )

    def _require_run(self) -> Run:
        if self._run is None:
            raise RuntimeError("run_not_started")
        return self._run


def _apply_run_output_metrics(run: Run, result: RunOutput) -> None:
    run.metrics.end_at = datetime.now(timezone.utc).timestamp()
    if result.metrics is None:
        return
    for field_name in (
        "duration_ms",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "token_cost",
        "steps_count",
        "tool_calls_count",
        "tool_errors_count",
        "first_token_latency",
        "response_latency",
    ):
        setattr(run.metrics, field_name, getattr(result.metrics, field_name))


__all__ = ["RunRecorder"]
