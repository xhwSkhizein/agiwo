import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.config_options import AgentConfigOptions
from agiwo.agent.schema import Step, StepDelta, StepAdapter, RunOutput, RunMetrics
from agiwo.agent.side_effect_processor import SideEffectProcessor
from agiwo.agent.step_factory import StepFactory

if TYPE_CHECKING:
    from typing import Self


class MetricsTracker:
    """Internal metrics tracker for aggregating execution statistics."""

    def __init__(self) -> None:
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.steps_count = 0
        self.tool_calls_count = 0
        self.assistant_steps_count = 0
        self.pending_tool_calls = None
        self.response_content = None

    def track(self, step: Step) -> None:
        self.steps_count += 1

        self._track_tokens(step)

        if step.is_assistant_step():
            self._track_assistant(step)
            return

        if step.role.value == "tool":
            self._track_tool()

    def _track_tokens(self, step: Step) -> None:
        if not step.metrics:
            return

        if step.metrics.total_tokens is not None:
            self.total_tokens += step.metrics.total_tokens
        if step.metrics.input_tokens is not None:
            self.input_tokens += step.metrics.input_tokens
        if step.metrics.output_tokens is not None:
            self.output_tokens += step.metrics.output_tokens

    def _track_assistant(self, step: Step) -> None:
        self.assistant_steps_count += 1
        if step.content:
            self.response_content = step.content
        if step.tool_calls:
            self.tool_calls_count += len(step.tool_calls)
            self.pending_tool_calls = step.tool_calls

    def _track_tool(self) -> None:
        self.pending_tool_calls = None


@dataclass
class RunState:
    """Encapsulates all mutable state for a single execution run."""

    context: ExecutionContext
    config: AgentConfigOptions
    messages: list[dict]
    pipeline: SideEffectProcessor
    tracker: MetricsTracker
    sf: StepFactory
    tool_schemas: list[dict] | None = None
    start_time: float = field(default_factory=time.time)
    current_step: int = 0
    termination_reason: str | None = None

    @classmethod
    def create(
        cls,
        context: ExecutionContext,
        config: AgentConfigOptions,
        messages: list[dict],
        pipeline: SideEffectProcessor,
        tool_schemas: list[dict] | None = None,
    ) -> "Self":
        return cls(
            context=context,
            config=config,
            messages=messages,
            pipeline=pipeline,
            tracker=MetricsTracker(),
            sf=StepFactory(context),
            tool_schemas=tool_schemas,
        )

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    async def record_step(self, step: Step, *, append_message: bool = True) -> None:
        """Queue for persistence, track metrics, emit event, optionally append to messages."""
        await self.pipeline.commit_step(step)
        self.tracker.track(step)
        if append_message:
            self.messages.append(StepAdapter.to_llm_message(step))

    async def emit_delta(self, step_id: str, delta: StepDelta) -> None:
        await self.pipeline.emit_step_delta(step_id, delta)

    def build_output(self) -> RunOutput:
        return RunOutput(
            response=self.tracker.response_content,
            run_id=self.context.run_id,
            session_id=self.context.session_id,
            metrics=RunMetrics(
                duration_ms=self.elapsed * 1000,
                total_tokens=self.tracker.total_tokens,
                input_tokens=self.tracker.input_tokens,
                output_tokens=self.tracker.output_tokens,
                steps_count=self.tracker.steps_count,
                tool_calls_count=self.tracker.tool_calls_count,
            ),
            termination_reason=self.termination_reason,
        )

    async def cleanup(self) -> None:
        pass
