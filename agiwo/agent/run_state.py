import time
from dataclasses import dataclass, field

from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.config_options import AgentConfigOptions
from agiwo.agent.schema import (
    StepRecord,
    step_to_message,
    RunOutput,
    RunMetrics,
    TerminationReason,
)


@dataclass
class RunState:
    """Encapsulates all mutable state for a single execution run."""

    context: ExecutionContext
    config: AgentConfigOptions
    messages: list[dict]
    tool_schemas: list[dict] | None = None
    start_time: float = field(default_factory=time.time)
    current_step: int = 0
    termination_reason: TerminationReason | None = None
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    steps_count: int = 0
    tool_calls_count: int = 0
    assistant_steps_count: int = 0
    pending_tool_calls: list[dict] | None = None
    response_content: str | None = None

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def track_step(self, step: StepRecord, *, append_message: bool = True) -> None:
        """Track metrics and optionally append to messages."""
        self._track_step(step)
        if append_message:
            self.messages.append(step_to_message(step))

    def build_output(self) -> RunOutput:
        return RunOutput(
            response=self.response_content,
            run_id=self.context.run_id,
            session_id=self.context.session_id,
            metrics=RunMetrics(
                duration_ms=self.elapsed * 1000,
                total_tokens=self.total_tokens,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                steps_count=self.steps_count,
                tool_calls_count=self.tool_calls_count,
            ),
            termination_reason=self.termination_reason,
        )

    def _track_step(self, step: StepRecord) -> None:
        self.steps_count += 1

        self._track_tokens(step)

        if step.is_assistant_step():
            self._track_assistant(step)
            return

        if step.is_tool_step():
            self._track_tool()

    def _track_tokens(self, step: StepRecord) -> None:
        if not step.metrics:
            return

        if step.metrics.total_tokens is not None:
            self.total_tokens += step.metrics.total_tokens
        if step.metrics.input_tokens is not None:
            self.input_tokens += step.metrics.input_tokens
        if step.metrics.output_tokens is not None:
            self.output_tokens += step.metrics.output_tokens

    def _track_assistant(self, step: StepRecord) -> None:
        self.assistant_steps_count += 1
        if step.content:
            self.response_content = step.content
        if step.tool_calls:
            self.tool_calls_count += len(step.tool_calls)
            self.pending_tool_calls = step.tool_calls

    def _track_tool(self) -> None:
        self.pending_tool_calls = None
