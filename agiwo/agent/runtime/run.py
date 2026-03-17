"""Run domain: Run, RunOutput, RunMetrics, LLMCallContext."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from agiwo.agent.input import UserInput
from agiwo.agent.runtime.core import RunStatus, TerminationReason


@dataclass
class RunMetrics:
    """Metrics for a single Run."""

    start_at: float = 0.0
    end_at: float = 0.0
    duration_ms: float = 0.0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    token_cost: float = 0.0
    steps_count: int = 0
    tool_calls_count: int = 0
    tool_errors_count: int = 0
    first_token_latency: float | None = None
    response_latency: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_at": self.start_at,
            "end_at": self.end_at,
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "token_cost": self.token_cost,
            "steps_count": self.steps_count,
            "tool_calls_count": self.tool_calls_count,
            "tool_errors_count": self.tool_errors_count,
            "first_token_latency": self.first_token_latency,
            "response_latency": self.response_latency,
        }


@dataclass
class LLMCallContext:
    """LLM call context for observability (not persisted)."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    request_params: dict[str, Any] | None = None
    finish_reason: str | None = None


@dataclass
class Run:
    """Unified Run metadata."""

    agent_id: str
    session_id: str
    user_input: UserInput
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    status: RunStatus = RunStatus.STARTING
    response_content: str | None = None
    metrics: RunMetrics = field(default_factory=RunMetrics)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    parent_run_id: str | None = None
    trace_id: str | None = None


@dataclass
class RunOutput:
    """Execution result from Agent.run()."""

    session_id: str | None = None
    run_id: str | None = None
    response: str | None = None
    metrics: RunMetrics | None = None
    termination_reason: TerminationReason | None = None
    error: str | None = None


__all__ = [
    "LLMCallContext",
    "Run",
    "RunMetrics",
    "RunOutput",
]
