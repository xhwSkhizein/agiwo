"""Metrics models for runs, steps, and scheduler states."""

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from agiwo.agent import StepMetrics


class RunMetricsSummary(BaseModel):
    run_count: int = 0
    completed_run_count: int = 0
    step_count: int = 0
    tool_calls_count: int = 0
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    token_cost: float = 0.0


class RunMetricsResponse(BaseModel):
    duration_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    token_cost: float | None = None
    steps_count: int | None = None
    tool_calls_count: int | None = None


class StepMetricsResponse(RunMetricsResponse):
    usage_source: str | None = None
    model_name: str | None = None
    provider: str | None = None
    first_token_latency_ms: float | None = None

    @classmethod
    def from_sdk(cls, metrics: "StepMetrics") -> "StepMetricsResponse":
        return cls(
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            total_tokens=metrics.total_tokens,
            usage_source=metrics.usage_source,
            model_name=metrics.model_name,
            provider=metrics.provider,
            first_token_latency_ms=metrics.first_token_latency_ms,
        )
