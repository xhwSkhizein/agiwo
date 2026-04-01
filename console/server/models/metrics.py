"""Shared metrics models used outside the API boundary."""

from pydantic import BaseModel


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
