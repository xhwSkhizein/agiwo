"""
API-layer Pydantic models for request/response serialization.
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from agiwo.agent import UserInput
from agiwo.agent.options import AgentOptionsInput
from agiwo.config.settings import ModelProvider
from agiwo.llm.factory import ModelParamsInput
from agiwo.llm.config_policy import validate_provider_model_params
from server.domain.run_metrics import RunMetricsSummary


# ── Session / Run / Step Responses ──────────────────────────────────────


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


class StepResponse(BaseModel):
    id: str
    session_id: str
    run_id: str
    sequence: int
    role: str
    agent_id: str | None = None
    content: object | None = None
    content_for_user: str | None = None
    reasoning_content: str | None = None
    user_input: UserInput | None = None
    tool_calls: list[dict[str, object]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metrics: StepMetricsResponse | None = None
    created_at: str | None = None
    parent_run_id: str | None = None
    depth: int = 0


class RunResponse(BaseModel):
    id: str
    agent_id: str
    session_id: str
    user_id: str | None = None
    user_input: UserInput
    status: str
    response_content: str | None = None
    metrics: RunMetricsResponse | None = None
    created_at: str | None = None
    updated_at: str | None = None
    parent_run_id: str | None = None


# ── Trace Responses ─────────────────────────────────────────────────────


class SpanResponse(BaseModel):
    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    kind: str
    name: str
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: float | None = None
    status: str
    error_message: str | None = None
    depth: int = 0
    attributes: dict[str, Any] = Field(default_factory=dict)
    input_preview: str | None = None
    output_preview: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    llm_details: dict[str, Any] | None = None
    tool_details: dict[str, Any] | None = None
    run_id: str | None = None
    step_id: str | None = None


class TraceResponse(BaseModel):
    trace_id: str
    agent_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: float | None = None
    status: str
    root_span_id: str | None = None
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_token_cost: float = 0.0
    max_depth: int = 0
    input_query: str | None = None
    final_output: str | None = None
    spans: list[SpanResponse] = Field(default_factory=list)


class TraceListItem(BaseModel):
    trace_id: str
    agent_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    start_time: str | None = None
    duration_ms: float | None = None
    status: str
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_token_cost: float = 0.0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    input_query: str | None = None
    final_output: str | None = None


# ── Agent Config ────────────────────────────────────────────────────────

class AgentConfigPayload(BaseModel):
    name: str
    description: str = ""
    model_provider: ModelProvider
    model_name: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: AgentOptionsInput = Field(default_factory=AgentOptionsInput)
    model_params: ModelParamsInput = Field(default_factory=ModelParamsInput)

    @model_validator(mode="after")
    def _validate_model_connection(self) -> "AgentConfigPayload":
        validate_provider_model_params(self.model_provider, self.model_params)
        return self


class AgentConfigResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    model_provider: str
    model_name: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: AgentOptionsInput = Field(default_factory=AgentOptionsInput)
    model_params: ModelParamsInput = Field(default_factory=ModelParamsInput)
    created_at: str
    updated_at: str


# ── Scheduler ──────────────────────────────────────────────────────────


class WakeConditionResponse(BaseModel):
    type: str
    wait_for: list[str] = Field(default_factory=list)
    wait_mode: str = "all"
    completed_ids: list[str] = Field(default_factory=list)
    time_value: float | None = None
    time_unit: str | None = None
    wakeup_at: str | None = None
    submitted_task: UserInput | None = None
    timeout_at: str | None = None


class AgentStateResponse(BaseModel):
    id: str
    session_id: str
    status: str
    task: UserInput
    parent_id: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    wake_condition: WakeConditionResponse | None = None
    result_summary: str | None = None
    signal_propagated: bool = False
    agent_config_id: str | None = None
    is_persistent: bool = False
    depth: int = 0
    wake_count: int = 0
    metrics: RunMetricsSummary = Field(default_factory=RunMetricsSummary)
    created_at: str | None = None
    updated_at: str | None = None


class AgentStateListItem(BaseModel):
    id: str
    status: str
    task: UserInput
    parent_id: str | None = None
    wake_condition: WakeConditionResponse | None = None
    result_summary: str | None = None
    agent_config_id: str | None = None
    is_persistent: bool = False
    depth: int = 0
    wake_count: int = 0
    metrics: RunMetricsSummary = Field(default_factory=RunMetricsSummary)
    created_at: str | None = None
    updated_at: str | None = None


class SchedulerStatsResponse(BaseModel):
    total: int
    pending: int
    running: int
    sleeping: int
    completed: int
    failed: int


# ── Scheduler Control ──────────────────────────────────────────────────


class SteerRequest(BaseModel):
    message: str
    urgent: bool = False


class CancelRequest(BaseModel):
    reason: str = "Cancelled by operator"


class ResumeRequest(BaseModel):
    message: str


class CreateAgentRequest(BaseModel):
    agent_config_id: str | None = None
    initial_task: str | None = None
    session_id: str | None = None


class PendingEventResponse(BaseModel):
    id: str
    target_agent_id: str
    source_agent_id: str | None = None
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class StepDeltaResponse(BaseModel):
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None


class StreamEventPayload(BaseModel):
    type: str
    run_id: str
    delta: StepDeltaResponse | None = None
    step: StepResponse | None = None
    data: dict[str, Any] | None = None
    agent_id: str | None = None
    span_id: str | None = None
    parent_run_id: str | None = None
    parent_span_id: str | None = None
    trace_id: str | None = None
    step_id: str | None = None
    depth: int = 0
    timestamp: str | None = None


# ── Chat ────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


# ── Generic ─────────────────────────────────────────────────────────────


class PaginatedResponse(BaseModel):
    items: list[Any]
    total: int
    limit: int
    offset: int
