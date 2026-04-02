"""Console API view models."""

import json
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from agiwo.agent import UserInput
from agiwo.config.settings import ModelProvider
from agiwo.llm.config_policy import validate_provider_model_params
from server.models.agent_config import (
    AgentOptionsInput,
    ModelParamsInput,
)
from server.models.metrics import RunMetricsSummary
from server.services.tool_catalog.tool_references import parse_tool_references

T = TypeVar("T")


def extract_content_parts(value: object) -> object:
    if isinstance(value, str):
        try:
            data = json.loads(value)
            if isinstance(data, dict) and data.get("__type") == "content_parts":
                return data.get("parts", [])
            return data
        except (json.JSONDecodeError, TypeError):
            return value
    return value


class AgentConfigPayload(BaseModel):
    name: str
    description: str = ""
    model_provider: ModelProvider
    model_name: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: AgentOptionsInput = Field(default_factory=AgentOptionsInput)
    model_params: ModelParamsInput = Field(default_factory=ModelParamsInput)

    @field_validator("tools", mode="before")
    @classmethod
    def _validate_tools(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("tools must be a list")
        return parse_tool_references(value)

    @model_validator(mode="after")
    def _validate_model_connection(self) -> "AgentConfigPayload":
        validate_provider_model_params(self.model_provider, self.model_params)
        return self


class AgentConfigResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    is_default: bool = False
    model_provider: str
    model_name: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: AgentOptionsInput = Field(default_factory=AgentOptionsInput)
    model_params: ModelParamsInput = Field(default_factory=ModelParamsInput)
    created_at: str
    updated_at: str


class PageResponse(BaseModel, Generic[T]):
    items: list[T] = Field(default_factory=list)
    limit: int
    offset: int
    has_more: bool
    total: int | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class CreateSessionRequest(BaseModel):
    pass


class ForkSessionRequest(BaseModel):
    context_summary: str


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


class SchedulerChatCancelRequest(BaseModel):
    state_id: str


class PendingEventResponse(BaseModel):
    id: str
    target_agent_id: str
    source_agent_id: str | None = None
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


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


class WakeConditionResponse(BaseModel):
    type: str
    wait_for: list[str] = Field(default_factory=list)
    wait_mode: str = "all"
    completed_ids: list[str] = Field(default_factory=list)
    time_value: float | None = None
    time_unit: str | None = None
    wakeup_at: str | None = None
    timeout_at: str | None = None


class AgentStateBase(BaseModel):
    id: str
    root_state_id: str | None = None
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

    @field_validator("task", mode="before")
    @classmethod
    def _extract_content_parts(cls, value: object) -> object:
        return extract_content_parts(value)


class AgentStateResponse(AgentStateBase):
    session_id: str
    pending_input: UserInput | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    signal_propagated: bool = False

    @field_validator("pending_input", mode="before")
    @classmethod
    def _extract_pending_input(cls, value: object) -> object:
        return extract_content_parts(value)


class AgentStateListItem(AgentStateBase):
    pass


class SchedulerStatsResponse(BaseModel):
    total: int
    pending: int
    running: int
    waiting: int
    idle: int
    queued: int
    completed: int
    failed: int


class SchedulerTreeStatsResponse(BaseModel):
    total: int
    running: int
    waiting: int
    queued: int
    idle: int
    completed: int
    failed: int
    cancelled: int


class SchedulerTreeNodeResponse(BaseModel):
    state_id: str
    root_state_id: str
    parent_state_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    session_id: str | None = None
    agent_id: str
    task_id: str | None = None
    status: str
    depth: int = 0
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    wake_condition: WakeConditionResponse | None = None
    pending_event_count: int = 0
    last_error: str | None = None
    result_summary: str | None = None


class SchedulerTreeResponse(BaseModel):
    root_state_id: str
    root_session_id: str | None = None
    nodes: list[SchedulerTreeNodeResponse] = Field(default_factory=list)
    stats: SchedulerTreeStatsResponse
    generated_at: str


class DashboardOverviewResponse(BaseModel):
    total_sessions: int
    total_traces: int
    total_agents: int
    total_tokens: int
    scheduler: SchedulerStatsResponse = Field(
        default_factory=lambda: SchedulerStatsResponse(
            total=0,
            pending=0,
            running=0,
            waiting=0,
            idle=0,
            queued=0,
            completed=0,
            failed=0,
        )
    )


class SessionSummaryResponse(BaseModel):
    session_id: str
    agent_id: str | None = None
    last_user_input: UserInput | None = None
    last_response: str | None = None
    run_count: int = 0
    step_count: int = 0
    metrics: RunMetricsSummary = Field(default_factory=RunMetricsSummary)
    created_at: str | None = None
    updated_at: str | None = None
    chat_context_scope_id: str | None = None
    created_by: str | None = None
    base_agent_id: str | None = None
    root_state_status: str | None = None
    source_session_id: str | None = None
    fork_context_summary: str | None = None

    @field_validator("last_user_input", mode="before")
    @classmethod
    def _extract_last_user_input(cls, value: object) -> object:
        return extract_content_parts(value)


class SessionRecordResponse(BaseModel):
    id: str
    chat_context_scope_id: str | None = None
    base_agent_id: str
    created_by: str
    created_at: str
    updated_at: str
    source_session_id: str | None = None
    fork_context_summary: str | None = None


class ChatContextResponse(BaseModel):
    scope_id: str
    channel_instance_id: str
    chat_id: str
    chat_type: str
    user_open_id: str
    base_agent_id: str
    current_session_id: str
    created_at: str
    updated_at: str


class SessionDetailResponse(BaseModel):
    summary: SessionSummaryResponse
    session: SessionRecordResponse | None = None
    chat_context: ChatContextResponse | None = None
    scheduler_state: AgentStateResponse | None = None


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


class TraceBase(BaseModel):
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
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_token_cost: float = 0.0
    input_query: str | None = None
    final_output: str | None = None


class TraceResponse(TraceBase):
    end_time: str | None = None
    root_span_id: str | None = None
    max_depth: int = 0
    spans: list[SpanResponse] = Field(default_factory=list)


class TraceListItem(TraceBase):
    model_config = {"extra": "ignore"}


class AgentProviderCapabilityResponse(BaseModel):
    value: str
    label: str
    default_model_name: str | None = None
    requires_base_url: bool = False
    requires_api_key_env_name: bool = False


class AgentCapabilitiesResponse(BaseModel):
    providers: list[AgentProviderCapabilityResponse] = Field(default_factory=list)


__all__ = [
    "AgentCapabilitiesResponse",
    "AgentConfigPayload",
    "AgentConfigResponse",
    "AgentProviderCapabilityResponse",
    "AgentStateBase",
    "AgentStateListItem",
    "AgentStateResponse",
    "ChatContextResponse",
    "CancelRequest",
    "ChatRequest",
    "CreateAgentRequest",
    "DashboardOverviewResponse",
    "CreateSessionRequest",
    "extract_content_parts",
    "ForkSessionRequest",
    "PageResponse",
    "PendingEventResponse",
    "RunMetricsResponse",
    "RunResponse",
    "SchedulerChatCancelRequest",
    "SchedulerStatsResponse",
    "SchedulerTreeNodeResponse",
    "SchedulerTreeResponse",
    "SchedulerTreeStatsResponse",
    "SessionDetailResponse",
    "SessionRecordResponse",
    "SessionSummaryResponse",
    "SpanResponse",
    "StepMetricsResponse",
    "StepResponse",
    "SteerRequest",
    "TraceListItem",
    "TraceResponse",
    "WakeConditionResponse",
]
