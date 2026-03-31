"""
API-layer Pydantic models for request/response serialization.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agiwo.agent import (
    AgentOptions,
    AgentStorageOptions,
    RunStepStorageConfig,
    TraceStorageConfig,
    UserInput,
)
from agiwo.config.settings import ModelProvider, settings
from agiwo.llm.config_policy import (
    sanitize_model_params_data,
    validate_provider_model_params,
)
from agiwo.skill.config import normalize_skill_dirs
from agiwo.tool.builtin.registry import BUILTIN_TOOLS, ensure_builtin_tools_loaded

ensure_builtin_tools_loaded()


# ── Tool References ───────────────────────────────────────────────────

AGENT_TOOL_PREFIX = "agent:"


class InvalidToolReferenceError(ValueError):
    def __init__(self, value: object) -> None:
        super().__init__(f"Invalid tool reference: {value!r}")
        self.value = value


def parse_tool_reference(value: object) -> str:
    if not isinstance(value, str):
        raise InvalidToolReferenceError(value)
    normalized = value.strip()
    if not normalized:
        raise InvalidToolReferenceError(value)
    if normalized.startswith(AGENT_TOOL_PREFIX):
        agent_id = normalized[len(AGENT_TOOL_PREFIX) :].strip()
        if not agent_id:
            raise InvalidToolReferenceError(value)
        return f"{AGENT_TOOL_PREFIX}{agent_id}"
    if normalized not in BUILTIN_TOOLS:
        raise InvalidToolReferenceError(normalized)
    return normalized


def parse_tool_references(values: list[object]) -> list[str]:
    return [parse_tool_reference(value) for value in values]


# ── Metrics ────────────────────────────────────────────────────────────


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


# ── Agent Options & Model Params ────────────────────────────────────────


def sanitize_agent_options_data(
    data: object,
    *,
    preserve_non_dict: bool = False,
) -> object:
    if not isinstance(data, dict):
        return data if preserve_non_dict else {}
    sanitized = dict(data)
    skills_dirs = normalize_skill_dirs(sanitized.get("skills_dirs"))
    if skills_dirs is None:
        sanitized.pop("skills_dirs", None)
    else:
        sanitized["skills_dirs"] = skills_dirs
    return sanitized


class AgentOptionsInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    config_root: str = ""
    max_steps: int = Field(default=50, ge=1)
    run_timeout: int = Field(default=600, ge=1)
    max_input_tokens_per_call: int | None = Field(default=None, ge=1)
    max_run_cost: float | None = Field(default=None, ge=0)
    enable_termination_summary: bool = True
    termination_summary_prompt: str = ""
    enable_skill: bool = Field(default_factory=lambda: settings.is_skills_enabled)
    skills_dirs: list[str] | None = None
    relevant_memory_max_token: int = Field(default=2048, ge=1)
    stream_cleanup_timeout: float = Field(default=300.0, gt=0)
    compact_prompt: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        return sanitize_agent_options_data(data, preserve_non_dict=True)

    def to_agent_options(
        self,
        *,
        run_step_storage: RunStepStorageConfig,
        trace_storage: TraceStorageConfig,
    ) -> AgentOptions:
        data = self.model_dump(exclude_none=True)
        return AgentOptions(
            **data,
            storage=AgentStorageOptions(
                run_step_storage=run_step_storage,
                trace_storage=trace_storage,
            ),
        )


class ModelParamsInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_url: str | None = None
    api_key_env_name: str | None = None
    max_output_tokens: int = Field(default=4096, ge=1)
    max_context_window: int = Field(default=200000, ge=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    cache_hit_price: float = Field(default=0.0, ge=0)
    input_price: float = Field(default=0.0, ge=0)
    output_price: float = Field(default=0.0, ge=0)
    aws_region: str | None = None
    aws_profile: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_plain_api_key(cls, data: object) -> object:
        return sanitize_model_params_data(
            data,
            preserve_non_dict=True,
            drop_null_keys=False,
        )


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
    model_config = ConfigDict(extra="ignore")


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
    timeout_at: str | None = None


class AgentStateBase(BaseModel):
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


class AgentStateResponse(AgentStateBase):
    session_id: str
    pending_input: UserInput | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    signal_propagated: bool = False


AgentStateListItem = AgentStateBase


class SchedulerStatsResponse(BaseModel):
    total: int
    pending: int
    running: int
    waiting: int
    idle: int
    queued: int
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


class SchedulerChatCancelRequest(BaseModel):
    state_id: str


class PendingEventResponse(BaseModel):
    id: str
    target_agent_id: str
    source_agent_id: str | None = None
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


# ── Chat ────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class CreateSessionRequest(BaseModel):
    chat_context_scope_id: str
    channel_instance_id: str = "console-web"
    user_open_id: str = "console-user"


class SwitchSessionRequest(BaseModel):
    chat_context_scope_id: str
    target_session_id: str


class ForkSessionRequest(BaseModel):
    context_summary: str
