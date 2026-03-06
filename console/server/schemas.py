"""
API-layer Pydantic models for request/response serialization.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from agiwo.agent.options import normalize_skills_dirs
from agiwo.config.settings import settings


# ── Session / Run / Step Responses ──────────────────────────────────────


class StepResponse(BaseModel):
    id: str
    session_id: str
    run_id: str
    sequence: int
    role: str
    agent_id: str | None = None
    content: Any | None = None
    content_for_user: str | None = None
    reasoning_content: str | None = None
    user_input: Any | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metrics: dict | None = None
    created_at: str | None = None
    parent_run_id: str | None = None
    depth: int = 0


class RunResponse(BaseModel):
    id: str
    agent_id: str
    session_id: str
    user_id: str | None = None
    user_input: Any
    status: str
    response_content: str | None = None
    metrics: dict | None = None
    created_at: str | None = None
    updated_at: str | None = None
    parent_run_id: str | None = None


class SessionSummary(BaseModel):
    session_id: str
    agent_id: str | None = None
    last_user_input: Any | None = None  # 结构化 UserInput
    last_response: str | None = None
    run_count: int = 0
    step_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


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
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
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
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    input_query: str | None = None
    final_output: str | None = None


# ── Agent Config ────────────────────────────────────────────────────────


class AgentOptionsPayload(BaseModel):
    config_root: str = ""
    max_steps: int = Field(default=10, ge=1)
    run_timeout: int = Field(default=600, ge=1)
    max_context_window_tokens: int = Field(default=32768, ge=1)
    max_tokens_per_run: int = Field(default=131072, ge=1)
    max_run_token_cost: float | None = Field(default=None, ge=0)
    enable_termination_summary: bool = True
    termination_summary_prompt: str = ""
    enable_skill: bool = Field(default_factory=lambda: settings.is_skills_enabled)
    skills_dirs: list[str] | None = None
    relevant_memory_max_token: int = Field(default=2048, ge=1)
    stream_cleanup_timeout: float = Field(default=300.0, gt=0)
    compact_prompt: str = ""

    model_config = {"extra": "ignore"}

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "skills_dirs" not in normalized and "skills_dir" in normalized:
            normalized["skills_dirs"] = normalized["skills_dir"]
        return normalized

    @field_validator("skills_dirs", mode="before")
    @classmethod
    def _normalize_skills_dirs(cls, value: Any) -> list[str] | None:
        return normalize_skills_dirs(value)


class ModelParamsPayload(BaseModel):
    max_output_tokens_per_call: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    cache_hit_price: float = Field(default=0.0, ge=0)
    input_price: float = Field(default=0.0, ge=0)
    output_price: float = Field(default=0.0, ge=0)

    model_config = {"extra": "ignore"}


class AgentConfigCreate(BaseModel):
    name: str
    description: str = ""
    model_provider: str  # "openai" | "deepseek" | "anthropic"
    model_name: str  # "gpt-4o" | "deepseek-chat" | ...
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: AgentOptionsPayload = Field(default_factory=AgentOptionsPayload)
    model_params: ModelParamsPayload = Field(default_factory=ModelParamsPayload)


class AgentConfigUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    options: AgentOptionsPayload | None = None
    model_params: ModelParamsPayload | None = None


class AgentConfigResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    model_provider: str
    model_name: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: AgentOptionsPayload = Field(default_factory=AgentOptionsPayload)
    model_params: ModelParamsPayload = Field(default_factory=ModelParamsPayload)
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
    submitted_task: str | None = None
    timeout_at: str | None = None


class AgentStateResponse(BaseModel):
    id: str
    session_id: str
    status: str
    task: str
    parent_id: str | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    wake_condition: WakeConditionResponse | None = None
    result_summary: str | None = None
    signal_propagated: bool = False
    is_persistent: bool = False
    depth: int = 0
    wake_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class AgentStateListItem(BaseModel):
    id: str
    status: str
    task: str
    parent_id: str | None = None
    wake_condition: WakeConditionResponse | None = None
    result_summary: str | None = None
    is_persistent: bool = False
    depth: int = 0
    wake_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class SchedulerStatsResponse(BaseModel):
    total: int
    pending: int
    running: int
    sleeping: int
    completed: int
    failed: int


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
