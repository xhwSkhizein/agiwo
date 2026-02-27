"""
API-layer Pydantic models for request/response serialization.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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
    last_user_input: str | None = None
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


class AgentConfigCreate(BaseModel):
    name: str
    description: str = ""
    model_provider: str  # "openai" | "deepseek" | "anthropic"
    model_name: str  # "gpt-4o" | "deepseek-chat" | ...
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    model_params: dict[str, Any] = Field(default_factory=dict)


class AgentConfigUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    system_prompt: str | None = None
    tools: list[str] | None = None
    options: dict[str, Any] | None = None
    model_params: dict[str, Any] | None = None


class AgentConfigResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    model_provider: str
    model_name: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    model_params: dict[str, Any] = Field(default_factory=dict)
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
