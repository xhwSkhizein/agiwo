"""
API-layer Pydantic models for request/response serialization.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agiwo.agent import (
    AgentOptions,
    AgentStorageOptions,
    AgentStreamItem,
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
from server.domain.run_metrics import RunMetricsSummary
from server.domain.tool_references import parse_tool_references


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


class AgentOptionsInput(AgentOptions):
    max_steps: int = Field(default=50, ge=1)
    run_timeout: int = Field(default=600, ge=1)
    max_input_tokens_per_call: int | None = Field(default=None, ge=1)
    max_run_cost: float | None = Field(default=None, ge=0)
    enable_skill: bool = Field(default_factory=lambda: settings.is_skills_enabled)
    skills_dirs: list[str] | None = None
    relevant_memory_max_token: int = Field(default=2048, ge=1)
    stream_cleanup_timeout: float = Field(default=300.0, gt=0)

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
        data.pop("storage", None)
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

    @classmethod
    def from_sdk(cls, metrics: "StepMetrics") -> "StepMetricsResponse":
        """Create a StepMetricsResponse from StepMetrics."""
        return cls(
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            total_tokens=metrics.total_tokens,
            usage_source=metrics.usage_source,
            model_name=metrics.model_name,
            provider=metrics.provider,
            first_token_latency_ms=metrics.first_token_latency_ms,
        )


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

    @classmethod
    def from_sdk(cls, step: "StepRecord") -> "StepResponse":
        """Create a StepResponse from a StepRecord."""
        return cls(
            id=step.id,
            session_id=step.session_id,
            run_id=step.run_id,
            sequence=step.sequence,
            role=step.role.value,
            agent_id=step.agent_id,
            content=step.content,
            content_for_user=step.content_for_user,
            reasoning_content=step.reasoning_content,
            user_input=step.user_input,
            tool_calls=[tc.model_dump() for tc in step.tool_calls] if step.tool_calls else None,
            tool_call_id=step.tool_call_id,
            name=step.name,
            metrics=StepMetricsResponse.from_sdk(step.metrics) if step.metrics else None,
            created_at=step.created_at.isoformat() if step.created_at else None,
            parent_run_id=step.parent_run_id,
            depth=step.depth,
        )


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


class AgentConfigReplace(AgentConfigPayload):
    """Full-replacement payload for persisted agent config updates.

    Unlike AgentConfigPayload (used for creation where options/model_params
    have default_factory defaults), this schema makes them required to ensure
    every field is explicitly provided on a full replace.
    """

    options: AgentOptionsInput
    model_params: ModelParamsInput


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

    @field_validator("task", mode="before")
    @classmethod
    def _extract_content_parts(cls, v: object) -> object:
        """Extract parts from serialized UserInput format."""
        if isinstance(v, str):
            import json
            try:
                data = json.loads(v)
                if isinstance(data, dict) and data.get("__type") == "content_parts":
                    return data.get("parts", [])
                return data
            except (json.JSONDecodeError, TypeError):
                return v
        return v


class AgentStateResponse(AgentStateBase):
    session_id: str
    pending_input: UserInput | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    signal_propagated: bool = False

    @field_validator("pending_input", mode="before")
    @classmethod
    def _extract_pending_input(cls, v: object) -> object:
        """Extract parts from serialized UserInput format."""
        if isinstance(v, str):
            import json
            try:
                data = json.loads(v)
                if isinstance(data, dict) and data.get("__type") == "content_parts":
                    return data.get("parts", [])
                return data
            except (json.JSONDecodeError, TypeError):
                return v
        return v

    @classmethod
    def from_sdk(cls, state: "AgentState") -> "AgentStateResponse":
        """Create an AgentStateResponse from an AgentState."""
        from server.domain.run_metrics import RunMetricsSummary

        # Convert wake_condition if present
        wake_condition = None
        if hasattr(state, "wake_condition") and state.wake_condition is not None:
            wc = state.wake_condition
            # Convert datetime fields to isoformat strings
            wakeup_at = getattr(wc, "wakeup_at", None)
            timeout_at = getattr(wc, "timeout_at", None)
            if wakeup_at is not None and hasattr(wakeup_at, "isoformat"):
                wakeup_at = wakeup_at.isoformat()
            if timeout_at is not None and hasattr(timeout_at, "isoformat"):
                timeout_at = timeout_at.isoformat()
            wake_condition = WakeConditionResponse(
                type=wc.type.value if hasattr(wc.type, "value") else str(wc.type),
                wait_for=list(getattr(wc, "wait_for", []) or []),
                wait_mode=wc.wait_mode.value if hasattr(wc.wait_mode, "value") else str(getattr(wc, "wait_mode", "all")),
                completed_ids=list(getattr(wc, "completed_ids", []) or []),
                time_value=getattr(wc, "time_value", None),
                time_unit=getattr(wc, "time_unit", None),
                wakeup_at=wakeup_at,
                timeout_at=timeout_at,
            )

        base_dict = {
            "id": state.id,
            "status": state.status.value if hasattr(state.status, "value") else str(state.status),
            "task": state.task,
            "parent_id": state.parent_id,
            "result_summary": state.result_summary,
            "agent_config_id": state.agent_config_id,
            "is_persistent": state.is_persistent,
            "depth": state.depth,
            "wake_count": state.wake_count,
            "metrics": getattr(state, "metrics", None) or RunMetricsSummary(),
            "created_at": state.created_at.isoformat() if state.created_at else None,
            "updated_at": state.updated_at.isoformat() if state.updated_at else None,
            "wake_condition": wake_condition,
        }
        return cls(
            **base_dict,
            session_id=state.session_id,
            pending_input=state.pending_input,
            config_overrides=state.config_overrides,
            signal_propagated=state.signal_propagated,
        )


class AgentStateListItem(AgentStateBase):
    @classmethod
    def from_sdk(cls, state: "AgentState") -> "AgentStateListItem":
        """Create an AgentStateListItem from an AgentState."""
        from server.domain.run_metrics import RunMetricsSummary

        return cls(
            id=state.id,
            status=state.status.value if hasattr(state.status, "value") else str(state.status),
            task=state.task,
            parent_id=state.parent_id,
            result_summary=state.result_summary,
            agent_config_id=state.agent_config_id,
            is_persistent=state.is_persistent,
            depth=state.depth,
            wake_count=state.wake_count,
            metrics=getattr(state, "metrics", None) or RunMetricsSummary(),
            created_at=state.created_at.isoformat() if state.created_at else None,
            updated_at=state.updated_at.isoformat() if state.updated_at else None,
        )


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


# ── Stream Event Serialization ────────────────────────────────────────


def stream_event_to_payload(event: AgentStreamItem) -> dict[str, Any]:
    """Convert an AgentStreamItem to a SSE payload dictionary."""
    from agiwo.agent import StepCompletedEvent, StepMetrics

    if isinstance(event, StepCompletedEvent):
        return {
            "type": "step",
            "step": StepResponse.from_sdk(event.step).model_dump(),
        }
    return {"type": "unknown", "data": str(event)}
