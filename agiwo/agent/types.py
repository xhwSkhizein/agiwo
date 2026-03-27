"""Runtime-domain models: enums, protocols, steps, runs, and streaming events."""

import dataclasses
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from agiwo.agent.input import ChannelContext, MessageContent, UserInput, UserMessage
from agiwo.agent.input_codec import (
    extract_text,
    normalize_to_message,
    to_message_content,
)
from agiwo.utils.serialization import serialize_optional_datetime
from agiwo.utils.tojson import to_json

if TYPE_CHECKING:
    from agiwo.agent.run_state import RunContext


def _fields_to_dict(obj: object) -> dict[str, Any]:
    """Serialize a dataclass to dict, handling datetime and enum fields."""
    result: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        if isinstance(val, datetime):
            result[f.name] = serialize_optional_datetime(val)
        elif isinstance(val, Enum):
            result[f.name] = val.value
        else:
            result[f.name] = val
    return result


class MessageRole(str, Enum):
    """Standard LLM message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RunStatus(str, Enum):
    """Agent run status."""

    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TerminationReason(str, Enum):
    """Reason why the agent execution terminated."""

    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    TIMEOUT = "timeout"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    MAX_INPUT_TOKENS_PER_CALL = "max_input_tokens_per_call"
    MAX_RUN_COST = "max_run_cost"
    ERROR = "error"
    ERROR_WITH_CONTEXT = "error_with_context"
    CANCELLED = "cancelled"
    TOOL_LIMIT = "tool_limit"
    SLEEPING = "sleeping"


# ---------------------------------------------------------------------------
# Step domain (from runtime/step.py)
# ---------------------------------------------------------------------------


@dataclass
class StepDelta:
    """Incremental update to a StepRecord for streaming."""

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _fields_to_dict(self)


@dataclass
class StepMetrics:
    """Metrics for a single StepRecord."""

    start_at: datetime | None = None
    end_at: datetime | None = None
    duration_ms: float | None = None
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    token_cost: float | None = None
    usage_source: str | None = None
    model_name: str | None = None
    provider: str | None = None
    first_token_latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _fields_to_dict(self)


def _serialize_user_input_structured(
    value: UserInput | None,
) -> str | dict[str, Any] | list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, UserMessage):
        return {
            "content": [part.to_dict() for part in value.content],
            "context": value.context.to_dict() if value.context else None,
        }
    if isinstance(value, list):
        return [part.to_dict() for part in value]
    return value


@dataclass
class StepRecord:
    """Unified Step record for persistence and streaming."""

    session_id: str
    run_id: str
    sequence: int
    role: MessageRole
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str | None = None
    content: MessageContent | None = None
    content_for_user: str | None = None
    reasoning_content: str | None = None
    user_input: UserInput | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metrics: StepMetrics | None = None
    created_at: datetime = field(default_factory=datetime.now)
    parent_run_id: str | None = None
    depth: int = 0

    def is_user_step(self) -> bool:
        return self.role == MessageRole.USER

    def is_assistant_step(self) -> bool:
        return self.role == MessageRole.ASSISTANT

    def is_tool_step(self) -> bool:
        return self.role == MessageRole.TOOL

    def get_llm_content(self) -> MessageContent | None:
        if self.content is not None:
            return self.content
        if self.user_input is not None:
            return to_message_content(normalize_to_message(self.user_input).content)
        return None

    def get_display_text(self) -> str:
        if self.user_input is not None:
            return extract_text(self.user_input)
        if self.content is not None:
            if isinstance(self.content, str):
                return self.content
            return json.dumps(self.content, ensure_ascii=False)
        return ""

    def get_channel_context(self) -> ChannelContext | None:
        if isinstance(self.user_input, UserMessage):
            return self.user_input.context
        return None

    def to_dict(self) -> dict[str, Any]:
        d = _fields_to_dict(self)
        d["user_input"] = _serialize_user_input_structured(self.user_input)
        d["metrics"] = self.metrics.to_dict() if self.metrics else None
        return d

    @classmethod
    def user(
        cls,
        ctx: "RunContext",
        *,
        sequence: int,
        user_input: UserInput | None = None,
        content: MessageContent | None = None,
        **overrides: Any,
    ) -> "StepRecord":
        context_attrs = _build_step_context_attrs(ctx, overrides)
        if user_input is not None:
            derived_content = to_message_content(
                normalize_to_message(user_input).content
            )
            return cls(
                sequence=sequence,
                role=MessageRole.USER,
                user_input=user_input,
                content=derived_content,
                **context_attrs,
            )
        if content is not None:
            return cls(
                sequence=sequence,
                role=MessageRole.USER,
                content=content,
                **context_attrs,
            )
        raise ValueError("StepRecord.user() requires either user_input or content")

    @classmethod
    def assistant(
        cls,
        ctx: "RunContext",
        *,
        sequence: int,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        reasoning_content: str | None = None,
        metrics: StepMetrics | None = None,
        **overrides: Any,
    ) -> "StepRecord":
        context_attrs = _build_step_context_attrs(ctx, overrides)
        return cls(
            sequence=sequence,
            role=MessageRole.ASSISTANT,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            metrics=metrics,
            **context_attrs,
        )

    @classmethod
    def tool(
        cls,
        ctx: "RunContext",
        *,
        sequence: int,
        tool_call_id: str,
        name: str,
        content: str,
        content_for_user: str | None = None,
        metrics: StepMetrics | None = None,
        **overrides: Any,
    ) -> "StepRecord":
        context_attrs = _build_step_context_attrs(ctx, overrides)
        return cls(
            sequence=sequence,
            role=MessageRole.TOOL,
            content=content,
            content_for_user=content_for_user,
            tool_call_id=tool_call_id,
            name=name,
            metrics=metrics,
            **context_attrs,
        )


def _build_step_context_attrs(
    ctx: "RunContext",
    overrides: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": ctx.session_id,
        "run_id": ctx.run_id,
        "agent_id": overrides.get("agent_id", ctx.agent_id),
        "parent_run_id": overrides.get("parent_run_id", ctx.parent_run_id),
        "depth": overrides.get("depth", ctx.depth),
    }


def step_to_message(step: StepRecord) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": step.role.value}
    if step.content is not None:
        msg["content"] = step.content
    if step.reasoning_content is not None:
        msg["reasoning_content"] = step.reasoning_content
    if step.tool_calls is not None:
        msg["tool_calls"] = step.tool_calls
    if step.tool_call_id is not None:
        msg["tool_call_id"] = step.tool_call_id
    if step.name is not None:
        msg["name"] = step.name
    return msg


def steps_to_messages(steps: list[StepRecord]) -> list[dict[str, Any]]:
    return [step_to_message(step) for step in steps]


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
        return _fields_to_dict(self)


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


@dataclass(kw_only=True)
class AgentStreamItemBase:
    """Base payload shared by all public agent stream items."""

    session_id: str
    run_id: str
    agent_id: str
    parent_run_id: str | None
    depth: int
    timestamp: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_context(cls, ctx: "RunContext", **kwargs: Any) -> "AgentStreamItemBase":
        return cls(
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            agent_id=ctx.agent_id,
            parent_run_id=ctx.parent_run_id,
            depth=ctx.depth,
            **kwargs,
        )

    def _base_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,  # type: ignore[attr-defined]
            "session_id": self.session_id,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "parent_run_id": self.parent_run_id,
            "depth": self.depth,
            "timestamp": serialize_optional_datetime(self.timestamp),
        }

    def to_dict(self) -> dict[str, Any]:
        return self._base_dict()

    def to_sse(self) -> str:
        return f"data: {to_json(self)}\n\n"


@dataclass(kw_only=True)
class RunStartedEvent(AgentStreamItemBase):
    type: Literal["run_started"] = "run_started"


@dataclass(kw_only=True)
class StepDeltaEvent(AgentStreamItemBase):
    step_id: str
    delta: StepDelta
    type: Literal["step_delta"] = "step_delta"

    def to_dict(self) -> dict[str, Any]:
        d = self._base_dict()
        d["step_id"] = self.step_id
        d["delta"] = self.delta.to_dict()
        return d


@dataclass(kw_only=True)
class StepCompletedEvent(AgentStreamItemBase):
    step: StepRecord
    type: Literal["step_completed"] = "step_completed"

    def to_dict(self) -> dict[str, Any]:
        d = self._base_dict()
        d["step"] = self.step.to_dict()
        return d


@dataclass(kw_only=True)
class RunCompletedEvent(AgentStreamItemBase):
    response: str | None = None
    metrics: RunMetrics | None = None
    termination_reason: TerminationReason | None = None
    type: Literal["run_completed"] = "run_completed"

    def to_dict(self) -> dict[str, Any]:
        d = self._base_dict()
        d["response"] = self.response
        d["metrics"] = self.metrics.to_dict() if self.metrics else None
        d["termination_reason"] = (
            self.termination_reason.value if self.termination_reason else None
        )
        return d


@dataclass(kw_only=True)
class RunFailedEvent(AgentStreamItemBase):
    error: str
    type: Literal["run_failed"] = "run_failed"

    def to_dict(self) -> dict[str, Any]:
        d = self._base_dict()
        d["error"] = self.error
        return d


AgentStreamItem: TypeAlias = (
    RunStartedEvent
    | StepDeltaEvent
    | StepCompletedEvent
    | RunCompletedEvent
    | RunFailedEvent
)


__all__ = [
    "AgentStreamItem",
    "AgentStreamItemBase",
    "LLMCallContext",
    "MessageRole",
    "Run",
    "RunCompletedEvent",
    "RunFailedEvent",
    "RunMetrics",
    "RunOutput",
    "RunStartedEvent",
    "RunStatus",
    "StepCompletedEvent",
    "StepDelta",
    "StepDeltaEvent",
    "StepMetrics",
    "StepRecord",
    "TerminationReason",
    "step_to_message",
    "steps_to_messages",
]
