import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

from agiwo.utils.tojson import to_json

if TYPE_CHECKING:
    from agiwo.agent.execution_context import ExecutionContext


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


@dataclass
class ContentPart:
    type: ContentType
    text: str | None = None
    url: str | None = None
    mime_type: str | None = None
    detail: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


UserInput = str | list[ContentPart]

MessageContent = str | list[dict[str, Any]]


def normalize_input(user_input: UserInput) -> list[ContentPart]:
    if isinstance(user_input, str):
        return [ContentPart(type=ContentType.TEXT, text=user_input)]
    return user_input


def extract_text(user_input: UserInput) -> str:
    if isinstance(user_input, str):
        return user_input
    texts = [
        part.text for part in user_input if part.type == ContentType.TEXT and part.text
    ]
    return "\n".join(texts)


def to_message_content(parts: list[ContentPart]) -> MessageContent:
    if len(parts) == 1 and parts[0].type == ContentType.TEXT:
        return parts[0].text or ""

    result: list[dict[str, Any]] = []
    for part in parts:
        if part.type == ContentType.TEXT:
            result.append({"type": "text", "text": part.text or ""})
        elif part.type == ContentType.IMAGE:
            block: dict[str, Any] = {
                "type": "image_url",
                "image_url": {"url": part.url or ""},
            }
            if part.detail:
                block["image_url"]["detail"] = part.detail
            result.append(block)
        elif part.type == ContentType.AUDIO:
            result.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "url": part.url or "",
                        "format": part.mime_type or "",
                    },
                }
            )
        elif part.type == ContentType.VIDEO:
            result.append(
                {
                    "type": "video_url",
                    "video_url": {"url": part.url or ""},
                }
            )
        elif part.type == ContentType.FILE:
            result.append(
                {
                    "type": "file",
                    "file": {"url": part.url or "", "mime_type": part.mime_type or ""},
                }
            )
    return result


class MessageRole(str, Enum):
    """Standard LLM message roles"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RunStatus(str, Enum):
    """Agent run status"""

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
    MAX_TOKENS = "max_tokens"
    ERROR = "error"
    ERROR_WITH_CONTEXT = "error_with_context"
    CANCELLED = "cancelled"
    TOOL_LIMIT = "tool_limit"


class EventType(str, Enum):
    """Event types for streaming"""

    STEP_DELTA = "step_delta"
    STEP_COMPLETED = "step_completed"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    ERROR = "error"
    TOOL_AUTH_REQUIRED = "tool_auth_required"
    TOOL_AUTH_DENIED = "tool_auth_denied"


@dataclass
class StepDelta:
    """Incremental update to a StepRecord for streaming."""

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict[str, int] | None = None


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

    model_name: str | None = None
    provider: str | None = None

    first_token_latency_ms: float | None = None


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

    steps_count: int = 0
    tool_calls_count: int = 0
    tool_errors_count: int = 0

    first_token_latency: float | None = None
    response_latency: float | None = None


@dataclass
class LLMCallContext:
    """LLM call context for observability (not persisted)."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    request_params: dict[str, Any] | None = None
    finish_reason: str | None = None


@dataclass
class StepRecord:
    """
    Unified Step record for persistence and streaming.
    LLM request context is carried separately via LLMCallContext.
    """

    session_id: str
    run_id: str
    sequence: int
    role: MessageRole
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    agent_id: str | None = None

    content: MessageContent | None = None
    content_for_user: str | None = None
    reasoning_content: str | None = None

    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    metrics: StepMetrics | None = None
    created_at: datetime = field(default_factory=datetime.now)

    parent_run_id: str | None = None
    depth: int = 0

    def is_assistant_step(self) -> bool:
        return self.role == MessageRole.ASSISTANT

    def is_tool_step(self) -> bool:
        return self.role == MessageRole.TOOL

    @classmethod
    def user(
        cls,
        ctx: "ExecutionContext",
        *,
        sequence: int,
        content: MessageContent,
        **overrides,
    ) -> "StepRecord":
        """Create a user step."""
        context_attrs = _build_step_context_attrs(ctx, overrides)
        return cls(
            sequence=sequence,
            role=MessageRole.USER,
            content=content,
            **context_attrs,
        )

    @classmethod
    def assistant(
        cls,
        ctx: "ExecutionContext",
        *,
        sequence: int,
        content: str | None = None,
        tool_calls: list[dict] | None = None,
        reasoning_content: str | None = None,
        metrics: StepMetrics | None = None,
        **overrides,
    ) -> "StepRecord":
        """Create an assistant step."""
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
        ctx: "ExecutionContext",
        *,
        sequence: int,
        tool_call_id: str,
        name: str,
        content: str,
        content_for_user: str | None = None,
        metrics: StepMetrics | None = None,
        **overrides,
    ) -> "StepRecord":
        """Create a tool step."""
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
    ctx: "ExecutionContext", overrides: dict[str, Any]
) -> dict[str, Any]:
    return {
        "session_id": ctx.session_id,
        "run_id": ctx.run_id,
        "agent_id": overrides.get("agent_id", ctx.agent_id),
        "parent_run_id": overrides.get("parent_run_id", ctx.parent_run_id),
        "depth": overrides.get("depth", ctx.depth),
    }


@dataclass
class StreamEvent:
    """Event for streaming and observability."""

    type: EventType
    run_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    step_id: str | None = None
    delta: StepDelta | None = None
    step: StepRecord | None = None
    llm: LLMCallContext | None = None
    data: dict | None = None

    parent_run_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    depth: int = 0
    agent_id: str | None = None
    trace_id: str | None = None

    def to_sse(self) -> str:
        return f"data: {to_json(self)}\n\n"


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


def step_to_message(step: StepRecord) -> dict[str, Any]:
    role_value = step.role.value
    msg: dict[str, Any] = {"role": role_value}

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
class MemoryRecord:
    content: str
    relevance_score: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
