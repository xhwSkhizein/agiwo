import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from agiwo.utils.tojson import to_json
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

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type.value}
        if self.text is not None:
            result["text"] = self.text
        if self.url is not None:
            result["url"] = self.url
        if self.mime_type is not None:
            result["mime_type"] = self.mime_type
        if self.detail is not None:
            result["detail"] = self.detail
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentPart":
        return cls(
            type=ContentType(data["type"]),
            text=data.get("text"),
            url=data.get("url"),
            mime_type=data.get("mime_type"),
            detail=data.get("detail"),
            metadata=data.get("metadata") or {},
        )


@dataclass
class ChannelContext:
    """Metadata about the channel/environment from which the user input originates."""

    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelContext":
        return cls(source=data["source"], metadata=data.get("metadata") or {})


@dataclass
class UserMessage:
    """Structured user input: multimodal content + optional channel context."""

    content: list[ContentPart]
    context: ChannelContext | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "__type": "user_message",
            "content": [p.to_dict() for p in self.content],
            "context": self.context.to_dict() if self.context else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserMessage":
        return cls(
            content=[ContentPart.from_dict(p) for p in data.get("content") or []],
            context=ChannelContext.from_dict(data["context"])
            if data.get("context")
            else None,
        )


UserInput = str | list[ContentPart] | UserMessage

MessageContent = str | list[dict[str, Any]]


def normalize_to_message(user_input: UserInput) -> UserMessage:
    """Normalize any UserInput form into a UserMessage."""
    if isinstance(user_input, str):
        return UserMessage(
            content=[ContentPart(type=ContentType.TEXT, text=user_input)]
        )
    if isinstance(user_input, list):
        return UserMessage(content=user_input)
    return user_input


def extract_text(user_input: UserInput) -> str:
    if isinstance(user_input, str):
        return user_input
    parts = user_input.content if isinstance(user_input, UserMessage) else user_input
    texts = [part.text for part in parts if part.type == ContentType.TEXT and part.text]
    return "\n".join(texts)


def serialize_user_input(user_input: UserInput) -> str:
    """Serialize UserInput to a string suitable for storage."""
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        return json.dumps(
            {"__type": "content_parts", "parts": [p.to_dict() for p in user_input]}
        )
    return json.dumps(user_input.to_dict())


def deserialize_user_input(s: str) -> UserInput:
    """Deserialize UserInput from a storage string."""
    if not s or not s.startswith("{"):
        return s
    try:
        data = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s
    t = data.get("__type")
    if t == "content_parts":
        return [ContentPart.from_dict(p) for p in data.get("parts") or []]
    if t == "user_message":
        return UserMessage.from_dict(data)
    return s


def _is_local_path(url: str) -> bool:
    return not (
        url.startswith("http://")
        or url.startswith("https://")
        or url.startswith("data:")
    )


def _format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    return f"{size_bytes / (1024 * 1024):.1f}MB"


def _render_local_resource(part: ContentPart) -> str:
    name = (part.metadata or {}).get("name", "")
    size = (part.metadata or {}).get("size", 0)
    mime = part.mime_type or ""
    meta_parts = [m for m in [mime, _format_file_size(size) if size else ""] if m]
    meta_str = f" ({', '.join(meta_parts)})" if meta_parts else ""
    type_labels = {
        ContentType.IMAGE: "图片",
        ContentType.AUDIO: "语音",
        ContentType.VIDEO: "视频",
        ContentType.FILE: "文件",
    }
    label = type_labels.get(part.type, "附件")
    lines = [f"[{label}: {name}{meta_str}]"]
    if part.url:
        lines.append(f"本地路径: {part.url}")
    if part.type in (ContentType.AUDIO, ContentType.VIDEO, ContentType.FILE):
        lines.append("如需处理此文件内容，请使用文件读取工具。")
    return "\n".join(lines)


def to_message_content(parts: list[ContentPart]) -> MessageContent:
    if len(parts) == 1 and parts[0].type == ContentType.TEXT:
        return parts[0].text or ""

    result: list[dict[str, Any]] = []
    for part in parts:
        if part.type == ContentType.TEXT:
            result.append({"type": "text", "text": part.text or ""})
        elif part.type == ContentType.IMAGE:
            url = part.url or ""
            if url and _is_local_path(url):
                result.append({"type": "text", "text": _render_local_resource(part)})
            else:
                block: dict[str, Any] = {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
                if part.detail:
                    block["image_url"]["detail"] = part.detail
                result.append(block)
        elif part.type == ContentType.AUDIO:
            url = part.url or ""
            if url and _is_local_path(url):
                result.append({"type": "text", "text": _render_local_resource(part)})
            else:
                result.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "url": url,
                            "format": part.mime_type or "",
                        },
                    }
                )
        elif part.type == ContentType.VIDEO:
            url = part.url or ""
            if url and _is_local_path(url):
                result.append({"type": "text", "text": _render_local_resource(part)})
            else:
                result.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": url},
                    }
                )
        elif part.type == ContentType.FILE:
            url = part.url or ""
            if url and _is_local_path(url):
                result.append({"type": "text", "text": _render_local_resource(part)})
            else:
                result.append(
                    {
                        "type": "file",
                        "file": {"url": url, "mime_type": part.mime_type or ""},
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
    MAX_OUTPUT_TOKENS_PER_CALL = "max_output_tokens_per_call"
    MAX_CONTEXT_WINDOW_TOKENS = "max_context_window_tokens"
    MAX_TOKENS_PER_RUN = "max_tokens_per_run"
    MAX_RUN_TOKEN_COST = "max_run_token_cost"
    ERROR = "error"
    ERROR_WITH_CONTEXT = "error_with_context"
    CANCELLED = "cancelled"
    TOOL_LIMIT = "tool_limit"
    SLEEPING = "sleeping"


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
    token_cost: float = 0.0

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

    For user steps:
    - `user_input` stores the original UserInput (source of truth)
    - `content` is derived from user_input for LLM message format (cached in memory)
    - Storage only persists user_input; content is computed on deserialization
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
        """Get content in LLM message format. For user steps, derives from user_input."""
        if self.content is not None:
            return self.content
        if self.user_input is not None:
            return to_message_content(normalize_to_message(self.user_input).content)
        return None

    def get_display_text(self) -> str:
        """Get plain text representation for display."""
        if self.user_input is not None:
            return extract_text(self.user_input)
        if self.content is not None:
            if isinstance(self.content, str):
                return self.content
            return json.dumps(self.content, ensure_ascii=False)
        return ""

    def get_channel_context(self) -> "ChannelContext | None":
        """Get channel context from user_input if available."""
        if isinstance(self.user_input, UserMessage):
            return self.user_input.context
        return None

    @classmethod
    def user(
        cls,
        ctx: "ExecutionContext",
        *,
        sequence: int,
        user_input: UserInput | None = None,
        content: MessageContent | None = None,
        **overrides,
    ) -> "StepRecord":
        """Create a user step.

        Args:
            user_input: Full UserInput (preferred for external user messages).
                        When provided, content is derived automatically.
            content: Direct MessageContent (for internal system-generated steps
                     like summary_request, compact_request).

        At least one of user_input or content must be provided.
        """
        context_attrs = _build_step_context_attrs(ctx, overrides)

        if user_input is not None:
            derived_content = to_message_content(normalize_to_message(user_input).content)
            return cls(
                sequence=sequence,
                role=MessageRole.USER,
                user_input=user_input,
                content=derived_content,
                **context_attrs,
            )
        elif content is not None:
            return cls(
                sequence=sequence,
                role=MessageRole.USER,
                content=content,
                **context_attrs,
            )
        else:
            raise ValueError("StepRecord.user() requires either user_input or content")

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

@dataclass
class CompactMetadata:
    """Metadata for a single compact operation."""

    session_id: str
    agent_id: str
    start_seq: int
    end_seq: int

    # Static info (no LLM analysis needed)
    before_token_estimate: int
    after_token_estimate: int
    message_count: int
    transcript_path: str

    # LLM analysis result (JSON)
    analysis: dict[str, Any]

    created_at: datetime = field(default_factory=datetime.now)
    compact_model: str = ""
    compact_tokens: int = 0

    def get_summary(self) -> str:
        """Extract summary from analysis result."""
        return self.analysis.get("summary", "")


@dataclass
class CompactResult:
    """Result of a compact operation."""

    compacted_messages: list[dict[str, Any]]
    metadata: CompactMetadata