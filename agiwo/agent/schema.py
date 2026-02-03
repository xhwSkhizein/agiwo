import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from dataclasses import dataclass, field

from agiwo.utils.tojson import to_json


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


class StepEventType(str, Enum):
    """Event types for Step-based streaming"""

    # Step-level events
    STEP_DELTA = "step_delta"  # Incremental update to a step
    STEP_COMPLETED = "step_completed"  # Step is complete (with final snapshot)

    # Run-level events
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"

    # Error events
    ERROR = "error"

    # Tool permission events
    TOOL_AUTH_REQUIRED = "tool_auth_required"  # Requires authorization
    TOOL_AUTH_DENIED = "tool_auth_denied"  # Authorization denied


@dataclass
class StepDelta:
    """
    Incremental update to a Step.
    ( FIXME: direct use StreamChunk ? )

    Used for streaming text content and tool calls as they arrive.
    """

    content: str | None = None  # Text to append
    reasoning_content: str | None = (
        None  # Reasoning content to append (e.g., DeepSeek thinking mode)
    )
    tool_calls: list[dict] | None = None  # Tool calls to append/update
    usage: dict[str, int] | None = None  # Token usage metrics


@dataclass
class StepMetrics:
    """
    Metrics for a single Step.

    Different fields are relevant for different step types:
    - Assistant steps: token counts, model info, latency
    - Tool steps: start_at, end_at, duration
    """

    # Timing (for assistant/tool steps)
    start_at: datetime | None = (
        None  # Actual execution start time (for LLM calls or tool execution)
    )
    end_at: datetime | None = None  # Actual execution end time
    duration_ms: float | None = None

    # Token usage (for assistant/model steps)
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None

    # Model info
    model_name: str | None = None
    provider: str | None = None

    # Latency
    first_token_latency_ms: float | None = None


@dataclass
class RunMetrics:
    """
    Metrics for a single (Agent) Run.

    Supports merge for aggregating metrics from child runs.
    """

    start_at: float = 0.0
    end_at: float = 0.0
    duration_ms: float = 0.0

    # Token usage
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    # Execution stats
    steps_count: int = 0
    tool_calls_count: int = 0
    tool_errors_count: int = 0

    # Latency
    first_token_latency: float | None = None
    response_latency: float | None = None


@dataclass
class Step:
    """
    Unified Step model that directly maps to LLM Message structure.

    This is the core data model that:
    1. Stores conversation history in the database
    2. Streams to the frontend in real-time
    3. Builds LLM context with zero conversion
    """

    # --- Indexing & Association ---
    session_id: str
    run_id: str  # Logical grouping for a single user query → response cycle
    sequence: int  # Global sequence within session (1, 2, 3, ...)
    role: MessageRole  # Core Content (Standard LLM Message)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # --- Agent Binding (for unified Fork/Resume) ---
    agent_id: str | None = None  # Agent ID that created this step

    # --- Core Content (Standard LLM Message) ---
    content: str | None = None  # LLM message content
    content_for_user: str | None = (
        None  # Display content for frontend (preferred for UI)
    )
    reasoning_content: str | None = (
        None  # Reasoning content (e.g., DeepSeek thinking mode)
    )

    # Assistant-specific fields
    tool_calls: list[dict] | None = None

    # Tool-specific fields
    tool_call_id: str | None = None
    name: str | None = None

    # --- Metadata ---
    metrics: StepMetrics | None = None
    created_at: datetime = field(default_factory=datetime.now)

    # --- Multi-Agent Context ---
    parent_run_id: str | None = None  # Parent run ID for nested executions

    # --- Observability (new) ---
    trace_id: str | None = None  # Trace ID for distributed tracing
    parent_span_id: str | None = None  # Parent span ID
    span_id: str | None = None  # Span ID
    depth: int = 0  # Nesting depth

    # --- LLM Call Context (for assistant steps) ---
    llm_messages: list[dict[str, Any]] | None = (
        None  # Complete message list sent to LLM
    )
    llm_tools: list[dict[str, Any]] | None = None  # Tool definitions sent to LLM
    llm_request_params: dict[str, Any] | None = (
        None  # Request parameters (temperature, max_tokens, etc.)
    )

    def is_user_step(self) -> bool:
        """Check if this is a user message"""
        return self.role == MessageRole.USER

    def is_assistant_step(self) -> bool:
        """Check if this is an assistant message"""
        return self.role == MessageRole.ASSISTANT

    def is_tool_step(self) -> bool:
        """Check if this is a tool result message"""
        return self.role == MessageRole.TOOL

    def has_tool_calls(self) -> bool:
        """Check if this assistant step has tool calls"""
        return self.is_assistant_step() and bool(self.tool_calls)


@dataclass
class StepEvent:
    """
    Unified event for Step-based streaming.

    Frontend receives these events via SSE and uses them to:
    1. Build up Steps incrementally (via delta)
    2. Finalize Steps (via snapshot)
    3. Track run status
    """

    type: StepEventType
    run_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # For STEP_DELTA and STEP_COMPLETED
    step_id: str | None = None

    # For STEP_DELTA - incremental updates
    delta: StepDelta | None = None

    # For STEP_COMPLETED - final state
    snapshot: Step | None = None

    # For RUN_* and ERROR events
    data: dict | None = None

    # Nesting context
    parent_run_id: str | None = None  # Parent run ID for nesting

    # Nesting depth (0 = top-level)
    span_id: str | None = None
    parent_span_id: str | None = None
    depth: int = 0

    # Nested execution context
    agent_id: str | None = None  # ID of nested Agent

    # Observability fields (injected by TraceCollector)
    trace_id: str | None = None

    def to_sse(self) -> str:
        """
        Convert to Server-Sent Events format.

        Returns:
            str: SSE-formatted string ready to send to client
        """
        return f"data: {to_json(self)}\n\n"


@dataclass
class Run:
    """
    Unified Run metadata for Agent.

    """

    agent_id: str
    session_id: str
    input_query: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    status: RunStatus = RunStatus.STARTING

    # Aggregated metadata
    response_content: str | None = None  # Final response content (extracted from Steps)
    metrics: RunMetrics = field(default_factory=RunMetrics)

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # --- Multi-Agent Context ---
    parent_run_id: str | None = None  # Parent run ID for nested executions

    # --- Observability ---
    trace_id: str | None = None  # Associated trace ID


@dataclass
class RunOutput:
    """
    Execution result from AgiwoAgent.run().

    Contains both the response and execution metrics.
    """

    session_id: str | None = None
    run_id: str | None = None
    response: str | None = None
    metrics: RunMetrics | None = None

    # Additional context
    termination_reason: str | None = None  # "max_steps", "max_iterations", etc.
    error: str | None = None


class StepAdapter:
    """Adapter for converting Step to/from LLM message format"""

    @staticmethod
    def to_llm_message(step: Step) -> dict[str, Any]:
        """
        Convert Step to LLM message format (OpenAI-compatible).

        Args:
            step: Step instance

        Returns:
            dict: Message in OpenAI format
        """
        # Handle role as enum or string
        role_value = step.role.value if hasattr(step.role, "value") else step.role
        msg: dict[str, Any] = {"role": role_value}

        if step.content is not None:
            msg["content"] = step.content

        # Include reasoning_content if present (will be handled by DeepseekModel preprocessing)
        if step.reasoning_content is not None:
            msg["reasoning_content"] = step.reasoning_content

        if step.tool_calls is not None:
            msg["tool_calls"] = step.tool_calls

        if step.tool_call_id is not None:
            msg["tool_call_id"] = step.tool_call_id

        if step.name is not None:
            msg["name"] = step.name

        return msg

    @staticmethod
    def from_llm_message(
        msg: dict, session_id: str, run_id: str, sequence: int
    ) -> Step:
        """
        Create Step from LLM message format.

        Args:
            msg: Message in OpenAI format
            session_id: Session ID
            run_id: Run ID
            sequence: Sequence number

        Returns:
            Step: Step instance
        """
        return Step(
            session_id=session_id,
            run_id=run_id,
            sequence=sequence,
            role=MessageRole(msg["role"]),
            content=msg.get("content"),
            reasoning_content=msg.get("reasoning_content"),
            tool_calls=msg.get("tool_calls"),
            tool_call_id=msg.get("tool_call_id"),
            name=msg.get("name"),
        )

    @staticmethod
    def steps_to_messages(steps: list[Step]) -> list[dict[str, Any]]:
        """
        Convert list of Steps to list of LLM messages.

        Args:
            steps: List of Step instances

        Returns:
            list: List of messages in OpenAI format
        """
        return [StepAdapter.to_llm_message(step) for step in steps]
