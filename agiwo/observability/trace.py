"""
Trace and Span models for distributed tracing.

Supports both internal storage and OpenTelemetry export.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SpanKind(str, Enum):
    """Span type classification"""

    AGENT = "agent"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"


class SpanStatus(str, Enum):
    """Span execution status"""

    UNSET = "unset"
    RUNNING = "running"
    OK = "ok"
    ERROR = "error"


class Span(BaseModel):
    """
    Execution span - minimal tracing unit.

    Design follows OpenTelemetry Span specification with Agio-specific extensions.
    """

    # === Identity ===
    span_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str
    parent_span_id: str | None = None

    # === Type & Name ===
    kind: SpanKind
    name: str  # e.g., "research_agent", "web_search"

    # === Timing ===
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    duration_ms: float | None = None

    # === Status ===
    status: SpanStatus = SpanStatus.RUNNING
    error_message: str | None = None

    # === Hierarchy ===
    depth: int = 0

    # === Context Attributes ===
    attributes: dict[str, Any] = Field(default_factory=dict)
    # Common attributes:
    # - agent_id: Agent ID
    # - model_id: LLM model ID
    # - tool_name: Tool name
    # - iteration: Loop iteration number
    # - branch_id: Parallel branch ID

    # === Input/Output Preview ===
    input_preview: str | None = None  # First 500 chars
    output_preview: str | None = None

    # === Metrics ===
    metrics: dict[str, Any] = Field(default_factory=dict)
    # Common metrics:
    # - tokens.input: Input token count
    # - tokens.output: Output token count
    # - tokens.total: Total token count
    # - first_token_ms: Time to first token

    # === LLM Call Details (for LLM_CALL spans) ===
    llm_details: dict[str, Any] | None = Field(default=None)
    # When kind == LLM_CALL, contains complete LLM call information:
    # {
    #   "request": {...},           # Request parameters (temperature, max_tokens, etc.)
    #   "messages": [...],          # Complete message list sent to LLM
    #   "tools": [...],             # Tool definitions
    #   "response_content": "...",   # Complete response content
    #   "response_tool_calls": [...], # Tool calls from response
    #   "finish_reason": "...",     # Finish reason
    #   "error": "...",             # Error message (if any)
    # }

    # === Tool Call Details (for TOOL_CALL spans) ===
    tool_details: dict[str, Any] | None = Field(default=None)
    # When kind == TOOL_CALL, contains complete tool call information:
    # {
    #   "tool_name": "...",         # Tool name
    #   "tool_id": "...",            # Tool identifier
    #   "tool_call_id": "...",      # Tool call ID
    #   "input_args": {...},         # Complete input arguments (not truncated)
    #   "output": "...",             # Complete execution result (not truncated)
    #   "error": "...",              # Error message (if any)
    #   "metrics": {...},            # Execution metrics (duration_ms, tool_exec_time_ms)
    #   "status": "...",             # Status: "completed" | "error"
    # }

    # === Associations ===
    run_id: str | None = None  # Associated Run ID
    step_id: str | None = None  # Associated Step ID

    def complete(
        self,
        status: SpanStatus = SpanStatus.OK,
        error_message: str | None = None,
        output_preview: str | None = None,
    ):
        """Mark span as completed"""
        self.end_time = datetime.now(timezone.utc)
        self.status = status
        self.error_message = error_message
        if output_preview is not None:
            self.output_preview = output_preview
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            self.duration_ms = delta.total_seconds() * 1000

    def to_otel_span_kind(self) -> int:
        """
        Convert to OpenTelemetry SpanKind.

        OTEL SpanKind values:
        - INTERNAL = 1
        - SERVER = 2
        - CLIENT = 3
        - PRODUCER = 4
        - CONSUMER = 5
        """
        mapping = {
            SpanKind.AGENT: 1,  # INTERNAL
            SpanKind.LLM_CALL: 3,  # CLIENT (calling external LLM API)
            SpanKind.TOOL_CALL: 3,  # CLIENT
        }
        return mapping.get(self.kind, 1)

    def to_otel_status_code(self) -> int:
        """
        Convert to OpenTelemetry StatusCode.

        OTEL StatusCode values:
        - UNSET = 0
        - OK = 1
        - ERROR = 2
        """
        mapping = {
            SpanStatus.UNSET: 0,
            SpanStatus.RUNNING: 0,  # Treat as UNSET
            SpanStatus.OK: 1,
            SpanStatus.ERROR: 2,
        }
        return mapping.get(self.status, 0)


class Trace(BaseModel):
    """
    Complete execution trace.

    A Trace contains one complete user request processing,
    which may be a single Agent run or nested Agent executions.
    """

    # === Identity ===
    trace_id: str = Field(default_factory=lambda: str(uuid4()))

    # === Context ===
    agent_id: str | None = None  # Agent ID
    session_id: str | None = None
    user_id: str | None = None

    # === Timing ===
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    duration_ms: float | None = None

    # === Status ===
    status: SpanStatus = SpanStatus.RUNNING

    # === Span List ===
    root_span_id: str | None = None
    spans: list[Span] = Field(default_factory=list)

    # === Aggregated Metrics ===
    total_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    max_depth: int = 0

    # === Input/Output ===
    input_query: str | None = None
    final_output: str | None = None

    def add_span(self, span: Span):
        """Add a span to the trace"""
        self.spans.append(span)
        self.max_depth = max(self.max_depth, span.depth)

        # Update aggregated metrics
        if span.kind == SpanKind.LLM_CALL:
            self.total_llm_calls += 1
            if span.metrics:
                self.total_tokens += span.metrics.get("tokens.total", 0) or span.metrics.get("total_tokens", 0)
                self.total_cache_read_tokens += span.metrics.get("tokens.cache_read", 0) or span.metrics.get("cache_read_tokens", 0)
                self.total_cache_creation_tokens += span.metrics.get("tokens.cache_creation", 0) or span.metrics.get("cache_creation_tokens", 0)
        elif span.kind == SpanKind.TOOL_CALL:
            self.total_tool_calls += 1

    def complete(
        self,
        status: SpanStatus = SpanStatus.OK,
        final_output: str | None = None,
    ):
        """Mark trace as completed"""
        self.end_time = datetime.now(timezone.utc)
        self.status = status
        if final_output is not None:
            self.final_output = final_output
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            self.duration_ms = delta.total_seconds() * 1000


__all__ = ["Span", "SpanKind", "SpanStatus", "Trace"]
