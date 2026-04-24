"""Run-scoped models and enums for agent execution."""

import dataclasses
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agiwo.agent.models.input import UserInput
from agiwo.agent.models.review import ReviewState
from agiwo.config.termination import TerminationReason
from agiwo.utils.serialization import serialize_optional_datetime


def fields_to_dict(obj: object) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_info in dataclasses.fields(obj):
        value = getattr(obj, field_info.name)
        if isinstance(value, datetime):
            result[field_info.name] = serialize_optional_datetime(value)
        elif isinstance(value, Enum):
            result[field_info.name] = value.value
        else:
            result[field_info.name] = value
    return result


class RunStatus(str, Enum):
    """Agent run status."""

    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
    before_token_estimate: int
    after_token_estimate: int
    message_count: int
    transcript_path: str
    analysis: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    compact_model: str = ""
    compact_tokens: int = 0

    def get_summary(self) -> str:
        return self.analysis.get("summary", "")


@dataclass
class TokenStats:
    """Token usage statistics."""

    total: int = 0
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_creation: int = 0
    cost: float = 0.0


@dataclass
class StepStats:
    """Step execution statistics."""

    total: int = 0
    tool_calls: int = 0
    assistant: int = 0
    current: int = 0


@dataclass
class CompactionState:
    """Compaction state and metadata."""

    last_metadata: CompactMetadata | None = None
    failure_count: int = 0


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    agent_id: str
    agent_name: str
    user_id: str | None = None
    depth: int = 0
    parent_run_id: str | None = None
    timeout_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunLedger:
    """Persistent run state - messages and aggregated statistics."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_schemas: list[dict[str, Any]] | None = None
    start_time: float = field(default_factory=time.time)
    termination_reason: TerminationReason | None = None
    response_content: str | None = None
    run_start_seq: int = 0

    # Aggregated statistics
    tokens: TokenStats = field(default_factory=TokenStats)
    steps: StepStats = field(default_factory=StepStats)
    compaction: CompactionState = field(default_factory=CompactionState)
    review: ReviewState = field(default_factory=ReviewState)


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

    @classmethod
    def from_ledger(cls, ledger: "RunLedger", *, elapsed_ms: float) -> "RunMetrics":
        return cls(
            duration_ms=elapsed_ms,
            total_tokens=ledger.tokens.total,
            input_tokens=ledger.tokens.input,
            output_tokens=ledger.tokens.output,
            cache_read_tokens=ledger.tokens.cache_read,
            cache_creation_tokens=ledger.tokens.cache_creation,
            token_cost=ledger.tokens.cost,
            steps_count=ledger.steps.total,
            tool_calls_count=ledger.steps.tool_calls,
        )

    def to_dict(self) -> dict[str, Any]:
        return fields_to_dict(self)


@dataclass
class RunOutput:
    """Execution result from Agent.run()."""

    session_id: str | None = None
    run_id: str | None = None
    response: str | None = None
    metrics: RunMetrics | None = None
    termination_reason: TerminationReason | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunView:
    run_id: str
    session_id: str
    agent_id: str
    status: RunStatus
    user_id: str | None = None
    response: str | None = None
    termination_reason: TerminationReason | None = None
    metrics: RunMetrics | None = None
    last_user_input: UserInput | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    parent_run_id: str | None = None


__all__ = [
    "CompactMetadata",
    "CompactionState",
    "MemoryRecord",
    "RunIdentity",
    "RunLedger",
    "RunMetrics",
    "RunOutput",
    "RunStatus",
    "RunView",
    "StepStats",
    "TerminationReason",
    "TokenStats",
    "fields_to_dict",
]
