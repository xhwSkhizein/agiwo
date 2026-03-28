"""Run-scoped models and enums for agent execution."""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agiwo.agent.models._serialization import fields_to_dict
from agiwo.agent.models.compact import CompactMetadata
from agiwo.agent.models.input import UserInput


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


@dataclass
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
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_schemas: list[dict[str, Any]] | None = None
    start_time: float = field(default_factory=time.time)
    termination_reason: TerminationReason | None = None
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    token_cost: float = 0.0
    steps_count: int = 0
    tool_calls_count: int = 0
    assistant_steps_count: int = 0
    response_content: str | None = None
    last_compact_metadata: CompactMetadata | None = None


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
        return fields_to_dict(self)


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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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


__all__ = [
    "Run",
    "RunIdentity",
    "RunLedger",
    "RunMetrics",
    "RunOutput",
    "RunStatus",
    "TerminationReason",
]
