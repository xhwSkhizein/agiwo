"""Foundational runtime types: protocols and enums."""

from enum import Enum
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentContext(Protocol):
    session_id: str
    run_id: str
    agent_id: str
    agent_name: str
    parent_run_id: str | None
    depth: int
    user_id: str | None
    trace_id: str | None
    timeout_at: float | None
    metadata: dict[str, Any]


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


__all__ = [
    "AgentContext",
    "MessageRole",
    "RunStatus",
    "TerminationReason",
]
