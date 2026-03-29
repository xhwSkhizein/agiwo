"""Shared termination reason enum used across agent, tool, and scheduler layers."""

from enum import Enum


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


__all__ = ["TerminationReason"]
