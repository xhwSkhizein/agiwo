"""Cooperative recoverable-pause control (not AbortSignal / cancel)."""

from dataclasses import dataclass
from enum import Enum


class RunControlSignal(str, Enum):
    """Internal objective-agnostic control signals for the run loop."""

    PAUSE = "pause"
    FORCE_FAIL = "force_fail"


class PauseReason:
    """Structured pause_reason values written to RunOutput.metadata.

    Keep these stable: Objective drain routing matches by exact membership,
    not substring sniffing.
    """

    LLM_BUDGET_DENIED = "llm_budget_denied"
    ACTIVE_SECONDS_EXCEEDED = "active_seconds_exceeded"
    LLM_COST_CEILING_EXCEEDED = "llm_cost_ceiling_exceeded"
    RECOVERABLE_PAUSE = "recoverable_pause"
    USER_PAUSE = "user_pause"
    USER_ARCHIVE = "user_archive"


# Pause reasons that should trigger Objective budget drain when a root run pauses.
BUDGET_DRAIN_PAUSE_REASONS = frozenset(
    {
        PauseReason.LLM_BUDGET_DENIED,
        PauseReason.ACTIVE_SECONDS_EXCEEDED,
        PauseReason.LLM_COST_CEILING_EXCEEDED,
    }
)


@dataclass
class PauseRequest:
    reason: str
    signal: RunControlSignal = RunControlSignal.PAUSE


__all__ = [
    "BUDGET_DRAIN_PAUSE_REASONS",
    "PauseReason",
    "PauseRequest",
    "RunControlSignal",
]
