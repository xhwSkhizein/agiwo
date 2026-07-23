"""Cooperative recoverable-pause control (not AbortSignal / cancel)."""

from dataclasses import dataclass
from enum import Enum


class RunControlSignal(str, Enum):
    """Internal control signals for the run loop."""

    PAUSE = "pause"
    FORCE_FAIL = "force_fail"


class PauseReason:
    """Structured pause_reason values written to RunOutput.metadata."""

    RECOVERABLE_PAUSE = "recoverable_pause"
    USER_PAUSE = "user_pause"
    USER_ARCHIVE = "user_archive"


@dataclass
class PauseRequest:
    reason: str
    signal: RunControlSignal = RunControlSignal.PAUSE


__all__ = [
    "PauseReason",
    "PauseRequest",
    "RunControlSignal",
]
