"""Cooperative recoverable-pause control (not AbortSignal / cancel)."""

from dataclasses import dataclass
from enum import Enum


class RunControlSignal(str, Enum):
    """Internal objective-agnostic control signals for the run loop."""

    PAUSE = "pause"
    FORCE_FAIL = "force_fail"


@dataclass
class PauseRequest:
    reason: str
    signal: RunControlSignal = RunControlSignal.PAUSE


__all__ = ["PauseRequest", "RunControlSignal"]
