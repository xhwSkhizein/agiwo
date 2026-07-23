"""Run-loop outcome and compaction cycle models."""

from dataclasses import dataclass
from typing import NamedTuple

from agiwo.agent.retry import RunBlockingFaultError


class CompactionCycleResult(NamedTuple):
    """Result of a compaction cycle."""

    compact_start_seq: int
    skip_assistant_turn: bool


@dataclass(frozen=True, slots=True)
class LoopCompleted:
    """Normal loop termination; proceed to finalize."""


@dataclass(frozen=True, slots=True)
class LoopPaused:
    """Recoverable pause after a committed checkpoint."""

    reason: str
    checkpoint_id: str


@dataclass(frozen=True, slots=True)
class LoopFault:
    """Blocking fault that must be finalized or failed at the run boundary."""

    error: RunBlockingFaultError


LoopExit = LoopCompleted | LoopPaused | LoopFault
