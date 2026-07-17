"""Model-call identity, ledger, and phase definitions for run-level attempt accounting."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class ModelCallPhase(str, Enum):
    """Closed set of business purposes for model calls within a single run."""

    ASSISTANT = "assistant"
    COMPACTION = "compaction"
    TERMINATION_SUMMARY = "termination_summary"
    ASSIGNMENT_FINALIZATION = "assignment_finalization"
    FINALIZATION_CORRECTION = "finalization_correction"


WORK_PHASES: frozenset[ModelCallPhase] = frozenset(
    {ModelCallPhase.ASSISTANT, ModelCallPhase.COMPACTION}
)

FINALIZATION_PHASES: frozenset[ModelCallPhase] = frozenset(
    {
        ModelCallPhase.TERMINATION_SUMMARY,
        ModelCallPhase.ASSIGNMENT_FINALIZATION,
        ModelCallPhase.FINALIZATION_CORRECTION,
    }
)


def new_logical_call_id() -> str:
    return str(uuid4())


@dataclass
class ModelCallPhaseStats:
    """Per-phase attempt counters for a run."""

    attempts: int = 0
    completed: int = 0
    failed: int = 0


@dataclass
class ModelCallLedger:
    """Run-scoped ledger of actual provider model-call attempts."""

    configured_limit: int = 50
    total_attempts: int = 0
    limit_trigger_ordinal: int | None = None
    phase_stats: dict[str, ModelCallPhaseStats] = field(default_factory=dict)
    finalization_consumed: dict[str, bool] = field(default_factory=dict)

    def phase_stats_for(self, phase: ModelCallPhase) -> ModelCallPhaseStats:
        key = phase.value
        if key not in self.phase_stats:
            self.phase_stats[key] = ModelCallPhaseStats()
        return self.phase_stats[key]

    def next_ordinal(self) -> int:
        return self.total_attempts + 1

    def record_attempt_started(self, phase: ModelCallPhase) -> int:
        ordinal = self.next_ordinal()
        stats = self.phase_stats_for(phase)
        stats.attempts += 1
        return ordinal

    def record_attempt_completed(self, phase: ModelCallPhase) -> None:
        self.total_attempts += 1
        stats = self.phase_stats_for(phase)
        stats.completed += 1
        self._maybe_record_limit_trigger()

    def record_attempt_failed(self, phase: ModelCallPhase) -> None:
        self.total_attempts += 1
        stats = self.phase_stats_for(phase)
        stats.failed += 1
        self._maybe_record_limit_trigger()

    def mark_finalization_consumed(self, phase: ModelCallPhase) -> None:
        self.finalization_consumed[phase.value] = True

    def is_finalization_consumed(self, phase: ModelCallPhase) -> bool:
        return self.finalization_consumed.get(phase.value, False)

    def at_or_over_limit(self) -> bool:
        return self.total_attempts >= self.configured_limit

    def _maybe_record_limit_trigger(self) -> None:
        if (
            self.limit_trigger_ordinal is None
            and self.total_attempts >= self.configured_limit
        ):
            self.limit_trigger_ordinal = self.total_attempts

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "max_steps_per_run": self.configured_limit,
            "model_call_attempts_total": self.total_attempts,
            "model_call_limit_trigger_ordinal": self.limit_trigger_ordinal,
            "model_call_phase_stats": {
                phase: {
                    "attempts": stats.attempts,
                    "completed": stats.completed,
                    "failed": stats.failed,
                }
                for phase, stats in self.phase_stats.items()
            },
        }


__all__ = [
    "FINALIZATION_PHASES",
    "ModelCallLedger",
    "ModelCallPhase",
    "ModelCallPhaseStats",
    "WORK_PHASES",
    "new_logical_call_id",
]
