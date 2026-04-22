"""Stable read models for replayed runtime decisions."""

from dataclasses import dataclass
from datetime import datetime

from agiwo.agent.models.run import CompactMetadata, TerminationReason


@dataclass(frozen=True, slots=True)
class TerminationDecisionView:
    session_id: str
    run_id: str
    agent_id: str
    sequence: int
    created_at: datetime
    reason: TerminationReason
    phase: str
    source: str


@dataclass(frozen=True, slots=True)
class CompactionDecisionView:
    session_id: str
    run_id: str
    agent_id: str
    sequence: int
    created_at: datetime
    metadata: CompactMetadata
    summary: str | None = None


@dataclass(frozen=True, slots=True)
class CompactionFailureDecisionView:
    session_id: str
    run_id: str
    agent_id: str
    sequence: int
    created_at: datetime
    error: str
    attempt: int
    max_attempts: int
    terminal: bool


@dataclass(frozen=True, slots=True)
class RetrospectDecisionView:
    session_id: str
    run_id: str
    agent_id: str
    sequence: int
    created_at: datetime
    affected_sequences: tuple[int, ...]
    affected_step_ids: tuple[str, ...]
    feedback: str | None = None
    replacement: str | None = None
    trigger: str | None = None


@dataclass(frozen=True, slots=True)
class RollbackDecisionView:
    session_id: str
    run_id: str
    agent_id: str
    sequence: int
    created_at: datetime
    start_sequence: int
    end_sequence: int
    reason: str


@dataclass(frozen=True, slots=True)
class RuntimeDecisionState:
    latest_termination: TerminationDecisionView | None = None
    latest_compaction: CompactionDecisionView | None = None
    latest_compaction_failure: CompactionFailureDecisionView | None = None
    latest_retrospect: RetrospectDecisionView | None = None
    latest_rollback: RollbackDecisionView | None = None

    def is_empty(self) -> bool:
        return (
            self.latest_termination is None
            and self.latest_compaction is None
            and self.latest_compaction_failure is None
            and self.latest_retrospect is None
            and self.latest_rollback is None
        )


__all__ = [
    "CompactionFailureDecisionView",
    "CompactionDecisionView",
    "RetrospectDecisionView",
    "RollbackDecisionView",
    "RuntimeDecisionState",
    "TerminationDecisionView",
]
