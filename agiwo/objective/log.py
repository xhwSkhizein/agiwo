"""ObjectiveLog fact models: append-only domain events."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from agiwo.objective.errors import ValidationError
from agiwo.utils.serialization import parse_datetime
from agiwo.objective.models import (
    Artifact,
    ContributionAnnotation,
    CurrentGoalAnalysis,
    GoalSourceRef,
    HandoffDecision,
    NewContributionSpec,
    ObjectiveBudget,
    ObjectiveContribution,
    ObjectiveStatus,
    ObjectiveUserInput,
    RunOutcome,
    RunRole,
    new_id,
    utc_now,
)


class ObjectiveFactKind(str, Enum):
    OBJECTIVE_CREATED = "ObjectiveCreated"
    OBJECTIVE_STATUS_CHANGED = "ObjectiveStatusChanged"
    OBJECTIVE_USER_INPUT = "ObjectiveUserInput"
    OBJECTIVE_USER_INPUT_EXTERNALIZED = "ObjectiveUserInputExternalized"
    CONTEXT_CAPACITY_EXCEEDED = "ContextCapacityExceeded"
    CONTRIBUTION_CREATED = "ContributionCreated"
    CONTRIBUTION_ANNOTATED = "ContributionAnnotated"
    CURRENT_GOAL_REVISED = "CurrentGoalRevised"
    ROOT_RUN_REQUESTED = "RootRunRequested"
    ROOT_RUN_STARTED = "RootRunStarted"
    ROOT_RUN_PAUSED = "RootRunPaused"
    ROOT_RUN_RESUMED = "RootRunResumed"
    ARTIFACT_REGISTERED = "ArtifactRegistered"
    DECISION_ACCEPTED = "DecisionAccepted"
    RUN_OUTCOME = "RunOutcome"
    VERIFICATION_REQUIRED_SET = "VerificationRequiredSet"
    ENTRY_COMPLEXITY_ASSESSED = "EntryComplexityAssessed"
    BUDGET_ADJUSTED = "BudgetAdjusted"
    BUDGET_USAGE_RECORDED = "BudgetUsageRecorded"
    ACTIVE_WINDOW_STARTED = "ActiveWindowStarted"
    ACTIVE_WINDOW_ENDED = "ActiveWindowEnded"
    WAITING_INTERVAL_STARTED = "WaitingIntervalStarted"
    WAITING_INTERVAL_ENDED = "WaitingIntervalEnded"
    CHECKPOINT_RECORDED = "CheckpointRecorded"
    DRAIN_STARTED = "DrainStarted"
    DRAIN_COMPLETED = "DrainCompleted"
    OBJECTIVE_DELIVERED = "ObjectiveDelivered"
    SYSTEM_FAULT = "SystemFault"


@dataclass(frozen=True, slots=True)
class ObjectiveLogEntry:
    """Common header for every ObjectiveLog fact."""

    fact_id: str
    objective_id: str
    sequence: int
    kind: ObjectiveFactKind
    occurred_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fact_id:
            raise ValidationError("fact_id is required")
        if not self.objective_id:
            raise ValidationError("objective_id is required")
        if self.sequence < 1:
            raise ValidationError("sequence must be >= 1", sequence=self.sequence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "objective_id": self.objective_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "occurred_at": self.occurred_at.isoformat(),
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectiveLogEntry":
        occurred_at = parse_datetime(data["occurred_at"])
        return cls(
            fact_id=data["fact_id"],
            objective_id=data["objective_id"],
            sequence=int(data["sequence"]),
            kind=ObjectiveFactKind(data["kind"]),
            occurred_at=occurred_at,
            payload=dict(data.get("payload") or {}),
        )


@dataclass(frozen=True, slots=True)
class FactDraft:
    """Unsequenced fact payload; identity is stamped by FactBatch."""

    kind: ObjectiveFactKind
    payload: dict[str, Any]
    occurred_at: datetime | None = None
    fact_id: str | None = None

    def materialize(
        self,
        *,
        objective_id: str,
        sequence: int,
        now: datetime | None = None,
    ) -> ObjectiveLogEntry:
        """Stamp identity fields for tests or intentional sequence gaps."""
        return ObjectiveLogEntry(
            fact_id=self.fact_id or new_id("fact_"),
            objective_id=objective_id,
            sequence=sequence,
            kind=self.kind,
            occurred_at=self.occurred_at or now or utc_now(),
            payload=dict(self.payload),
        )


class FactBatch:
    """Allocate sequence / fact_id / objective_id for a contiguous fact batch."""

    def __init__(
        self,
        *,
        objective_id: str,
        start_sequence: int = 1,
        now: datetime | None = None,
    ) -> None:
        if not objective_id:
            raise ValidationError("objective_id is required")
        if start_sequence < 1:
            raise ValidationError(
                "start_sequence must be >= 1", sequence=start_sequence
            )
        self._objective_id = objective_id
        self._next_seq = start_sequence
        self._now = now or utc_now()
        self._facts: list[ObjectiveLogEntry] = []

    @property
    def objective_id(self) -> str:
        return self._objective_id

    @property
    def now(self) -> datetime:
        return self._now

    @property
    def next_sequence(self) -> int:
        return self._next_seq

    @property
    def facts(self) -> list[ObjectiveLogEntry]:
        return list(self._facts)

    def skip_to(self, sequence: int) -> None:
        """Advance the next sequence (for intentional gaps in tests)."""
        if sequence < self._next_seq:
            raise ValidationError(
                "cannot skip backwards",
                sequence=sequence,
                next_sequence=self._next_seq,
            )
        self._next_seq = sequence

    def add(self, draft: FactDraft) -> ObjectiveLogEntry:
        entry = draft.materialize(
            objective_id=self._objective_id,
            sequence=self._next_seq,
            now=self._now,
        )
        self._next_seq += 1
        self._facts.append(entry)
        return entry

    def add_many(self, drafts: Iterable[FactDraft]) -> list[ObjectiveLogEntry]:
        return [self.add(draft) for draft in drafts]


def materialize_facts(
    objective_id: str,
    drafts: Sequence[FactDraft],
    *,
    start_sequence: int = 1,
    now: datetime | None = None,
) -> list[ObjectiveLogEntry]:
    """Convenience for tests: stamp a list of drafts into contiguous entries."""
    batch = FactBatch(objective_id=objective_id, start_sequence=start_sequence, now=now)
    batch.add_many(drafts)
    return batch.facts


def fact_objective_created(
    *,
    session_id: str,
    budget: ObjectiveBudget,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.OBJECTIVE_CREATED,
        payload={
            "session_id": session_id,
            "status": ObjectiveStatus.CREATED.value,
            "budget": budget.to_dict(),
        },
    )


def fact_objective_status_changed(
    *,
    from_status: ObjectiveStatus,
    to_status: ObjectiveStatus,
    reason: str,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.OBJECTIVE_STATUS_CHANGED,
        payload={
            "from_status": from_status.value,
            "to_status": to_status.value,
            "reason": reason,
        },
    )


def fact_objective_user_input(
    *,
    user_input: ObjectiveUserInput,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.OBJECTIVE_USER_INPUT,
        payload=user_input.to_dict(),
        occurred_at=user_input.created_at,
    )


def fact_user_input_externalized(
    *,
    input_id: str,
    artifact_id: str,
    authorized_at: datetime | None = None,
) -> FactDraft:
    at = authorized_at or utc_now()
    return FactDraft(
        kind=ObjectiveFactKind.OBJECTIVE_USER_INPUT_EXTERNALIZED,
        payload={
            "input_id": input_id,
            "artifact_id": artifact_id,
            "authorized_at": at.isoformat(),
        },
        occurred_at=at,
    )


def fact_context_capacity_exceeded(
    *,
    context_limit_tokens: int,
    estimated_input_tokens: int,
    externalizable_input_ids: list[str],
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.CONTEXT_CAPACITY_EXCEEDED,
        payload={
            "context_limit_tokens": context_limit_tokens,
            "estimated_input_tokens": estimated_input_tokens,
            "externalizable_input_ids": list(externalizable_input_ids),
        },
    )


def fact_contribution_created(
    *,
    contribution: ObjectiveContribution,
    source_run_id: str | None = None,
) -> FactDraft:
    payload = contribution.to_dict()
    if source_run_id is not None:
        payload["source_run_id"] = source_run_id
    return FactDraft(
        kind=ObjectiveFactKind.CONTRIBUTION_CREATED,
        payload=payload,
    )


def fact_contribution_annotated(
    *,
    annotation: ContributionAnnotation,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.CONTRIBUTION_ANNOTATED,
        payload=annotation.to_dict(),
        occurred_at=annotation.time,
    )


def fact_current_goal_revised(
    *,
    analysis: CurrentGoalAnalysis,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.CURRENT_GOAL_REVISED,
        payload=analysis.to_dict(),
    )


def fact_root_run_requested(
    *,
    run_id: str,
    role: RunRole,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.ROOT_RUN_REQUESTED,
        payload={
            "run_id": run_id,
            "role": role.value,
        },
    )


def fact_root_run_started(
    *,
    run_id: str,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.ROOT_RUN_STARTED,
        payload={"run_id": run_id},
    )


def fact_root_run_paused(
    *,
    run_id: str,
    reason: str,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.ROOT_RUN_PAUSED,
        payload={"run_id": run_id, "reason": reason},
    )


def fact_root_run_resumed(
    *,
    run_id: str,
    reason: str,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.ROOT_RUN_RESUMED,
        payload={"run_id": run_id, "reason": reason},
    )


def fact_artifact_registered(
    *,
    artifact: Artifact,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.ARTIFACT_REGISTERED,
        payload=artifact.to_dict(),
    )


def fact_decision_accepted(
    *,
    run_id: str,
    decision: HandoffDecision,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.DECISION_ACCEPTED,
        payload={"run_id": run_id, "decision": decision.to_dict()},
    )


def fact_run_outcome(
    *,
    outcome: RunOutcome,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.RUN_OUTCOME,
        payload=outcome.to_dict(),
    )


def fact_verification_required_set(
    *,
    run_id: str,
    milestone_id: str | None = None,
) -> FactDraft:
    payload: dict[str, Any] = {"run_id": run_id}
    if milestone_id is not None:
        payload["milestone_id"] = milestone_id
    return FactDraft(
        kind=ObjectiveFactKind.VERIFICATION_REQUIRED_SET,
        payload=payload,
    )


def fact_entry_complexity_assessed(
    *,
    score: int | None,
    threshold: int,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.ENTRY_COMPLEXITY_ASSESSED,
        payload={"score": score, "threshold": threshold},
    )


def fact_budget_adjusted(
    *,
    budget: ObjectiveBudget,
    reason: str,
    previous_budget: ObjectiveBudget | None = None,
) -> FactDraft:
    payload: dict[str, Any] = {"budget": budget.to_dict(), "reason": reason}
    if previous_budget is not None:
        payload["previous_budget"] = previous_budget.to_dict()
    return FactDraft(
        kind=ObjectiveFactKind.BUDGET_ADJUSTED,
        payload=payload,
    )


def fact_budget_usage_recorded(
    *,
    dimension: str,
    delta: float,
    used_after: float,
    provenance: dict[str, Any] | None = None,
) -> FactDraft:
    payload: dict[str, Any] = {
        "dimension": dimension,
        "delta": delta,
        "used_after": used_after,
    }
    if provenance:
        payload.update(provenance)
    return FactDraft(
        kind=ObjectiveFactKind.BUDGET_USAGE_RECORDED,
        payload=payload,
    )


def fact_active_window_started(
    *,
    started_at: datetime | None = None,
) -> FactDraft:
    at = started_at or utc_now()
    return FactDraft(
        kind=ObjectiveFactKind.ACTIVE_WINDOW_STARTED,
        payload={"started_at": at.isoformat()},
        occurred_at=at,
    )


def fact_active_window_ended(
    *,
    ended_at: datetime | None = None,
    elapsed_seconds: float = 0.0,
) -> FactDraft:
    at = ended_at or utc_now()
    return FactDraft(
        kind=ObjectiveFactKind.ACTIVE_WINDOW_ENDED,
        payload={"ended_at": at.isoformat(), "elapsed_seconds": elapsed_seconds},
        occurred_at=at,
    )


def fact_waiting_interval_started(
    *,
    reason: str,
    started_at: datetime | None = None,
) -> FactDraft:
    at = started_at or utc_now()
    return FactDraft(
        kind=ObjectiveFactKind.WAITING_INTERVAL_STARTED,
        payload={"reason": reason, "started_at": at.isoformat()},
        occurred_at=at,
    )


def fact_waiting_interval_ended(
    *,
    ended_at: datetime | None = None,
) -> FactDraft:
    at = ended_at or utc_now()
    return FactDraft(
        kind=ObjectiveFactKind.WAITING_INTERVAL_ENDED,
        payload={"ended_at": at.isoformat()},
        occurred_at=at,
    )


def fact_checkpoint_recorded(
    *,
    run_id: str,
    cursor: dict[str, Any],
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.CHECKPOINT_RECORDED,
        payload={
            "run_id": run_id,
            "cursor": dict(cursor),
        },
    )


def fact_drain_started(
    *,
    reason: str,
    source: str = "objective_service",
    barrier_run_ids: list[str] | None = None,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.DRAIN_STARTED,
        payload={
            "reason": reason,
            "source": source,
            "barrier_run_ids": list(barrier_run_ids or ()),
        },
    )


def fact_drain_completed(
    *,
    next_status: ObjectiveStatus,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.DRAIN_COMPLETED,
        payload={"next_status": next_status.value},
    )


def fact_objective_delivered(
    *,
    final_outcome_id: str,
    report: str,
    artifact_ids: list[str] | None = None,
    delivered_at: datetime | None = None,
) -> FactDraft:
    at = delivered_at or utc_now()
    return FactDraft(
        kind=ObjectiveFactKind.OBJECTIVE_DELIVERED,
        payload={
            "final_outcome_id": final_outcome_id,
            "report": report,
            "artifact_ids": list(artifact_ids or []),
            "delivered_at": at.isoformat(),
        },
        occurred_at=at,
    )


def fact_system_fault(
    *,
    fault_code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> FactDraft:
    return FactDraft(
        kind=ObjectiveFactKind.SYSTEM_FAULT,
        payload={
            "fault_code": fault_code,
            "message": message,
            "details": dict(details or {}),
        },
    )


def contribution_from_spec(
    spec: NewContributionSpec, *, contribution_id: str | None = None
) -> ObjectiveContribution:
    return ObjectiveContribution(
        contribution_id=contribution_id or new_id("ctr_"),
        content=spec.content,
        summary=spec.summary,
    )


def expand_outcome_derived_facts(
    *,
    outcome: RunOutcome,
    occurred_at: datetime | None = None,
) -> list[FactDraft]:
    """Expand Outcome-carried contributions / annotations / goal update into drafts."""
    at = occurred_at or utc_now()
    drafts: list[FactDraft] = []
    for index, spec in enumerate(outcome.new_contributions):
        contribution_id = f"ctr_{outcome.outcome_id}_{index}"
        drafts.append(
            FactDraft(
                kind=ObjectiveFactKind.CONTRIBUTION_CREATED,
                payload={
                    **contribution_from_spec(
                        spec, contribution_id=contribution_id
                    ).to_dict(),
                    "source_run_id": outcome.run_id,
                },
                occurred_at=at,
            )
        )
    for spec in outcome.contribution_annotations:
        drafts.append(
            fact_contribution_annotated(
                annotation=ContributionAnnotation(
                    contribution_id=spec.contribution_id,
                    annotation=spec.annotation,
                    from_run_id=outcome.run_id,
                    time=at,
                    deactivate=spec.deactivate,
                ),
            )
        )
    update = outcome.objective_update
    if update is not None:
        sources = update.sources or (
            GoalSourceRef(kind="outcome", id=outcome.outcome_id),
        )
        drafts.append(
            FactDraft(
                kind=ObjectiveFactKind.CURRENT_GOAL_REVISED,
                payload=CurrentGoalAnalysis(
                    revision=update.expected_revision + 1,
                    intent=update.intent,
                    scope=update.scope,
                    success_criteria=update.success_criteria,
                    assumptions=()
                    if update.assumptions is None
                    else update.assumptions,
                    sources=sources,
                ).to_dict(),
                occurred_at=at,
            )
        )
    return drafts


__all__ = [
    "FactBatch",
    "FactDraft",
    "ObjectiveFactKind",
    "ObjectiveLogEntry",
    "contribution_from_spec",
    "expand_outcome_derived_facts",
    "fact_active_window_ended",
    "fact_active_window_started",
    "fact_artifact_registered",
    "fact_budget_adjusted",
    "fact_budget_usage_recorded",
    "fact_checkpoint_recorded",
    "fact_context_capacity_exceeded",
    "fact_contribution_annotated",
    "fact_contribution_created",
    "fact_current_goal_revised",
    "fact_decision_accepted",
    "fact_drain_completed",
    "fact_drain_started",
    "fact_entry_complexity_assessed",
    "fact_objective_created",
    "fact_objective_delivered",
    "fact_objective_status_changed",
    "fact_objective_user_input",
    "fact_root_run_paused",
    "fact_root_run_requested",
    "fact_root_run_resumed",
    "fact_root_run_started",
    "fact_run_outcome",
    "fact_system_fault",
    "fact_user_input_externalized",
    "fact_verification_required_set",
    "fact_waiting_interval_ended",
    "fact_waiting_interval_started",
    "materialize_facts",
]
