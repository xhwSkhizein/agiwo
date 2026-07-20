"""Objective-level metrics projected from committed ObjectiveView facts (P6-02).

These numbers are views of ObjectiveLog (and optionally linked RunLog review
spans). They are not a second source of truth.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from agiwo.agent.models.run import RunStatus
from agiwo.objective.models import ObjectiveStatus, RunRole
from agiwo.objective.projection import ObjectiveView


@dataclass(frozen=True, slots=True)
class BudgetMetric:
    limit: float
    used: float
    remaining: float


@dataclass(frozen=True, slots=True)
class ObjectiveMetrics:
    """Aggregates explainable from a single ObjectiveView replay."""

    objective_id: str
    session_id: str
    status: str
    is_terminal: bool
    delivery_count: int
    root_run_count: int
    work_count: int
    verification_count: int
    handoff_like_root_run_count: int
    completed_root_run_count: int
    paused_root_run_count: int
    user_input_count: int
    artifact_count: int
    timeline_event_count: int
    system_fault_count: int
    first_started_at: datetime | None
    current_active_started_at: datetime | None
    created_at: datetime | None
    updated_at: datetime | None
    total_elapsed_seconds: float | None
    budget: dict[str, BudgetMetric] = field(default_factory=dict)
    truth_source: str = "ObjectiveLog via ObjectiveView"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "first_started_at",
            "current_active_started_at",
            "created_at",
            "updated_at",
        ):
            value = payload[key]
            if isinstance(value, datetime):
                payload[key] = value.isoformat()
        payload["budget"] = {
            name: asdict(metric) for name, metric in self.budget.items()
        }
        return payload


def project_objective_metrics(
    view: ObjectiveView,
    *,
    now: datetime | None = None,
) -> ObjectiveMetrics:
    """Build ObjectiveMetrics from a projected ObjectiveView (replay-safe)."""
    budget = {
        "handoffs": BudgetMetric(
            limit=view.budget.handoffs.limit,
            used=view.budget.handoffs.used,
            remaining=view.budget.handoffs.remaining,
        ),
        "verification_attempts": BudgetMetric(
            limit=view.budget.verification_attempts.limit,
            used=view.budget.verification_attempts.used,
            remaining=view.budget.verification_attempts.remaining,
        ),
        "llm_cost_usd": BudgetMetric(
            limit=view.budget.llm_cost_usd.limit,
            used=view.budget.llm_cost_usd.used,
            remaining=view.budget.llm_cost_usd.remaining,
        ),
        "active_seconds": BudgetMetric(
            limit=view.budget.active_seconds.limit,
            used=view.budget.active_seconds.used,
            remaining=view.budget.active_seconds.remaining,
        ),
    }
    end = now or view.updated_at
    elapsed: float | None = None
    if view.created_at is not None and end is not None:
        elapsed = max(0.0, (end - view.created_at).total_seconds())

    work = sum(1 for r in view.root_runs if r.role is RunRole.WORK)
    verification = sum(1 for r in view.root_runs if r.role is RunRole.VERIFICATION)
    handoff_like = max(0, len(view.root_runs) - 1)

    return ObjectiveMetrics(
        objective_id=view.objective_id,
        session_id=view.session_id,
        status=view.status.value
        if isinstance(view.status, ObjectiveStatus)
        else str(view.status),
        is_terminal=view.is_terminal,
        delivery_count=1 if view.delivery_report else 0,
        root_run_count=len(view.root_runs),
        work_count=work,
        verification_count=verification,
        handoff_like_root_run_count=handoff_like,
        completed_root_run_count=sum(
            1 for r in view.root_runs if r.status is RunStatus.COMPLETED
        ),
        paused_root_run_count=sum(
            1 for r in view.root_runs if r.status is RunStatus.PAUSED
        ),
        user_input_count=len(view.user_inputs),
        artifact_count=len(view.artifacts),
        timeline_event_count=len(view.timeline),
        system_fault_count=len(view.system_faults),
        first_started_at=view.first_started_at,
        current_active_started_at=view.current_active_started_at,
        created_at=view.created_at,
        updated_at=view.updated_at,
        total_elapsed_seconds=elapsed,
        budget=budget,
    )


__all__ = [
    "BudgetMetric",
    "ObjectiveMetrics",
    "project_objective_metrics",
]
