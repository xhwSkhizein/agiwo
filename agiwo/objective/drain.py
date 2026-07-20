"""DRAINING entry and completion for recoverable Objective pauses."""

from datetime import datetime
from typing import Literal

from agiwo.objective.active_time import facts_close_active_window
from agiwo.objective.log import (
    FactDraft,
    fact_drain_completed,
    fact_drain_started,
    fact_objective_status_changed,
)
from agiwo.objective.models import ObjectiveStatus, utc_now
from agiwo.objective.projection import ObjectiveView

DrainReason = Literal["budget", "user_pause", "user_archive"]

_DRAIN_TARGET: dict[DrainReason, ObjectiveStatus] = {
    "budget": ObjectiveStatus.BUDGET_PAUSED,
    "user_pause": ObjectiveStatus.USER_PAUSED,
    "user_archive": ObjectiveStatus.USER_PAUSED,
}


def drain_facts_when_idle(
    view: ObjectiveView,
    *,
    reason: DrainReason,
    now: datetime | None = None,
    source: str = "objective_service",
    barrier_run_ids: tuple[str, ...] = (),
) -> list[FactDraft]:
    """Enter DRAINING; complete immediately when the barrier is empty.

    Non-empty ``barrier_run_ids`` leave the Objective in DRAINING until the
    caller converges Runs via ``Scheduler.request_recoverable_pause`` and
    commits ``DrainCompleted``.
    """
    if view.status in {
        ObjectiveStatus.DRAINING,
        ObjectiveStatus.BUDGET_PAUSED,
        ObjectiveStatus.USER_PAUSED,
        ObjectiveStatus.COMPLETED,
        ObjectiveStatus.FAILED,
    }:
        return []

    at = now or utc_now()
    target = _DRAIN_TARGET[reason]
    drafts: list[FactDraft] = []
    drafts.extend(facts_close_active_window(view, now=at))
    drafts.append(
        fact_drain_started(
            reason=reason,
            source=source,
            barrier_run_ids=list(barrier_run_ids),
        )
    )

    if barrier_run_ids:
        # Remain in DRAINING until P3-04 pause barrier converges.
        return drafts

    drafts.append(fact_drain_completed(next_status=target))
    return drafts


def resume_status_facts(
    view: ObjectiveView,
    *,
    now: datetime | None = None,
) -> list[FactDraft]:
    """Return RUNNING status change for a paused Objective (Run restore is separate)."""
    if view.status not in {
        ObjectiveStatus.BUDGET_PAUSED,
        ObjectiveStatus.USER_PAUSED,
    }:
        return []
    draft = fact_objective_status_changed(
        from_status=view.status,
        to_status=ObjectiveStatus.RUNNING,
        reason="user_resume",
    )
    if now is None:
        return [draft]
    return [FactDraft(kind=draft.kind, payload=draft.payload, occurred_at=now)]


__all__ = [
    "DrainReason",
    "drain_facts_when_idle",
    "resume_status_facts",
]
