"""Objective active-window accounting (per-window, shared across parallel Runs)."""

from collections.abc import Callable
from datetime import datetime

from agiwo.objective.errors import BudgetBoundaryHit
from agiwo.objective.log import (
    FactDraft,
    fact_active_window_ended,
    fact_active_window_started,
    fact_budget_usage_recorded,
    fact_waiting_interval_ended,
    fact_waiting_interval_started,
)
from agiwo.objective.models import utc_now
from agiwo.objective.projection import ObjectiveView

Clock = Callable[[], datetime]


def current_window_used_seconds(
    view: ObjectiveView,
    checked_at: datetime,
) -> float:
    started = view.current_active_started_at
    if started is None:
        return 0.0
    return max(0.0, (checked_at - started).total_seconds())


def check_active_time(
    view: ObjectiveView,
    *,
    checked_at: datetime,
    pending_action: str,
) -> None:
    """Raise BudgetBoundaryHit when the current window reaches its limit."""
    used = current_window_used_seconds(view, checked_at)
    limit = view.budget.active_seconds.limit
    if used + 1e-12 >= limit:
        raise BudgetBoundaryHit(
            dimension="active_seconds",
            checked_at=checked_at,
            used=used,
            limit=limit,
            pending_action=pending_action,
            required=0.0,
            first_started_at=(
                view.first_started_at.isoformat() if view.first_started_at else None
            ),
            current_active_started_at=(
                view.current_active_started_at.isoformat()
                if view.current_active_started_at
                else None
            ),
        )


def facts_ensure_active_window(
    view: ObjectiveView,
    *,
    now: datetime,
) -> list[FactDraft]:
    """Open a window if none is open (first start or after waiting/resume)."""
    if view.current_active_started_at is not None:
        return []
    return [fact_active_window_started(started_at=now)]


def facts_close_active_window(
    view: ObjectiveView,
    *,
    now: datetime,
    waiting_reason: str | None = None,
) -> list[FactDraft]:
    """Close the open window and optionally start a waiting interval."""
    drafts: list[FactDraft] = []
    if view.current_active_started_at is not None:
        elapsed = current_window_used_seconds(view, now)
        drafts.append(
            fact_active_window_ended(
                ended_at=now,
                elapsed_seconds=elapsed,
            )
        )
        if elapsed > 0:
            used_after = view.budget.active_seconds.used + elapsed
            drafts.append(
                fact_budget_usage_recorded(
                    dimension="active_seconds",
                    delta=elapsed,
                    used_after=used_after,
                    provenance={"source": "active_window_ended"},
                )
            )
    if waiting_reason is not None:
        drafts.append(
            fact_waiting_interval_started(
                reason=waiting_reason,
                started_at=now,
            )
        )
    return drafts


def facts_resume_active_window(
    view: ObjectiveView,
    *,
    now: datetime,
) -> list[FactDraft]:
    """End waiting and open a fresh active window; first_started_at stays."""
    drafts: list[FactDraft] = []
    if view.current_active_started_at is not None:
        drafts.extend(facts_close_active_window(view, now=now))
    drafts.append(fact_waiting_interval_ended(ended_at=now))
    drafts.append(fact_active_window_started(started_at=now))
    return drafts


def default_clock() -> datetime:
    return utc_now()


__all__ = [
    "Clock",
    "check_active_time",
    "current_window_used_seconds",
    "default_clock",
    "facts_close_active_window",
    "facts_ensure_active_window",
    "facts_resume_active_window",
]
