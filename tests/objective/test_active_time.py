"""P3-03: Objective active window with injectable clock."""

from datetime import datetime, timedelta, timezone

import pytest

from agiwo.objective.active_time import (
    check_active_time,
    current_window_used_seconds,
    facts_close_active_window,
    facts_ensure_active_window,
    facts_resume_active_window,
)
from agiwo.objective.errors import BudgetBoundaryHit
from agiwo.objective.log import (
    FactBatch,
    fact_active_window_started,
    fact_objective_created,
)
from agiwo.objective.models import ObjectiveBudget
from agiwo.objective.projection import project_objective


def _dt(seconds: float = 0.0) -> datetime:
    return datetime(2026, 7, 18, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        seconds=seconds
    )


def _created(*, limit: float = 60.0):
    return fact_objective_created(
        session_id="s1",
        budget=ObjectiveBudget.create(
            handoffs=5,
            verification_attempts=5,
            llm_cost_usd=1.0,
            active_seconds=limit,
        ),
    )


def _view(*, limit: float = 60.0, window_start: datetime | None = _dt(0)):
    batch = FactBatch(objective_id="o1", start_sequence=1, now=_dt())
    batch.add(_created(limit=limit))
    if window_start is not None:
        batch.add(fact_active_window_started(started_at=window_start))
    view = project_objective(batch.facts, objective_id="o1")
    assert view is not None
    return view


def test_current_window_used_from_started_at() -> None:
    view = _view(window_start=_dt(0))
    assert current_window_used_seconds(view, _dt(30)) == 30.0
    assert current_window_used_seconds(view, _dt(60)) == 60.0


def test_check_rejects_at_or_over_limit() -> None:
    view = _view(limit=30.0, window_start=_dt(0))
    check_active_time(view, checked_at=_dt(29.9), pending_action="x")
    with pytest.raises(BudgetBoundaryHit) as exc:
        check_active_time(view, checked_at=_dt(30.0), pending_action="x")
    assert exc.value.dimension == "active_seconds"


def test_first_started_at_immutable_and_new_window_resets_used() -> None:
    batch = FactBatch(objective_id="o1", start_sequence=1, now=_dt())
    stream = []
    stream.append(batch.add(_created(limit=60.0)))
    stream.append(batch.add(fact_active_window_started(started_at=_dt(0))))
    open_view = project_objective(stream, objective_id="o1")
    assert open_view is not None
    assert open_view.first_started_at == _dt(0)

    close_batch = FactBatch(objective_id="o1", start_sequence=3, now=_dt(40))
    stream.extend(
        close_batch.add_many(
            facts_close_active_window(
                open_view,
                now=_dt(40),
                waiting_reason="expects_user_reply",
            )
        )
    )
    waiting = project_objective(stream, objective_id="o1")
    assert waiting is not None
    assert waiting.current_active_started_at is None
    assert waiting.first_started_at == _dt(0)
    assert waiting.budget.active_seconds.used == 40.0

    resume_batch = FactBatch(
        objective_id="o1", start_sequence=len(stream) + 1, now=_dt(100)
    )
    stream.extend(
        resume_batch.add_many(facts_resume_active_window(waiting, now=_dt(100)))
    )
    resumed = project_objective(stream, objective_id="o1")
    assert resumed is not None
    assert resumed.first_started_at == _dt(0)
    assert resumed.current_active_started_at == _dt(100)
    assert current_window_used_seconds(resumed, _dt(100)) == 0.0
    assert current_window_used_seconds(resumed, _dt(110)) == 10.0
    assert resumed.budget.active_seconds.used == 40.0


def test_ensure_window_is_noop_when_open() -> None:
    view = _view(window_start=_dt(0))
    assert facts_ensure_active_window(view, now=_dt(5)) == []


def test_agent_projection_uses_current_window() -> None:
    view = _view(limit=60.0, window_start=_dt(0))
    proj = view.budget_for_agent(checked_at=_dt(15))
    assert proj["used_active_seconds"] == 15.0
    assert proj["remaining_active_seconds"] == 45.0
    assert proj["historical_active_seconds"] == 0.0


def test_parallel_runs_share_one_window_clock() -> None:
    view = _view(window_start=_dt(0))
    assert current_window_used_seconds(view, _dt(20)) == 20.0
    assert current_window_used_seconds(view, _dt(20)) == 20.0
