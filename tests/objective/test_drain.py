"""P3-05: DRAINING → BUDGET_PAUSED / USER_PAUSED (idle and barrier members)."""

import pytest

from agiwo.objective.drain import drain_facts_when_idle
from agiwo.objective.log import FactBatch, fact_objective_created, materialize_facts
from agiwo.objective.models import ObjectiveBudget, ObjectiveStatus
from agiwo.objective.projection import project_objective


def _budget() -> ObjectiveBudget:
    return ObjectiveBudget.create(
        handoffs=1,
        verification_attempts=1,
        llm_cost_usd=1,
        active_seconds=60,
    )


def _created_facts() -> list:
    return materialize_facts(
        "o1",
        [fact_objective_created(session_id="s1", budget=_budget())],
    )


def _drain_facts(view, **kwargs):
    batch = FactBatch(objective_id=view.objective_id, start_sequence=2)
    batch.add_many(drain_facts_when_idle(view, **kwargs))
    return batch.facts


def test_idle_budget_drain_completes_to_budget_paused() -> None:
    facts = _created_facts()
    view = project_objective(facts, objective_id="o1")
    assert view is not None
    drain = _drain_facts(view, reason="budget")
    final = project_objective(facts + drain, objective_id="o1")
    assert final is not None
    assert final.status == ObjectiveStatus.BUDGET_PAUSED


@pytest.mark.parametrize("reason", ["user_pause", "user_archive"])
def test_idle_user_drain_reasons(reason: str) -> None:
    facts = _created_facts()
    view = project_objective(facts, objective_id="o1")
    assert view is not None
    drain = _drain_facts(view, reason=reason)  # type: ignore[arg-type]
    final = project_objective(facts + drain, objective_id="o1")
    assert final is not None
    assert final.status == ObjectiveStatus.USER_PAUSED
    assert any(f.payload.get("reason") == reason for f in drain)


def test_nonempty_barrier_stays_draining() -> None:
    facts = _created_facts()
    view = project_objective(facts, objective_id="o1")
    assert view is not None
    drain = _drain_facts(
        view,
        reason="user_pause",
        barrier_run_ids=("run_1",),
    )
    final = project_objective(facts + drain, objective_id="o1")
    assert final is not None
    assert final.status == ObjectiveStatus.DRAINING
    assert any(f.payload.get("barrier_run_ids") == ["run_1"] for f in drain)
