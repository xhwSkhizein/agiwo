"""P6-02: Objective metrics projection from committed views."""

from datetime import datetime, timezone

import pytest

from agiwo.agent.models.run import RunStatus
from agiwo.objective import (
    BudgetLimits,
    CreateObjectiveRequest,
    ObjectiveService,
    project_objective_metrics,
)
from agiwo.objective.models import (
    BudgetDimension,
    ObjectiveBudget,
    ObjectiveStatus,
    RunRole,
    new_id,
)
from agiwo.objective.projection import ObjectiveView, RootRunView
from agiwo.objective.store.memory import InMemoryObjectiveStore

from tests.objective.e2e.harness import user_message


@pytest.mark.asyncio
async def test_metrics_replay_is_idempotent_after_create() -> None:
    store = InMemoryObjectiveStore()
    service = ObjectiveService(store)
    created = await service.create_objective(
        CreateObjectiveRequest(
            session_id="sess-metrics",
            user_message=user_message("measure me"),
            budget=BudgetLimits(
                handoffs=10,
                verification_attempts=5,
                llm_cost_usd=5.0,
                active_seconds=600,
            ),
            idempotency_key=new_id(),
        )
    )
    view = await service.get_view(created.objective_id)
    assert view is not None

    first = project_objective_metrics(view)
    second = project_objective_metrics(view)
    assert first.to_dict() == second.to_dict()
    assert first.objective_id == created.objective_id
    assert first.budget["handoffs"].limit == 10
    assert first.timeline_event_count >= 1
    assert first.truth_source.startswith("ObjectiveLog")

    via_service = await service.get_metrics(created.objective_id)
    assert via_service is not None
    assert via_service.to_dict() == first.to_dict()
    blob = str(first.to_dict())
    assert "prompt" not in blob.lower()
    assert "review_score" not in blob


@pytest.mark.asyncio
async def test_metrics_endpoint_counts_follow_view_after_mainline() -> None:
    """Use a projected view stub to assert delivery/root-run rollups."""
    now = datetime.now(timezone.utc)
    budget = ObjectiveBudget(
        handoffs=BudgetDimension(limit=10, used=2),
        verification_attempts=BudgetDimension(limit=5, used=1),
        llm_cost_usd=BudgetDimension(limit=5.0, used=0.5),
        active_seconds=BudgetDimension(limit=600, used=30),
    )
    view = ObjectiveView(
        objective_id="obj_m",
        session_id="sess_m",
        status=ObjectiveStatus.COMPLETED,
        budget=budget,
        root_runs=(
            RootRunView(
                run_id="r1",
                role=RunRole.WORK,
                status=RunStatus.COMPLETED,
            ),
            RootRunView(
                run_id="r2",
                role=RunRole.WORK,
                status=RunStatus.COMPLETED,
            ),
            RootRunView(
                run_id="r3",
                role=RunRole.VERIFICATION,
                status=RunStatus.COMPLETED,
            ),
        ),
        delivery_report="done",
        delivery_outcome_id="out_1",
        created_at=now,
        updated_at=now,
        timeline=(),
    )
    metrics = project_objective_metrics(view, now=now)
    assert metrics.delivery_count == 1
    assert metrics.work_count == 2
    assert metrics.verification_count == 1
    assert metrics.completed_root_run_count == 3
    assert metrics.is_terminal is True
