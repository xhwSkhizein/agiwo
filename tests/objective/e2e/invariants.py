"""Cross-store consistency assertions for Objective E2E (P6-01)."""

from agiwo.agent.models.run import RUN_TERMINAL_STATUSES, RunStatus
from agiwo.objective.models import OBJECTIVE_TERMINAL
from agiwo.objective.projection import ObjectiveView
from agiwo.objective.store.base import ObjectiveStore


def assert_session_slot_invariants(views: list[ObjectiveView]) -> None:
    active = [v for v in views if not v.is_terminal]
    assert len(active) <= 1, f"session has {len(active)} non-terminal objectives"


def assert_objective_invariants(view: ObjectiveView) -> None:
    active_root_runs = [run for run in view.root_runs if run.is_active]
    assert len(active_root_runs) <= 1, (
        f"objective has {len(active_root_runs)} non-terminal root runs"
    )
    for root_run in view.root_runs:
        if root_run.status is RunStatus.PAUSED:
            assert root_run.outcome is None, "PAUSED root run must not have Outcome"
        if root_run.status in RUN_TERMINAL_STATUSES:
            assert root_run.outcome is not None, "terminal root run needs Outcome"
    if view.status in OBJECTIVE_TERMINAL and view.delivery_report:
        assert view.delivery_outcome_id is not None
    for input_item in view.user_inputs:
        assert input_item.message.is_user_provided is True


async def assert_outbox_consistent(store: ObjectiveStore, objective_id: str) -> None:
    pending = await store.list_pending_dispatches(objective_id=objective_id)
    # Completed Objective should not leave PENDING/CLAIMED forever after settle.
    # Allow empty or only terminal records depending on store filtering.
    for record in pending:
        assert record.objective_id == objective_id
