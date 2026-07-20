"""P1-01 invariant-focused tests."""

import pytest

from agiwo.agent.models.run import RunStatus
from agiwo.objective.errors import InvariantViolation
from agiwo.objective.models import (
    ObjectiveStatus,
    assert_single_active_objective,
    assert_single_active_root_run,
    is_objective_resumable,
    is_root_run_resumable,
)


def test_two_active_objectives_fail() -> None:
    with pytest.raises(InvariantViolation):
        assert_single_active_objective(
            session_id="sess",
            active_objective_ids=["a", "b"],
        )


def test_two_active_root_runs_fail() -> None:
    with pytest.raises(InvariantViolation):
        assert_single_active_root_run(
            objective_id="obj",
            active_run_ids=["x", "y"],
        )


def test_terminal_not_resumable_paused_is() -> None:
    for status in (ObjectiveStatus.COMPLETED, ObjectiveStatus.FAILED):
        assert not is_objective_resumable(status)
    for status in (
        ObjectiveStatus.WAITING_USER,
        ObjectiveStatus.BUDGET_PAUSED,
        ObjectiveStatus.USER_PAUSED,
    ):
        assert is_objective_resumable(status)

    for status in (
        RunStatus.COMPLETED,
        RunStatus.INTERRUPTED,
        RunStatus.FAILED,
        RunStatus.RUNNING,
        None,
    ):
        assert not is_root_run_resumable(status)
    assert is_root_run_resumable(RunStatus.PAUSED)
