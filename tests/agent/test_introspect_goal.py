import pytest

from agiwo.agent.introspect.goal import (
    GoalValidationError,
    handle_goal_tool_result,
    update_goal_milestones,
)
from agiwo.agent.introspect.models import GoalState, Milestone
from agiwo.tool.base import ToolResult


def test_declare_milestones_activates_first_pending() -> None:
    state = GoalState()
    result = ToolResult.success(
        tool_name="declare_milestones",
        tool_call_id="tc-declare",
        content="ok",
        output={"milestones": [{"id": "inspect", "description": "Inspect auth"}]},
    )

    update = handle_goal_tool_result(result, state, current_seq=4)

    assert update is not None
    assert update.active_milestone_id == "inspect"
    assert [(m.id, m.status, m.declared_at_seq) for m in state.milestones] == [
        ("inspect", "active", 4)
    ]


def test_duplicate_milestone_ids_fail_fast() -> None:
    state = GoalState()

    with pytest.raises(GoalValidationError, match="duplicate milestone id"):
        update_goal_milestones(
            state,
            [
                Milestone(id="inspect", description="Inspect"),
                Milestone(id="inspect", description="Inspect again"),
            ],
            current_seq=1,
            source_tool_call_id="tc",
        )


def test_multiple_active_milestones_fail_fast() -> None:
    state = GoalState()

    with pytest.raises(GoalValidationError, match="at most one active"):
        update_goal_milestones(
            state,
            [
                Milestone(id="a", description="A", status="active"),
                Milestone(id="b", description="B", status="active"),
            ],
            current_seq=1,
            source_tool_call_id="tc",
        )


def test_active_switch_marks_milestone_switch() -> None:
    state = GoalState(
        milestones=[
            Milestone(id="inspect", description="Inspect", status="active"),
            Milestone(id="fix", description="Fix", status="pending"),
        ],
        active_milestone_id="inspect",
    )

    update = update_goal_milestones(
        state,
        [
            Milestone(id="inspect", description="Inspect", status="completed"),
            Milestone(id="fix", description="Fix", status="active"),
        ],
        current_seq=9,
        source_tool_call_id="tc",
        reason="activated",
    )

    assert update.milestone_switch is True
    assert state.active_milestone_id == "fix"
