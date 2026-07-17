import pytest

from agiwo.agent.models.plan import Milestone, RunPlan
from agiwo.agent.plan import (
    PlanValidationError,
    apply_plan_changes,
    handle_plan_tool_result,
)
from agiwo.tool.base import ToolResult


def test_update_plan_activates_first_pending() -> None:
    plan = RunPlan()
    result = ToolResult.success(
        tool_name="update_plan",
        tool_call_id="tc-update",
        content="ok",
        output={"changes": [{"id": "inspect", "description": "Inspect auth"}]},
    )

    update = handle_plan_tool_result(result, plan, current_seq=4)

    assert update is not None
    assert update.active_milestone_id == "inspect"
    assert update.revision == 1
    assert [(m.id, m.status, m.declared_at_seq) for m in plan.milestones] == [
        ("inspect", "active", 4)
    ]


def test_duplicate_active_in_changes_fail_fast() -> None:
    plan = RunPlan()

    with pytest.raises(PlanValidationError, match="at most one active"):
        apply_plan_changes(
            plan,
            [
                {"id": "a", "description": "A", "status": "active"},
                {"id": "b", "description": "B", "status": "active"},
            ],
            current_seq=1,
            source_tool_call_id="tc",
        )


def test_omitted_items_preserved() -> None:
    plan = RunPlan(
        milestones=[
            Milestone(id="inspect", description="Inspect", status="active"),
            Milestone(id="fix", description="Fix", status="pending"),
        ],
        revision=1,
    )

    update = apply_plan_changes(
        plan,
        [{"id": "inspect", "status": "completed"}],
        current_seq=5,
        source_tool_call_id="tc",
    )

    assert update is not None
    assert [(m.id, m.status) for m in plan.milestones] == [
        ("inspect", "completed"),
        ("fix", "active"),
    ]


def test_reopen_completed_to_pending() -> None:
    plan = RunPlan(
        milestones=[
            Milestone(
                id="inspect",
                description="Inspect",
                status="completed",
                completed_at_seq=3,
            ),
        ],
        revision=2,
    )

    update = apply_plan_changes(
        plan,
        [{"id": "inspect", "status": "pending"}],
        current_seq=6,
        source_tool_call_id="tc",
    )

    assert update is not None
    assert plan.milestones[0].status == "active"
    assert plan.milestones[0].completed_at_seq is None
    assert plan.active_milestone_id == "inspect"


def test_active_switch_demotes_old_active() -> None:
    plan = RunPlan(
        milestones=[
            Milestone(id="inspect", description="Inspect", status="active"),
            Milestone(id="fix", description="Fix", status="pending"),
        ],
        revision=1,
    )

    update = apply_plan_changes(
        plan,
        [
            {"id": "inspect", "status": "completed"},
            {"id": "fix", "status": "active"},
        ],
        current_seq=9,
        source_tool_call_id="tc",
        reason="activated",
    )

    assert update.milestone_switch is True
    assert plan.active_milestone_id == "fix"
    assert plan.milestones[0].status == "completed"
    assert plan.milestones[1].status == "active"


def test_order_preservation_when_appending() -> None:
    plan = RunPlan(
        milestones=[
            Milestone(id="a", description="A", status="active"),
            Milestone(id="b", description="B", status="pending"),
        ],
        revision=1,
    )

    apply_plan_changes(
        plan,
        [{"id": "c", "description": "C"}],
        current_seq=3,
        source_tool_call_id="tc",
    )

    assert [milestone.id for milestone in plan.milestones] == ["a", "b", "c"]
