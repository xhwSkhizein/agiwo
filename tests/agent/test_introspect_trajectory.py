from agiwo.agent.introspect.models import IntrospectionState
from agiwo.agent.introspect.trajectory import (
    maybe_build_introspection_notice,
    parse_introspection_outcome,
    strip_system_review_notices,
)
from agiwo.agent.models.plan import Milestone, RunPlan
from agiwo.tool.base import ToolResult


def test_step_interval_builds_notice() -> None:
    plan = RunPlan(
        milestones=[
            Milestone(id="inspect", description="Inspect auth", status="active")
        ],
    )
    state = IntrospectionState()
    first = ToolResult.success(tool_name="search", tool_call_id="tc1", content="one")
    second = ToolResult.success(tool_name="read", tool_call_id="tc2", content="two")

    assert (
        maybe_build_introspection_notice(
            first, plan, state, step_interval=2, review_on_error=True
        )
        is None
    )
    notice = maybe_build_introspection_notice(
        second, plan, state, step_interval=2, review_on_error=True
    )

    assert notice is not None
    assert notice.trigger_reason == "step_interval"
    assert notice.step_count == 2
    assert "<system-review>" in notice.content


def test_review_trajectory_does_not_increment_counter() -> None:
    plan = RunPlan()
    state = IntrospectionState(review_count_since_boundary=5)
    result = ToolResult.success(
        tool_name="review_trajectory",
        tool_call_id="tc-review",
        content="review",
        output={"aligned": True, "experience": "ok"},
    )

    notice = maybe_build_introspection_notice(
        result, plan, state, step_interval=6, review_on_error=True
    )

    assert notice is None
    assert state.review_count_since_boundary == 5


def test_parse_misaligned_outcome_advances_boundary() -> None:
    plan = RunPlan(
        milestones=[Milestone(id="inspect", description="Inspect", status="active")],
    )
    result = ToolResult.success(
        tool_name="review_trajectory",
        tool_call_id="tc-review",
        content="Trajectory review: aligned=False. JWT drifted",
        output={"aligned": False, "experience": "JWT drifted"},
    )

    outcome = parse_introspection_outcome(
        result,
        plan,
        current_seq=12,
        assistant_step_id="step-call",
        tool_step_id="step-review",
    )

    assert outcome is not None
    assert outcome.aligned is False
    assert outcome.boundary_seq == 12
    assert outcome.experience == "JWT drifted"


def test_strip_system_review_notices() -> None:
    content = "result\n\n<system-review>\ncheck\n</system-review>"

    assert strip_system_review_notices(content) == "result"
