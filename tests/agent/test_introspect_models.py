from agiwo.agent.introspect.models import IntrospectionOutcome, IntrospectionState
from agiwo.agent.models.plan import Milestone, RunPlan


def test_run_plan_tracks_active_milestone() -> None:
    plan = RunPlan(
        milestones=[Milestone(id="inspect", description="Inspect", status="active")],
    )

    assert plan.active_milestone is not None
    assert plan.active_milestone.description == "Inspect"
    assert plan.active_milestone_id == "inspect"


def test_introspection_state_defaults_to_clean_boundary() -> None:
    state = IntrospectionState()

    assert state.review_count_since_boundary == 0
    assert state.consecutive_errors == 0
    assert state.last_boundary_seq == 0
    assert state.pending_trigger is None
    assert state.latest_aligned_checkpoint is None
    assert state.latest_tool_usefulness == []


def test_introspection_outcome_advances_boundary() -> None:
    outcome = IntrospectionOutcome(
        aligned=False,
        boundary_seq=12,
        experience="The search drifted.",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
    )

    assert outcome.boundary_seq == 12
    assert outcome.experience == "The search drifted."
