from agiwo.agent.introspect.models import (
    ContentUpdate,
    ContextRepairPlan,
    GoalState,
    IntrospectionOutcome,
    IntrospectionState,
    Milestone,
)


def test_goal_state_tracks_active_milestone() -> None:
    state = GoalState(
        milestones=[Milestone(id="inspect", description="Inspect", status="active")],
        active_milestone_id="inspect",
    )

    assert state.active_milestone is not None
    assert state.active_milestone.description == "Inspect"


def test_introspection_state_defaults_to_clean_boundary() -> None:
    state = IntrospectionState()

    assert state.review_count_since_boundary == 0
    assert state.consecutive_errors == 0
    assert state.last_boundary_seq == 0
    assert state.pending_trigger is None
    assert state.latest_aligned_checkpoint is None


def test_context_repair_plan_reports_affected_steps() -> None:
    plan = ContextRepairPlan(
        mode="step_back",
        start_seq=3,
        end_seq=8,
        experience="Search drifted into unrelated JWT code.",
        content_updates=[
            ContentUpdate(
                step_id="step-search",
                tool_call_id="tc-search",
                content="[EXPERIENCE] Search drifted into unrelated JWT code.",
            )
        ],
    )

    assert plan.affected_count == 1


def test_introspection_outcome_advances_boundary() -> None:
    outcome = IntrospectionOutcome(
        aligned=False,
        mode="step_back",
        boundary_seq=12,
        experience="The search drifted.",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
    )

    assert outcome.boundary_seq == 12
    assert outcome.mode == "step_back"
