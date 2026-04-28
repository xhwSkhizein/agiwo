from agiwo.agent.introspect.models import Milestone
from agiwo.agent.introspect.replay import build_introspect_state_from_entries
from agiwo.agent.models.log import (
    GoalMilestonesUpdated,
    IntrospectionCheckpointRecorded,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    ToolStepCommitted,
)
from agiwo.agent.models.step import MessageRole


def test_introspect_replay_restores_goal_checkpoint_and_count() -> None:
    state = build_introspect_state_from_entries(
        [
            GoalMilestonesUpdated(
                sequence=1,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                milestones=[
                    Milestone(id="inspect", description="Inspect", status="active")
                ],
                active_milestone_id="inspect",
                reason="declared",
            ),
            ToolStepCommitted(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                step_id="step-search",
                role=MessageRole.TOOL,
                tool_call_id="tc-search",
                name="web_search",
                content="results",
            ),
            IntrospectionCheckpointRecorded(
                sequence=3,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                checkpoint_seq=3,
                milestone_id="inspect",
                review_tool_call_id="tc-review",
                review_step_id="step-review",
            ),
            ToolStepCommitted(
                sequence=4,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                step_id="step-read",
                role=MessageRole.TOOL,
                tool_call_id="tc-read",
                name="web_reader",
                content="page",
            ),
            ToolStepCommitted(
                sequence=5,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                step_id="step-review-tool",
                role=MessageRole.TOOL,
                tool_call_id="tc-review-2",
                name="review_trajectory",
                content="Trajectory review",
            ),
        ]
    )

    assert [m.id for m in state.goal.milestones] == ["inspect"]
    assert state.goal.active_milestone_id == "inspect"
    assert state.introspection.latest_aligned_checkpoint is not None
    assert state.introspection.latest_aligned_checkpoint.seq == 3
    assert state.introspection.last_boundary_seq == 3
    assert state.introspection.review_count_since_boundary == 1


def test_introspect_replay_tracks_pending_trigger_until_outcome() -> None:
    state = build_introspect_state_from_entries(
        [
            IntrospectionTriggered(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                trigger_reason="step_interval",
                active_milestone_id="inspect",
                review_count_since_boundary=8,
                trigger_tool_call_id="tc-search",
                trigger_tool_step_id="step-search",
                notice_step_id="step-search",
            )
        ]
    )

    assert state.introspection.pending_trigger is not None
    assert state.introspection.pending_trigger.trigger_reason == "step_interval"
    assert state.introspection.notice_requested is True

    state = build_introspect_state_from_entries(
        [
            IntrospectionTriggered(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                trigger_reason="step_interval",
                active_milestone_id="inspect",
                review_count_since_boundary=8,
                trigger_tool_call_id="tc-search",
                trigger_tool_step_id="step-search",
                notice_step_id="step-search",
            ),
            IntrospectionOutcomeRecorded(
                sequence=3,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                aligned=True,
                mode="metadata_only",
                boundary_seq=3,
            ),
        ]
    )

    assert state.introspection.pending_trigger is None
    assert state.introspection.notice_requested is False
    assert state.introspection.last_boundary_seq == 3
    assert state.introspection.review_count_since_boundary == 0


def test_introspect_replay_derives_pending_milestone_switch_until_triggered() -> None:
    state = build_introspect_state_from_entries(
        [
            GoalMilestonesUpdated(
                sequence=1,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                milestones=[
                    Milestone(id="inspect", description="Inspect", status="active"),
                    Milestone(id="fix", description="Fix", status="pending"),
                ],
                active_milestone_id="inspect",
                reason="declared",
            ),
            GoalMilestonesUpdated(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                milestones=[
                    Milestone(id="inspect", description="Inspect", status="completed"),
                    Milestone(id="fix", description="Fix", status="active"),
                ],
                active_milestone_id="fix",
                reason="activated",
            ),
        ]
    )

    assert state.introspection.pending_milestone_switch is True
    assert state.introspection.consecutive_errors == 0

    state = build_introspect_state_from_entries(
        [
            GoalMilestonesUpdated(
                sequence=1,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                milestones=[
                    Milestone(id="inspect", description="Inspect", status="active"),
                    Milestone(id="fix", description="Fix", status="pending"),
                ],
                active_milestone_id="inspect",
                reason="declared",
            ),
            GoalMilestonesUpdated(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                milestones=[
                    Milestone(id="inspect", description="Inspect", status="completed"),
                    Milestone(id="fix", description="Fix", status="active"),
                ],
                active_milestone_id="fix",
                reason="activated",
            ),
            IntrospectionTriggered(
                sequence=3,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                trigger_reason="milestone_switch",
                active_milestone_id="fix",
                review_count_since_boundary=0,
            ),
        ]
    )

    assert state.introspection.pending_milestone_switch is False
    assert state.introspection.pending_trigger is not None
