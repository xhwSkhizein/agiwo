from agiwo.agent.models.log import (
    ReviewCheckpointRecorded,
    ReviewMilestonesUpdated,
    ReviewOutcomeRecorded,
    ReviewTriggerDecided,
    ToolStepCommitted,
)
from agiwo.agent.models.review import Milestone
from agiwo.agent.models.step import MessageRole
from agiwo.agent.review.replay import build_review_state_from_entries


def test_review_replay_restores_milestones_checkpoint_and_count() -> None:
    state = build_review_state_from_entries(
        [
            ReviewMilestonesUpdated(
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
            ReviewCheckpointRecorded(
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

    assert [m.id for m in state.milestones] == ["inspect"]
    assert state.latest_checkpoint is not None
    assert state.latest_checkpoint.seq == 3
    assert state.review_count_since_checkpoint == 1


def test_review_replay_tracks_pending_notice_until_outcome() -> None:
    state = build_review_state_from_entries(
        [
            ReviewTriggerDecided(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                trigger_reason="step_interval",
                active_milestone_id="inspect",
                review_count_since_checkpoint=8,
                trigger_tool_call_id="tc-search",
                trigger_tool_step_id="step-search",
                notice_step_id="step-search",
            )
        ]
    )

    assert state.pending_review_notice is not None
    assert state.pending_review_notice.trigger_reason == "step_interval"

    state = build_review_state_from_entries(
        [
            ReviewTriggerDecided(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                trigger_reason="step_interval",
                active_milestone_id="inspect",
                review_count_since_checkpoint=8,
                trigger_tool_call_id="tc-search",
                trigger_tool_step_id="step-search",
                notice_step_id="step-search",
            ),
            ReviewOutcomeRecorded(
                sequence=3,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                aligned=True,
                mode="metadata_only",
            ),
        ]
    )

    assert state.pending_review_notice is None
    assert state.review_count_since_checkpoint == 0
