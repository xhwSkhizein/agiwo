from agiwo.agent.introspect.models import Milestone
from agiwo.agent.models.log import (
    GoalMilestonesUpdated,
    IntrospectionCheckpointRecorded,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    RunStarted,
)
from agiwo.agent.trace_writer import AgentTraceCollector


def test_introspection_run_log_facts_project_to_runtime_spans() -> None:
    trace = AgentTraceCollector().build_from_entries(
        [
            RunStarted(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                user_input="inspect",
            ),
            GoalMilestonesUpdated(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                milestones=[
                    Milestone(id="inspect", description="Inspect", status="active")
                ],
                active_milestone_id="inspect",
                source_tool_call_id="tc-milestones",
                source_step_id="step-milestones",
                reason="declared",
            ),
            IntrospectionTriggered(
                sequence=3,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                trigger_reason="step_interval",
                active_milestone_id="inspect",
                review_count_since_boundary=8,
                trigger_tool_call_id="tc-search",
                trigger_tool_step_id="step-search",
                notice_step_id="step-search",
            ),
            IntrospectionCheckpointRecorded(
                sequence=4,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                checkpoint_seq=42,
                milestone_id="inspect",
                review_tool_call_id="tc-review",
                review_step_id="step-review",
            ),
            IntrospectionOutcomeRecorded(
                sequence=5,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                aligned=True,
                mode="metadata_only",
                boundary_seq=42,
                active_milestone_id="inspect",
                review_tool_call_id="tc-review",
                review_step_id="step-review",
                hidden_step_ids=["step-review"],
            ),
        ]
    )

    spans_by_name = {span.name: span for span in trace.spans}
    assert spans_by_name["review_milestones"].attributes["milestones"] == [
        {
            "id": "inspect",
            "description": "Inspect",
            "status": "active",
            "declared_at_seq": 0,
            "completed_at_seq": None,
        }
    ]
    assert (
        spans_by_name["review_trigger"].attributes["review_count_since_checkpoint"] == 8
    )
    assert spans_by_name["review_checkpoint"].attributes["checkpoint_seq"] == 42
    assert spans_by_name["review_outcome"].attributes["aligned"] is True
