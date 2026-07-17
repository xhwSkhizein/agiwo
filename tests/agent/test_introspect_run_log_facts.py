from agiwo.agent.introspect.models import Milestone
from agiwo.agent.models.log import (
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    RunLogEntryKind,
    RunPlanUpdated,
)
from agiwo.agent.storage.serialization import (
    deserialize_run_log_entry_from_storage,
    serialize_run_log_entry_for_storage,
)


def test_run_plan_updated_round_trips() -> None:
    entry = RunPlanUpdated(
        sequence=1,
        session_id="sess",
        run_id="run",
        agent_id="agent",
        milestones=[Milestone(id="inspect", description="Inspect", status="active")],
        revision=3,
        source_tool_call_id="tc",
        source_step_id="step",
        reason="declared",
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert payload["kind"] == RunLogEntryKind.RUN_PLAN_UPDATED.value
    assert isinstance(restored, RunPlanUpdated)
    assert restored.milestones[0].id == "inspect"
    assert restored.revision == 3
    assert restored.active_milestone_id == "inspect"


def test_introspection_outcome_round_trips_tool_usefulness() -> None:
    entry = IntrospectionOutcomeRecorded(
        sequence=2,
        session_id="sess",
        run_id="run",
        agent_id="agent",
        aligned=False,
        boundary_seq=12,
        experience="drifted",
        tool_usefulness=[
            {"tool_call_id": "tc-search", "tool_name": "search", "score": 2},
            {"tool_call_id": "tc-read", "tool_name": "read", "score": None},
        ],
    )

    restored = deserialize_run_log_entry_from_storage(
        serialize_run_log_entry_for_storage(entry)
    )

    assert isinstance(restored, IntrospectionOutcomeRecorded)
    assert restored.boundary_seq == 12
    assert restored.tool_usefulness[0]["score"] == 2
    assert restored.tool_usefulness[1]["score"] is None


def test_introspection_trigger_round_trips() -> None:
    entry = IntrospectionTriggered(
        sequence=4,
        session_id="sess",
        run_id="run",
        agent_id="agent",
        trigger_reason="step_interval",
        active_milestone_id="inspect",
        review_count_since_boundary=8,
        trigger_tool_call_id="tc",
        trigger_tool_step_id="step",
        notice_step_id="step",
    )

    restored = deserialize_run_log_entry_from_storage(
        serialize_run_log_entry_for_storage(entry)
    )

    assert isinstance(restored, IntrospectionTriggered)
    assert restored.review_count_since_boundary == 8
