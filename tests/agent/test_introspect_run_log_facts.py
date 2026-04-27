from agiwo.agent.introspect.models import Milestone
from agiwo.agent.models.log import (
    ContextRepairApplied,
    GoalMilestonesUpdated,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    RunLogEntryKind,
)
from agiwo.agent.storage.serialization import (
    deserialize_run_log_entry_from_storage,
    serialize_run_log_entry_for_storage,
)


def test_goal_milestones_updated_round_trips() -> None:
    entry = GoalMilestonesUpdated(
        sequence=1,
        session_id="sess",
        run_id="run",
        agent_id="agent",
        milestones=[Milestone(id="inspect", description="Inspect", status="active")],
        active_milestone_id="inspect",
        source_tool_call_id="tc",
        source_step_id="step",
        reason="declared",
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert payload["kind"] == RunLogEntryKind.GOAL_MILESTONES_UPDATED.value
    assert isinstance(restored, GoalMilestonesUpdated)
    assert restored.milestones[0].id == "inspect"


def test_introspection_outcome_round_trips_boundary_and_repair_range() -> None:
    entry = IntrospectionOutcomeRecorded(
        sequence=2,
        session_id="sess",
        run_id="run",
        agent_id="agent",
        aligned=False,
        mode="step_back",
        boundary_seq=12,
        repair_start_seq=4,
        repair_end_seq=11,
        condensed_step_ids=["step-search"],
    )

    restored = deserialize_run_log_entry_from_storage(
        serialize_run_log_entry_for_storage(entry)
    )

    assert isinstance(restored, IntrospectionOutcomeRecorded)
    assert restored.boundary_seq == 12
    assert restored.repair_start_seq == 4


def test_context_repair_applied_round_trips() -> None:
    entry = ContextRepairApplied(
        sequence=3,
        session_id="sess",
        run_id="run",
        agent_id="agent",
        mode="step_back",
        affected_count=1,
        start_seq=4,
        end_seq=11,
        experience="drifted",
    )

    restored = deserialize_run_log_entry_from_storage(
        serialize_run_log_entry_for_storage(entry)
    )

    assert isinstance(restored, ContextRepairApplied)
    assert restored.mode == "step_back"
    assert restored.affected_count == 1


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
