import pytest

from agiwo.agent.models.log import (
    ReviewCheckpointRecorded,
    ReviewMilestonesUpdated,
    ReviewOutcomeRecorded,
    ReviewTriggerDecided,
    RunLogEntryKind,
)
from agiwo.agent.models.review import Milestone
from agiwo.agent.storage.serialization import (
    deserialize_run_log_entry_from_storage,
    serialize_run_log_entry_for_storage,
)


def test_review_milestones_updated_round_trips() -> None:
    entry = ReviewMilestonesUpdated(
        sequence=10,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        milestones=[
            Milestone(id="inspect", description="Inspect auth flow", status="active")
        ],
        active_milestone_id="inspect",
        source_tool_call_id="tc-milestones",
        source_step_id="step-milestones",
        reason="declared",
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert payload["kind"] == RunLogEntryKind.REVIEW_MILESTONES_UPDATED.value
    assert isinstance(restored, ReviewMilestonesUpdated)
    assert restored.milestones[0].id == "inspect"
    assert restored.active_milestone_id == "inspect"
    assert restored.reason == "declared"


def test_review_trigger_decided_round_trips() -> None:
    entry = ReviewTriggerDecided(
        sequence=11,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        trigger_reason="step_interval",
        active_milestone_id="inspect",
        review_count_since_checkpoint=8,
        trigger_tool_call_id="tc-search",
        trigger_tool_step_id="step-search",
        notice_step_id="step-search",
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert payload["kind"] == RunLogEntryKind.REVIEW_TRIGGER_DECIDED.value
    assert isinstance(restored, ReviewTriggerDecided)
    assert restored.trigger_reason == "step_interval"
    assert restored.review_count_since_checkpoint == 8


def test_review_checkpoint_recorded_round_trips() -> None:
    entry = ReviewCheckpointRecorded(
        sequence=12,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        checkpoint_seq=42,
        milestone_id="inspect",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert payload["kind"] == RunLogEntryKind.REVIEW_CHECKPOINT_RECORDED.value
    assert isinstance(restored, ReviewCheckpointRecorded)
    assert restored.checkpoint_seq == 42
    assert restored.milestone_id == "inspect"


def test_review_outcome_recorded_round_trips() -> None:
    entry = ReviewOutcomeRecorded(
        sequence=13,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        aligned=False,
        mode="step_back",
        experience="JWT search was not useful",
        active_milestone_id="inspect",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
        hidden_step_ids=["step-review-call", "step-review"],
        notice_cleaned_step_ids=["step-search"],
        condensed_step_ids=["step-search"],
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert payload["kind"] == RunLogEntryKind.REVIEW_OUTCOME_RECORDED.value
    assert isinstance(restored, ReviewOutcomeRecorded)
    assert restored.aligned is False
    assert restored.mode == "step_back"
    assert restored.condensed_step_ids == ["step-search"]


def test_review_outcome_recorded_round_trips_unknown_alignment() -> None:
    entry = ReviewOutcomeRecorded(
        sequence=14,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        aligned=None,
        mode="metadata_only",
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert "aligned" not in payload
    assert isinstance(restored, ReviewOutcomeRecorded)
    assert restored.aligned is None


def test_review_milestones_updated_rejects_invalid_milestone_payload() -> None:
    with pytest.raises(ValueError, match="Invalid ReviewMilestonesUpdated"):
        deserialize_run_log_entry_from_storage(
            {
                "kind": RunLogEntryKind.REVIEW_MILESTONES_UPDATED.value,
                "sequence": 15,
                "session_id": "sess-1",
                "run_id": "run-1",
                "agent_id": "agent-1",
                "milestones": ["bad-payload"],
            }
        )
