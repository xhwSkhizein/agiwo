import pytest

from agiwo.agent.models.log import (
    ReviewCheckpointRecorded,
    ReviewMilestonesUpdated,
    ReviewOutcomeRecorded,
    ReviewTriggerDecided,
)
from agiwo.agent.models.review import Milestone
from agiwo.agent.models.run import RunIdentity
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_writer import (
    RunStateWriter,
    build_review_checkpoint_recorded_entry,
    build_review_milestones_updated_entry,
    build_review_outcome_recorded_entry,
    build_review_trigger_decided_entry,
)
from agiwo.agent.storage.base import InMemoryRunLogStorage


def _context() -> RunContext:
    return RunContext(
        identity=RunIdentity(
            run_id="run-1",
            agent_id="agent-1",
            agent_name="agent",
        ),
        session_runtime=SessionRuntime(
            session_id="sess-1",
            run_log_storage=InMemoryRunLogStorage(),
        ),
    )


@pytest.mark.asyncio
async def test_state_writer_records_review_facts() -> None:
    context = _context()
    writer = RunStateWriter(context)

    entries = []
    entries.extend(
        await writer.record_review_milestones_updated(
            milestones=[
                Milestone(id="inspect", description="Inspect auth", status="active")
            ],
            active_milestone_id="inspect",
            source_tool_call_id="tc-milestones",
            source_step_id="step-milestones",
            reason="declared",
        )
    )
    entries.extend(
        await writer.record_review_trigger_decided(
            trigger_reason="step_interval",
            active_milestone_id="inspect",
            review_count_since_checkpoint=8,
            trigger_tool_call_id="tc-search",
            trigger_tool_step_id="step-search",
            notice_step_id="step-search",
        )
    )
    entries.extend(
        await writer.record_review_checkpoint_recorded(
            checkpoint_seq=9,
            milestone_id="inspect",
            review_tool_call_id="tc-review",
            review_step_id="step-review",
        )
    )
    entries.extend(
        await writer.record_review_outcome_recorded(
            aligned=True,
            mode="metadata_only",
            experience=None,
            active_milestone_id="inspect",
            review_tool_call_id="tc-review",
            review_step_id="step-review",
            hidden_step_ids=["step-review-call", "step-review"],
            notice_cleaned_step_ids=["step-search"],
            condensed_step_ids=[],
        )
    )

    assert [type(entry) for entry in entries] == [
        ReviewMilestonesUpdated,
        ReviewTriggerDecided,
        ReviewCheckpointRecorded,
        ReviewOutcomeRecorded,
    ]


def test_review_fact_builders_use_run_context_identity() -> None:
    context = _context()

    entries = [
        build_review_milestones_updated_entry(
            context,
            sequence=1,
            milestones=[
                Milestone(id="inspect", description="Inspect auth", status="active")
            ],
            active_milestone_id="inspect",
            source_tool_call_id="tc-milestones",
            source_step_id="step-milestones",
            reason="declared",
        ),
        build_review_trigger_decided_entry(
            context,
            sequence=2,
            trigger_reason="step_interval",
            active_milestone_id="inspect",
            review_count_since_checkpoint=8,
            trigger_tool_call_id=None,
            trigger_tool_step_id=None,
            notice_step_id=None,
        ),
        build_review_checkpoint_recorded_entry(
            context,
            sequence=3,
            checkpoint_seq=9,
            milestone_id="inspect",
            review_tool_call_id="tc-review",
            review_step_id="step-review",
        ),
        build_review_outcome_recorded_entry(
            context,
            sequence=4,
            aligned=None,
            mode="metadata_only",
            experience=None,
            active_milestone_id="inspect",
            review_tool_call_id="tc-review",
            review_step_id="step-review",
            hidden_step_ids=[],
            notice_cleaned_step_ids=[],
            condensed_step_ids=[],
        ),
    ]

    assert [entry.sequence for entry in entries] == [1, 2, 3, 4]
    assert {entry.session_id for entry in entries} == {"sess-1"}
    assert {entry.run_id for entry in entries} == {"run-1"}
    assert {entry.agent_id for entry in entries} == {"agent-1"}
    assert isinstance(entries[1], ReviewTriggerDecided)
    assert entries[1].trigger_tool_call_id is None
