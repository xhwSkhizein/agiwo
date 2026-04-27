import pytest

from agiwo.agent.introspect.models import Milestone
from agiwo.agent.models.log import (
    GoalMilestonesUpdated,
    IntrospectionOutcomeRecorded,
    StepCondensedContentUpdated,
)
from agiwo.agent.models.run import RunIdentity
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_writer import RunStateWriter
from agiwo.agent.storage.base import InMemoryRunLogStorage


def _make_context(storage: InMemoryRunLogStorage) -> RunContext:
    return RunContext(
        identity=RunIdentity(run_id="run", agent_id="agent", agent_name="agent"),
        session_runtime=SessionRuntime(session_id="sess", run_log_storage=storage),
    )


@pytest.mark.asyncio
async def test_writer_records_goal_update() -> None:
    storage = InMemoryRunLogStorage()
    context = _make_context(storage)
    writer = RunStateWriter(context)

    entries = await writer.record_goal_milestones_updated(
        milestones=[Milestone(id="inspect", description="Inspect", status="active")],
        active_milestone_id="inspect",
        source_tool_call_id="tc",
        source_step_id="step",
        reason="declared",
    )

    assert isinstance(entries[0], GoalMilestonesUpdated)
    stored = await storage.list_entries(session_id="sess")
    assert isinstance(stored[0], GoalMilestonesUpdated)


@pytest.mark.asyncio
async def test_writer_records_introspection_outcome_boundary() -> None:
    storage = InMemoryRunLogStorage()
    context = _make_context(storage)
    writer = RunStateWriter(context)

    entries = await writer.record_introspection_outcome_recorded(
        aligned=False,
        mode="step_back",
        experience="drifted",
        active_milestone_id="inspect",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
        hidden_step_ids=["step-call", "step-review"],
        notice_cleaned_step_ids=[],
        condensed_step_ids=["step-search"],
        boundary_seq=12,
        repair_start_seq=4,
        repair_end_seq=11,
    )

    assert isinstance(entries[0], IntrospectionOutcomeRecorded)
    assert entries[0].boundary_seq == 12


@pytest.mark.asyncio
async def test_writer_records_step_condensed_content_update() -> None:
    storage = InMemoryRunLogStorage()
    context = _make_context(storage)
    writer = RunStateWriter(context)

    entries = await writer.record_step_condensed_content_updated(
        step_id="step-search",
        condensed_content="[EXPERIENCE] drifted",
    )

    assert isinstance(entries[0], StepCondensedContentUpdated)
    assert entries[0].step_id == "step-search"
