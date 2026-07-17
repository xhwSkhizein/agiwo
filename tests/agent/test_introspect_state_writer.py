import pytest

from agiwo.agent.introspect.models import Milestone
from agiwo.agent.models.log import (
    IntrospectionOutcomeRecorded,
    RunPlanUpdated,
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
async def test_writer_records_run_plan_updated() -> None:
    storage = InMemoryRunLogStorage()
    context = _make_context(storage)
    writer = RunStateWriter(context)

    entries = await writer.record_run_plan_updated(
        milestones=[Milestone(id="inspect", description="Inspect", status="active")],
        revision=1,
        source_tool_call_id="tc",
        source_step_id="step",
        reason="declared",
    )

    assert isinstance(entries[0], RunPlanUpdated)
    assert entries[0].revision == 1
    assert entries[0].active_milestone_id == "inspect"
    stored = await storage.list_entries(session_id="sess")
    assert isinstance(stored[0], RunPlanUpdated)


@pytest.mark.asyncio
async def test_writer_records_introspection_outcome_boundary() -> None:
    storage = InMemoryRunLogStorage()
    context = _make_context(storage)
    writer = RunStateWriter(context)

    entries = await writer.record_introspection_outcome_recorded(
        aligned=False,
        experience="drifted",
        tool_usefulness=[
            {"tool_call_id": "tc-search", "tool_name": "search", "score": 1}
        ],
        active_milestone_id="inspect",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
        boundary_seq=12,
    )

    assert isinstance(entries[0], IntrospectionOutcomeRecorded)
    assert entries[0].boundary_seq == 12
