from datetime import datetime, timezone

import pytest

from agiwo.agent import (
    CompactionApplied,
    MessageRole,
    RetrospectApplied,
    RunFinished,
    RunStarted,
    TerminationDecided,
    TerminationReason,
    UserStepCommitted,
)
from agiwo.agent.storage.base import InMemoryRunLogStorage

from server.services.runtime.run_query_service import RunQueryService


@pytest.mark.asyncio
async def test_run_query_service_lists_runs_newest_first() -> None:
    now = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            RunStarted(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                user_input="one",
                created_at=now,
            ),
            RunFinished(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                response="first",
                created_at=now,
            ),
            RunStarted(
                sequence=3,
                session_id="sess-1",
                run_id="run-2",
                agent_id="agent-1",
                user_input="two",
                created_at=now.replace(minute=1),
            ),
            RunFinished(
                sequence=4,
                session_id="sess-1",
                run_id="run-2",
                agent_id="agent-1",
                response="second",
                created_at=now.replace(minute=1),
            ),
        ]
    )
    service = RunQueryService(run_storage=storage)

    page = await service.list_runs(session_id="sess-1", limit=20, offset=0)

    assert [run.run_id for run in page.items] == ["run-2", "run-1"]
    assert page.has_more is False
    assert page.total is None


@pytest.mark.asyncio
async def test_run_query_service_lists_session_steps_and_total() -> None:
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            UserStepCommitted(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                step_id="step-1",
                role=MessageRole.USER,
                content="hello",
                user_input="hello",
            )
        ]
    )
    service = RunQueryService(run_storage=storage)

    page = await service.list_session_steps("sess-1", limit=20, order="asc")

    assert [step.id for step in page.items] == ["step-1"]
    assert page.total == 1
    assert page.has_more is False


@pytest.mark.asyncio
async def test_run_query_service_exposes_runtime_decision_state() -> None:
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            CompactionApplied(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                start_sequence=1,
                end_sequence=2,
                before_token_estimate=500,
                after_token_estimate=120,
                message_count=2,
                transcript_path="/tmp/compact.json",
                summary="compact",
            ),
            RetrospectApplied(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                affected_sequences=[2],
                affected_step_ids=["step-2"],
                replacement="summary",
                trigger="token_threshold",
            ),
            TerminationDecided(
                sequence=3,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                termination_reason=TerminationReason.COMPLETED,
                phase="after_tool_batch",
                source="finished",
            ),
        ]
    )
    service = RunQueryService(run_storage=storage)

    state = await service.get_runtime_decision_state("sess-1", agent_id="agent-1")
    snapshot = await service.get_session_run_snapshot("sess-1")

    assert state.latest_compaction is not None
    assert state.latest_compaction.summary == "compact"
    assert state.latest_retrospect is not None
    assert state.latest_retrospect.trigger == "token_threshold"
    assert state.latest_termination is not None
    assert state.latest_termination.reason is TerminationReason.COMPLETED
    assert snapshot.runtime_decisions.latest_termination is not None
