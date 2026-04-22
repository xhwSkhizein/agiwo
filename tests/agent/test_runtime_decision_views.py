from datetime import datetime, timezone

import pytest

from agiwo.agent import (
    CompactionApplied,
    RetrospectApplied,
    RunRolledBack,
    TerminationDecided,
    TerminationReason,
)
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.agent.storage.serialization import build_runtime_decision_state_from_entries
from agiwo.agent.storage.sqlite import SQLiteRunLogStorage


def _decision_entries() -> list:
    now = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)
    return [
        TerminationDecided(
            sequence=1,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            termination_reason=TerminationReason.MAX_STEPS,
            phase="before_termination",
            source="limit",
            created_at=now,
        ),
        CompactionApplied(
            sequence=2,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            start_sequence=1,
            end_sequence=8,
            before_token_estimate=1000,
            after_token_estimate=200,
            message_count=4,
            transcript_path="/tmp/compact.json",
            summary="short summary",
            created_at=now,
        ),
        RetrospectApplied(
            sequence=3,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            affected_sequences=[7, 8],
            affected_step_ids=["step-7"],
            feedback="summarized",
            replacement="summary",
            trigger="token_threshold",
            created_at=now,
        ),
        RunRolledBack(
            sequence=4,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            start_sequence=2,
            end_sequence=4,
            reason="scheduler_no_progress_periodic",
            created_at=now,
        ),
        TerminationDecided(
            sequence=5,
            session_id="sess-1",
            run_id="run-2",
            agent_id="agent-1",
            termination_reason=TerminationReason.TIMEOUT,
            phase="after_tool_batch",
            source="scheduler_timeout",
            created_at=now,
        ),
    ]


def test_build_runtime_decision_state_from_entries_replays_latest_views() -> None:
    state = build_runtime_decision_state_from_entries(_decision_entries())

    assert state.latest_termination is not None
    assert state.latest_termination.reason is TerminationReason.TIMEOUT
    assert state.latest_termination.run_id == "run-2"
    assert state.latest_compaction is not None
    assert state.latest_compaction.metadata.start_seq == 1
    assert state.latest_compaction.summary == "short summary"
    assert state.latest_retrospect is not None
    assert state.latest_retrospect.affected_sequences == (7, 8)
    assert state.latest_retrospect.trigger == "token_threshold"
    assert state.latest_rollback is not None
    assert state.latest_rollback.end_sequence == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_kind", ["memory", "sqlite"])
async def test_storage_get_runtime_decision_state_filters_latest_entries(
    tmp_path,
    storage_kind: str,
) -> None:
    if storage_kind == "memory":
        storage = InMemoryRunLogStorage()
    else:
        storage = SQLiteRunLogStorage(str(tmp_path / "run-log.db"))

    try:
        await storage.append_entries(_decision_entries())
        await storage.append_entries(
            [
                TerminationDecided(
                    sequence=10,
                    session_id="sess-2",
                    run_id="run-x",
                    agent_id="agent-9",
                    termination_reason=TerminationReason.CANCELLED,
                    phase="before_termination",
                    source="foreign_session",
                )
            ]
        )

        state = await storage.get_runtime_decision_state(
            session_id="sess-1",
            agent_id="agent-1",
        )
        run_one_state = await storage.get_runtime_decision_state(
            session_id="sess-1",
            agent_id="agent-1",
            run_id="run-1",
        )

        assert state.latest_termination is not None
        assert state.latest_termination.reason is TerminationReason.TIMEOUT
        assert state.latest_termination.run_id == "run-2"
        assert state.latest_retrospect is not None
        assert state.latest_retrospect.replacement == "summary"
        assert state.latest_compaction is not None
        assert state.latest_compaction.metadata.after_token_estimate == 200
        assert state.latest_rollback is not None
        assert state.latest_rollback.reason == "scheduler_no_progress_periodic"

        assert run_one_state.latest_termination is not None
        assert run_one_state.latest_termination.reason is TerminationReason.MAX_STEPS
        assert run_one_state.latest_termination.run_id == "run-1"
    finally:
        await storage.close()
