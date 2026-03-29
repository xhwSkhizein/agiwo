from datetime import datetime

import pytest

from agiwo.agent.models.run import CompactMetadata
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_ops import (
    record_compaction_metadata,
    replace_messages,
    set_termination_reason,
    track_step_state,
)
from agiwo.agent.storage.base import InMemoryRunStepStorage
from agiwo.agent.types import (
    StepMetrics,
    StepRecord,
    TerminationReason,
)


def _make_state() -> RunContext:
    return RunContext(
        session_runtime=SessionRuntime(
            session_id="session-1",
            run_step_storage=InMemoryRunStepStorage(),
        ),
        run_id="run-1",
        agent_id="agent-1",
        agent_name="agent",
    )


def test_track_step_state_updates_counters_response_and_messages() -> None:
    state = _make_state()
    step = StepRecord.assistant(
        state,
        sequence=1,
        content="final answer",
        tool_calls=[{"id": "call-1"}, {"id": "call-2"}],
        metrics=StepMetrics(
            token_cost=1.5,
            total_tokens=10,
            input_tokens=4,
            output_tokens=6,
            cache_read_tokens=2,
            cache_creation_tokens=3,
        ),
    )

    track_step_state(state, step)

    ledger = state.ledger
    assert ledger.steps_count == 1
    assert ledger.assistant_steps_count == 1
    assert ledger.tool_calls_count == 2
    assert ledger.response_content == "final answer"
    assert ledger.token_cost == pytest.approx(1.5)
    assert ledger.total_tokens == 10
    assert ledger.input_tokens == 4
    assert ledger.output_tokens == 6
    assert ledger.cache_read_tokens == 2
    assert ledger.cache_creation_tokens == 3
    assert ledger.messages == [step.to_message()]


def test_track_step_state_skips_message_append_when_disabled() -> None:
    state = _make_state()
    step = StepRecord.user(state, sequence=1, content="hello")

    track_step_state(state, step, append_message=False)

    ledger = state.ledger
    assert ledger.steps_count == 1
    assert ledger.assistant_steps_count == 0
    assert ledger.messages == []


def test_runtime_state_ops_only_touch_mutable_ledger_state() -> None:
    state = _make_state()

    replace_messages(state, [{"role": "assistant", "content": "summary"}])
    record_compaction_metadata(
        state,
        CompactMetadata(
            session_id="session-1",
            agent_id="agent-1",
            start_seq=1,
            end_seq=2,
            before_token_estimate=100,
            after_token_estimate=10,
            message_count=2,
            transcript_path="/tmp/t.jsonl",
            analysis={"summary": "summary"},
            created_at=datetime(2026, 3, 26, 12, 0, 0),
        ),
    )
    set_termination_reason(state, TerminationReason.CANCELLED)

    ledger = state.ledger
    assert ledger.messages == [{"role": "assistant", "content": "summary"}]
    assert ledger.last_compact_metadata is not None
    assert ledger.termination_reason == TerminationReason.CANCELLED
    assert state.run_id == "run-1"
    assert state.session_id == "session-1"


def test_ledger_fields_not_exposed_directly_on_run_context() -> None:
    """Ledger state (messages, counters, etc.) lives only in ``state.ledger``.

    There must be no read path for these fields directly on RunContext,
    so accidental ``state.messages`` or ``state.termination_reason`` will
    fail loudly rather than silently read stale data.
    """
    state = _make_state()

    removed_attrs = [
        "messages",
        "tool_schemas",
        "termination_reason",
        "last_compact_metadata",
        "total_tokens",
        "steps_count",
        "response_content",
        "tool_calls_count",
    ]
    for attr in removed_attrs:
        assert not hasattr(state, attr), (
            f"RunContext should not expose '{attr}' directly"
        )

    replace_messages(state, [{"role": "assistant", "content": "summary"}])
    snapshot = state.copy_messages()
    snapshot[0]["content"] = "mutated"
    assert state.ledger.messages == [{"role": "assistant", "content": "summary"}]
