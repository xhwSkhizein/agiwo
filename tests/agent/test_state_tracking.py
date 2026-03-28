from datetime import datetime

import pytest

from agiwo.agent.models.compact import CompactMetadata
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_ops import (
    record_compaction_metadata,
    replace_messages,
    set_termination_reason,
    track_step_state,
)
from agiwo.agent.storage.base import InMemoryRunStepStorage
from agiwo.agent.storage.session import InMemorySessionStorage
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
            session_storage=InMemorySessionStorage(),
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

    assert state.steps_count == 1
    assert state.assistant_steps_count == 1
    assert state.tool_calls_count == 2
    assert state.response_content == "final answer"
    assert state.token_cost == pytest.approx(1.5)
    assert state.total_tokens == 10
    assert state.input_tokens == 4
    assert state.output_tokens == 6
    assert state.cache_read_tokens == 2
    assert state.cache_creation_tokens == 3
    assert list(state.messages) == [step.to_message()]


def test_track_step_state_skips_message_append_when_disabled() -> None:
    state = _make_state()
    step = StepRecord.user(state, sequence=1, content="hello")

    track_step_state(state, step, append_message=False)

    assert state.steps_count == 1
    assert state.assistant_steps_count == 0
    assert list(state.messages) == []


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

    assert list(state.messages) == [{"role": "assistant", "content": "summary"}]
    assert state.last_compact_metadata is not None
    assert state.termination_reason == TerminationReason.CANCELLED
    assert state.run_id == "run-1"
    assert state.session_id == "session-1"


def test_run_context_disallows_direct_assignment_to_mutable_ledger_fields() -> None:
    state = _make_state()
    metadata = CompactMetadata(
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
    )

    with pytest.raises(AttributeError):
        state.messages = [{"role": "assistant", "content": "summary"}]

    with pytest.raises(AttributeError):
        state.messages.append({"role": "assistant", "content": "summary"})

    with pytest.raises(AttributeError):
        state.tool_schemas = [{"type": "function"}]

    with pytest.raises(AttributeError):
        state.termination_reason = TerminationReason.CANCELLED

    with pytest.raises(AttributeError):
        state.last_compact_metadata = metadata

    replace_messages(state, [{"role": "assistant", "content": "summary"}])
    snapshot = state.messages
    snapshot[0]["content"] = "mutated"
    assert list(state.messages) == [{"role": "assistant", "content": "summary"}]
