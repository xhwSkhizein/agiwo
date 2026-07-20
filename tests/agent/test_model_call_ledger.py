"""Tests for model-call ledger, RunLimitPolicy, and max_steps_per_run."""

from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, AgentConfig, AgentOptions
from agiwo.agent.llm_caller import ModelCallLimitExceeded, execute_model_call
from agiwo.agent.models.config import AgentOptions as AgentOptionsModel
from agiwo.agent.models.log import (
    LLMCallCompleted,
    LLMCallFailed,
    LLMCallStarted,
)
from agiwo.agent.models.model_call import ModelCallLedger, ModelCallPhase
from agiwo.agent.models.run import RunIdentity
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_writer import RunStateWriter
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.agent.termination.run_limit import RunLimitPolicy
from agiwo.agent import TerminationReason
from agiwo.llm.base import Model, StreamChunk


class _FixedResponseModel(Model):
    def __init__(self, response: str = "ok", *, fail_times: int = 0) -> None:
        super().__init__(id="ledger-model", name="ledger-model", temperature=0.0)
        self._response = response
        self._fail_times = fail_times
        self._calls = 0

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        self._calls += 1
        if self._calls <= self._fail_times:
            raise ConnectionError("simulated provider failure")
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


class _FlakyStreamModel(Model):
    def __init__(self) -> None:
        super().__init__(id="flaky", name="flaky", temperature=0.0)
        self._calls = 0

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        self._calls += 1
        if self._calls == 1:
            yield StreamChunk(content="partial ")
            raise RuntimeError("stream interrupted")
        yield StreamChunk(content="ok")
        yield StreamChunk(finish_reason="stop")


async def _noop_sleep(_seconds: float) -> None:
    return None


def _make_context(*, limit: int = 50) -> RunContext:
    session_runtime = SessionRuntime(
        session_id="session-ledger",
        run_log_storage=InMemoryRunLogStorage(),
    )
    context = RunContext(
        identity=RunIdentity(
            run_id="run-ledger",
            agent_id="agent-ledger",
            agent_name="ledger-agent",
        ),
        session_runtime=session_runtime,
    )
    context.ledger.model_calls.configured_limit = limit
    return context


def test_agent_options_rejects_legacy_max_steps() -> None:
    with pytest.raises(ValueError, match="max_steps_per_run"):
        AgentOptionsModel(max_steps=10)


def test_run_limit_policy_blocks_work_at_limit() -> None:
    ledger = ModelCallLedger(configured_limit=2, total_attempts=2)
    policy = RunLimitPolicy()

    decision = policy.check_before_attempt(ledger, ModelCallPhase.ASSISTANT)

    assert decision.allowed is False
    assert decision.reason == "work_limit_exceeded"


def test_run_limit_policy_allows_one_finalization_over_limit() -> None:
    ledger = ModelCallLedger(configured_limit=1, total_attempts=1)
    policy = RunLimitPolicy()

    summary = policy.check_before_attempt(ledger, ModelCallPhase.TERMINATION_SUMMARY)
    assignment = policy.check_before_attempt(ledger, ModelCallPhase.RUN_FINALIZATION)
    correction = policy.check_before_attempt(
        ledger, ModelCallPhase.FINALIZATION_CORRECTION
    )

    assert summary.allowed is True
    assert assignment.allowed is True
    assert correction.allowed is True


def test_run_limit_policy_allows_each_finalization_phase_once() -> None:
    ledger = ModelCallLedger(configured_limit=1, total_attempts=1)
    policy = RunLimitPolicy()

    for phase in (
        ModelCallPhase.RUN_FINALIZATION,
        ModelCallPhase.FINALIZATION_CORRECTION,
    ):
        decision = policy.check_before_attempt(ledger, phase)
        assert decision.allowed is True
        ledger.mark_finalization_consumed(phase)

    for phase in (
        ModelCallPhase.RUN_FINALIZATION,
        ModelCallPhase.FINALIZATION_CORRECTION,
    ):
        decision = policy.check_before_attempt(ledger, phase)
        assert decision.allowed is False


def test_run_limit_policy_exhausts_finalization_slot() -> None:
    ledger = ModelCallLedger(configured_limit=1, total_attempts=2)
    ledger.mark_finalization_consumed(ModelCallPhase.TERMINATION_SUMMARY)
    policy = RunLimitPolicy()

    decision = policy.check_before_attempt(ledger, ModelCallPhase.TERMINATION_SUMMARY)

    assert decision.allowed is False


@pytest.mark.asyncio
async def test_execute_model_call_records_phase_and_identity() -> None:
    state = _make_context(limit=5)
    writer = RunStateWriter(state)

    result = await execute_model_call(
        model=_FixedResponseModel("assistant reply"),
        state=state,
        writer=writer,
        phase=ModelCallPhase.ASSISTANT,
        abort_signal=None,
    )

    assert result.step.content == "assistant reply"
    assert state.ledger.model_calls.total_attempts == 1
    entries = await state.session_runtime.list_run_log_entries()
    started = next(entry for entry in entries if isinstance(entry, LLMCallStarted))
    completed = next(entry for entry in entries if isinstance(entry, LLMCallCompleted))
    assert started.phase is ModelCallPhase.ASSISTANT
    assert started.logical_call_id == completed.logical_call_id
    assert started.attempt_no == 1
    assert started.call_ordinal == 1


@pytest.mark.asyncio
async def test_provider_retry_keeps_logical_call_id_and_increments_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agiwo.agent.llm_caller.asyncio.sleep", _noop_sleep)
    state = _make_context(limit=5)
    writer = RunStateWriter(state)

    await execute_model_call(
        model=_FixedResponseModel("after retry", fail_times=1),
        state=state,
        writer=writer,
        phase=ModelCallPhase.ASSISTANT,
        abort_signal=None,
    )

    entries = await state.session_runtime.list_run_log_entries()
    failed = [entry for entry in entries if isinstance(entry, LLMCallFailed)]
    completed = [entry for entry in entries if isinstance(entry, LLMCallCompleted)]
    assert len(failed) == 1
    assert len(completed) == 1
    assert failed[0].logical_call_id == completed[0].logical_call_id
    assert failed[0].attempt_no == 1
    assert completed[0].attempt_no == 2
    assert failed[0].call_ordinal == 1
    assert completed[0].call_ordinal == 2
    assert state.ledger.model_calls.total_attempts == 2


@pytest.mark.asyncio
async def test_provider_retry_rechecks_work_limit_before_next_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed attempt that fills the work limit must not get a free retry."""
    monkeypatch.setattr("agiwo.agent.llm_caller.asyncio.sleep", _noop_sleep)
    state = _make_context(limit=1)
    writer = RunStateWriter(state)
    model = _FixedResponseModel("should-not-complete", fail_times=2)

    with pytest.raises(ModelCallLimitExceeded):
        await execute_model_call(
            model=model,
            state=state,
            writer=writer,
            phase=ModelCallPhase.ASSISTANT,
            abort_signal=None,
        )

    assert model._calls == 1
    assert state.ledger.model_calls.total_attempts == 1
    entries = await state.session_runtime.list_run_log_entries()
    assert any(isinstance(entry, LLMCallFailed) for entry in entries)


@pytest.mark.asyncio
async def test_finalization_over_limit_retry_has_no_second_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over-limit finalization is single-shot; provider retry is refused."""
    monkeypatch.setattr("agiwo.agent.llm_caller.asyncio.sleep", _noop_sleep)
    state = _make_context(limit=1)
    state.ledger.model_calls.total_attempts = 1
    state.ledger.model_calls.limit_trigger_ordinal = 1
    writer = RunStateWriter(state)
    model = _FixedResponseModel("summary", fail_times=2)

    with pytest.raises(ModelCallLimitExceeded):
        await execute_model_call(
            model=model,
            state=state,
            writer=writer,
            phase=ModelCallPhase.TERMINATION_SUMMARY,
            abort_signal=None,
            messages=[{"role": "user", "content": "summarize"}],
        )

    assert model._calls == 1
    assert state.ledger.model_calls.total_attempts == 2
    assert state.ledger.model_calls.is_finalization_consumed(
        ModelCallPhase.TERMINATION_SUMMARY
    )
    entries = await state.session_runtime.list_run_log_entries()
    assert any(isinstance(entry, LLMCallFailed) for entry in entries)


@pytest.mark.asyncio
async def test_termination_summary_allowed_over_work_limit() -> None:
    state = _make_context(limit=1)
    state.ledger.model_calls.total_attempts = 1
    state.ledger.model_calls.limit_trigger_ordinal = 1
    state.ledger.termination_reason = TerminationReason.MAX_STEPS
    writer = RunStateWriter(state)

    await execute_model_call(
        model=_FixedResponseModel("summary"),
        state=state,
        writer=writer,
        phase=ModelCallPhase.TERMINATION_SUMMARY,
        abort_signal=None,
        messages=[{"role": "user", "content": "summarize"}],
    )

    entries = await state.session_runtime.list_run_log_entries()
    completed = next(entry for entry in entries if isinstance(entry, LLMCallCompleted))
    assert completed.phase is ModelCallPhase.TERMINATION_SUMMARY
    assert state.ledger.model_calls.total_attempts == 2
    assert state.ledger.model_calls.limit_trigger_ordinal == 1


@pytest.mark.asyncio
async def test_execute_model_call_raises_when_work_limit_exceeded() -> None:
    state = _make_context(limit=1)
    state.ledger.model_calls.total_attempts = 1
    writer = RunStateWriter(state)

    with pytest.raises(ModelCallLimitExceeded):
        await execute_model_call(
            model=_FixedResponseModel(),
            state=state,
            writer=writer,
            phase=ModelCallPhase.ASSISTANT,
            abort_signal=None,
        )


@pytest.mark.asyncio
async def test_run_metrics_include_model_call_ledger_fields() -> None:
    agent = Agent(
        AgentConfig(
            name="metrics-ledger",
            options=AgentOptions(max_steps_per_run=2, enable_termination_summary=False),
        ),
        model=_FixedResponseModel("done"),
    )

    result = await agent.run("hello", session_id="metrics-ledger-session")

    assert result.metrics is not None
    assert result.metrics.max_steps_per_run == 2
    assert result.metrics.model_call_attempts_total == 1
    assert result.metrics.model_call_phase_stats is not None
    assert result.metrics.model_call_phase_stats["assistant"]["completed"] == 1


@pytest.mark.asyncio
async def test_max_steps_per_run_zero_terminates_before_assistant() -> None:
    agent = Agent(
        AgentConfig(
            name="zero-limit",
            options=AgentOptions(max_steps_per_run=0, enable_termination_summary=False),
        ),
        model=_FixedResponseModel(),
    )

    result = await agent.run("hello", session_id="zero-limit-session")

    assert result.termination_reason == TerminationReason.MAX_STEPS
    entries = await agent.run_log_storage.list_entries(session_id="zero-limit-session")
    llm_started = [entry for entry in entries if isinstance(entry, LLMCallStarted)]
    assert llm_started == []
