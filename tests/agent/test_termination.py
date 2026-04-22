import importlib
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, AgentConfig
from agiwo.agent.models.log import (
    LLMCallCompleted,
    LLMCallStarted,
    RunLogEntryKind,
)
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.run import RunIdentity
from agiwo.agent.models.step import LLMCallContext, StepMetrics, StepView
from agiwo.agent import TerminationReason
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.agent.termination.summarizer import maybe_generate_termination_summary
from agiwo.llm.base import Model, StreamChunk


class _FixedResponseModel(Model):
    def __init__(self, response: str = "ok") -> None:
        super().__init__(id="term-model", name="term-model", temperature=0.0)
        self._response = response

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


def test_termination_modules_expose_prompt_and_summary_helpers() -> None:
    prompts_module = importlib.import_module("agiwo.agent.termination.prompts")
    limits_module = importlib.import_module("agiwo.agent.termination.limits")
    summarizer_module = importlib.import_module("agiwo.agent.termination.summarizer")

    assert hasattr(prompts_module, "DEFAULT_TERMINATION_USER_PROMPT")
    assert hasattr(prompts_module, "TERMINATION_SUMMARY_REASONS")
    assert hasattr(prompts_module, "format_termination_reason")
    assert hasattr(prompts_module, "render_termination_summary_prompt")
    assert hasattr(limits_module, "check_non_recoverable_limits")
    assert hasattr(limits_module, "check_post_llm_limits")
    assert hasattr(summarizer_module, "maybe_generate_termination_summary")


def test_render_termination_summary_prompt_substitutes_known_reason() -> None:
    prompts_module = importlib.import_module("agiwo.agent.termination.prompts")

    rendered = prompts_module.render_termination_summary_prompt(
        "Stopped because %s.",
        TerminationReason.MAX_STEPS,
    )

    assert rendered == "Stopped because reaching the maximum number of execution steps."


def test_render_termination_summary_prompt_preserves_templates_without_placeholder() -> (
    None
):
    prompts_module = importlib.import_module("agiwo.agent.termination.prompts")

    rendered = prompts_module.render_termination_summary_prompt(
        "Use this exact custom prompt.",
        TerminationReason.CANCELLED,
    )

    assert rendered == "Use this exact custom prompt."


def test_format_termination_reason_accepts_raw_string_fallback() -> None:
    prompts_module = importlib.import_module("agiwo.agent.termination.prompts")

    assert prompts_module.format_termination_reason("custom_stop") == "custom_stop"


def _make_context() -> RunContext:
    return RunContext(
        identity=RunIdentity(
            run_id="run-1",
            agent_id="agent-1",
            agent_name="agent",
        ),
        session_runtime=SessionRuntime(
            session_id="session-1",
            run_log_storage=InMemoryRunLogStorage(),
        ),
    )


def test_check_non_recoverable_limits_returns_max_steps_reason() -> None:
    limits_module = importlib.import_module("agiwo.agent.termination.limits")
    state = _make_context()

    reason = limits_module.check_non_recoverable_limits(
        state,
        AgentOptions(max_steps=3),
        current_step=3,
    )

    assert reason == TerminationReason.MAX_STEPS


def test_check_post_llm_limits_returns_max_input_reason() -> None:
    limits_module = importlib.import_module("agiwo.agent.termination.limits")
    state = _make_context()
    step = StepView.assistant(
        state,
        sequence=1,
        content="answer",
        metrics=StepMetrics(input_tokens=42),
    )

    reason = limits_module.check_post_llm_limits(
        state,
        step,
        LLMCallContext(messages=[]),
        options=AgentOptions(),
        max_input_tokens_per_call=10,
    )

    assert reason == TerminationReason.MAX_INPUT_TOKENS_PER_CALL


@pytest.mark.asyncio
async def test_run_records_termination_decision_entry() -> None:
    agent = Agent(
        AgentConfig(
            name="termination-test",
            description="termination test",
            options=AgentOptions(max_steps=0),
        ),
        model=_FixedResponseModel(),
    )

    result = await agent.run("hello", session_id="termination-session")

    assert result.termination_reason == TerminationReason.MAX_STEPS
    entries = await agent.run_log_storage.list_entries(session_id="termination-session")
    termination_entry = next(
        entry for entry in entries if entry.kind is RunLogEntryKind.TERMINATION_DECIDED
    )
    assert termination_entry.termination_reason == TerminationReason.MAX_STEPS
    assert termination_entry.phase == "pre_llm"
    assert termination_entry.source == "non_recoverable_limit"


@pytest.mark.asyncio
async def test_termination_summary_writes_canonical_llm_call_facts() -> None:
    state = _make_context()
    state.ledger.termination_reason = TerminationReason.MAX_STEPS
    state.ledger.messages = [{"role": "user", "content": "hello"}]

    await maybe_generate_termination_summary(
        state=state,
        options=AgentOptions(enable_termination_summary=True),
        model=_FixedResponseModel(response="summary response"),
        abort_signal=None,
    )

    entries = await state.session_runtime.list_run_log_entries(run_id="run-1")
    llm_started = [entry for entry in entries if isinstance(entry, LLMCallStarted)]
    llm_completed = [entry for entry in entries if isinstance(entry, LLMCallCompleted)]

    assert len(llm_started) == 1
    assert len(llm_completed) == 1
    assert "Execution Limit Reached" in llm_started[0].messages[-1]["content"]
    assert "Please provide a summary report" in llm_started[0].messages[-1]["content"]
    assert llm_completed[0].content == "summary response"
