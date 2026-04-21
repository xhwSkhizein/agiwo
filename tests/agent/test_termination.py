import importlib

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.step import LLMCallContext, StepMetrics, StepRecord
from agiwo.agent import TerminationReason
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunStepStorage


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
        session_runtime=SessionRuntime(
            session_id="session-1",
            run_step_storage=InMemoryRunStepStorage(),
        ),
        run_id="run-1",
        agent_id="agent-1",
        agent_name="agent",
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
    step = StepRecord.assistant(
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
