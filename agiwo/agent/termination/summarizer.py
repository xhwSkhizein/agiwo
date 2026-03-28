"""Termination-summary execution for interrupted agent runs."""

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.llm_caller import stream_assistant_step
from agiwo.agent.models.step import StepRecord
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.step_committer import commit_step
from agiwo.agent.termination.prompts import (
    DEFAULT_TERMINATION_USER_PROMPT,
    TERMINATION_SUMMARY_REASONS,
    render_termination_summary_prompt,
)
from agiwo.llm.base import Model
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


async def maybe_generate_termination_summary(
    *,
    state: RunContext,
    options: AgentOptions,
    model: Model,
    abort_signal: AbortSignal | None,
) -> None:
    if not options.enable_termination_summary:
        return
    if state.termination_reason not in TERMINATION_SUMMARY_REASONS:
        return

    prompt_template = (
        options.termination_summary_prompt or DEFAULT_TERMINATION_USER_PROMPT
    )
    user_prompt = render_termination_summary_prompt(
        prompt_template,
        state.termination_reason,
    )

    sequence = await state.session_runtime.allocate_sequence()
    summary_user_step = StepRecord.user(
        state,
        sequence=sequence,
        content=user_prompt,
        name="summary_request",
    )
    await commit_step(state, summary_user_step, append_message=True)

    step, llm_context = await stream_assistant_step(
        model,
        state,
        abort_signal,
        messages=state.copy_messages(),
        tools=None,
    )
    step.name = "summary"
    await commit_step(state, step, llm=llm_context, append_message=False)

    logger.info(
        "summary_generated",
        tokens=step.metrics.total_tokens if step.metrics else 0,
    )


__all__ = ["maybe_generate_termination_summary"]
