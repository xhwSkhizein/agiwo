"""Termination-summary execution for interrupted agent runs."""

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.llm_caller import ModelCallLimitExceeded, execute_model_call
from agiwo.agent.models.model_call import ModelCallPhase
from agiwo.agent.models.step import StepView
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.step_commit import StepCommitter
from agiwo.agent.runtime.state_writer import RunStateWriter
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
    commit_step: StepCommitter,
) -> None:
    if not options.enable_termination_summary:
        return
    if state.ledger.termination_reason not in TERMINATION_SUMMARY_REASONS:
        return

    prompt_template = (
        options.termination_summary_prompt or DEFAULT_TERMINATION_USER_PROMPT
    )
    user_prompt = render_termination_summary_prompt(
        prompt_template,
        state.ledger.termination_reason,
    )
    writer = RunStateWriter(state)

    sequence = await state.session_runtime.allocate_sequence()
    summary_user_step = StepView.user(
        state,
        sequence=sequence,
        content=user_prompt,
        name="summary_request",
    )
    await commit_step(summary_user_step, append_message=True)

    try:
        call_result = await execute_model_call(
            model=model,
            state=state,
            writer=writer,
            phase=ModelCallPhase.TERMINATION_SUMMARY,
            abort_signal=abort_signal,
            messages=state.snapshot_messages(),
            use_state_tools=False,
        )
        step = call_result.step
        llm_context = call_result.llm_context
        step.name = "summary"
        await commit_step(step, llm=llm_context, append_message=False)

        logger.info(
            "summary_generated",
            tokens=step.metrics.total_tokens if step.metrics else 0,
        )
    except ModelCallLimitExceeded:
        logger.warning(
            "summary_generation_skipped_limit",
            run_id=state.run_id,
            termination_reason=state.ledger.termination_reason,
        )
    except Exception:  # noqa: BLE001 - summary is best-effort
        logger.warning(
            "summary_generation_failed",
            run_id=state.run_id,
            termination_reason=state.ledger.termination_reason,
            exc_info=True,
        )


__all__ = ["maybe_generate_termination_summary"]
