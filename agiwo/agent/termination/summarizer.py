"""Termination-summary execution for interrupted agent runs."""

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.llm_caller import stream_assistant_step
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
        started_entries = await writer.record_llm_call_started(
            messages=state.snapshot_messages(),
            tools=None,
        )
        await state.session_runtime.project_run_log_entries(
            started_entries,
            run_id=state.run_id,
            agent_id=state.agent_id,
            parent_run_id=state.parent_run_id,
            depth=state.depth,
        )
        step, llm_context = await stream_assistant_step(
            model,
            state,
            abort_signal,
            messages=state.snapshot_messages(),
            use_state_tools=False,
        )
        step.name = "summary"
        await commit_step(step, llm=llm_context, append_message=False)
        completed_entries = await writer.record_llm_call_completed(
            step=step,
            llm=llm_context,
        )
        await state.session_runtime.project_run_log_entries(
            completed_entries,
            run_id=state.run_id,
            agent_id=state.agent_id,
            parent_run_id=state.parent_run_id,
            depth=state.depth,
        )

        logger.info(
            "summary_generated",
            tokens=step.metrics.total_tokens if step.metrics else 0,
        )
    except Exception:  # noqa: BLE001 - summary is best-effort
        logger.warning(
            "summary_generation_failed",
            run_id=state.run_id,
            termination_reason=state.ledger.termination_reason,
            exc_info=True,
        )


__all__ = ["maybe_generate_termination_summary"]
