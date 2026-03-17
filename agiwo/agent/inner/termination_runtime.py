import time

from agiwo.agent.inner.llm_handler import LLMStreamHandler
from agiwo.agent.inner.run_recorder import RunRecorder
from agiwo.agent.inner.run_state import RunState
from agiwo.agent.inner.summarizer import (
    DEFAULT_TERMINATION_USER_PROMPT,
    _format_termination_reason,
)
from agiwo.agent.options import AgentOptions
from agiwo.agent.runtime import LLMCallContext, StepRecord, TerminationReason
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class ExecutionTerminationRuntime:
    """Own limit checks and termination summary generation."""

    SUMMARY_REASONS = frozenset(
        {
            TerminationReason.MAX_STEPS,
            TerminationReason.TIMEOUT,
            TerminationReason.MAX_RUN_COST,
            TerminationReason.ERROR_WITH_CONTEXT,
        }
    )

    def __init__(
        self,
        *,
        options: AgentOptions,
        max_input_tokens_per_call: int,
        llm_handler: LLMStreamHandler,
    ) -> None:
        self._options = options
        self._max_input_tokens_per_call = max_input_tokens_per_call
        self._llm_handler = llm_handler

    def check_non_recoverable_limits(self, state: RunState) -> TerminationReason | None:
        if state.current_step >= self._options.max_steps:
            logger.warning(
                "limit_hit_max_steps",
                current_step=state.current_step,
                max_steps=self._options.max_steps,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_STEPS

        if self._options.run_timeout and state.elapsed > self._options.run_timeout:
            logger.warning(
                "limit_hit_timeout",
                elapsed=state.elapsed,
                run_timeout=self._options.run_timeout,
                run_id=state.context.run_id,
            )
            return TerminationReason.TIMEOUT

        if state.context.timeout_at and time.time() >= state.context.timeout_at:
            logger.warning(
                "limit_hit_context_timeout",
                timeout_at=state.context.timeout_at,
                run_id=state.context.run_id,
            )
            return TerminationReason.TIMEOUT

        return None

    def check_post_llm_limits(
        self,
        state: RunState,
        step: StepRecord,
        llm_context: LLMCallContext,
    ) -> TerminationReason | None:
        input_tokens = step.metrics.input_tokens if step.metrics else 0
        if input_tokens and input_tokens > self._max_input_tokens_per_call:
            logger.warning(
                "limit_hit_max_input_tokens_per_call",
                input_tokens=input_tokens,
                max_input_tokens_per_call=self._max_input_tokens_per_call,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_INPUT_TOKENS_PER_CALL

        if (
            self._options.max_run_cost is not None
            and state.token_cost >= self._options.max_run_cost
        ):
            logger.warning(
                "limit_hit_max_run_cost",
                token_cost=state.token_cost,
                max_run_cost=self._options.max_run_cost,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_RUN_COST

        finish_reason = llm_context.finish_reason
        if finish_reason and finish_reason.strip().lower() in {"length", "max_tokens"}:
            logger.warning(
                "limit_hit_max_output_tokens",
                finish_reason=finish_reason,
                run_id=state.context.run_id,
            )
            return TerminationReason.MAX_OUTPUT_TOKENS

        return None

    async def maybe_generate_summary(
        self,
        *,
        state: RunState,
        run_recorder: RunRecorder,
        abort_signal: AbortSignal | None,
    ) -> None:
        if not self._options.enable_termination_summary:
            return
        if state.termination_reason not in self.SUMMARY_REASONS:
            return

        prompt_template = (
            self._options.termination_summary_prompt
            or DEFAULT_TERMINATION_USER_PROMPT
        )
        termination_reason_str = _format_termination_reason(state.termination_reason)
        user_prompt = (
            prompt_template % termination_reason_str
            if "%s" in prompt_template
            else prompt_template
        )

        summary_user_step = await run_recorder.create_user_step(
            content=user_prompt,
            name="summary_request",
        )
        await run_recorder.commit_step(summary_user_step, append_message=True)

        step, llm_context = await self._llm_handler.stream_assistant_step(
            state,
            run_recorder,
            abort_signal,
            messages=state.messages,
            tools=None,
        )
        step.name = "summary"
        await run_recorder.commit_step(step, llm=llm_context, append_message=False)

        logger.info(
            "summary_generated",
            tokens=step.metrics.total_tokens if step.metrics else 0,
        )


__all__ = ["ExecutionTerminationRuntime"]
