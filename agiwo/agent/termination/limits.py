"""Termination-limit checks for run-loop stop conditions."""

import time

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.run import TerminationReason
from agiwo.agent.models.step import LLMCallContext, StepRecord
from agiwo.agent.runtime.context import RunContext
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def check_non_recoverable_limits(
    state: RunContext,
    options: AgentOptions,
    current_step: int,
) -> TerminationReason | None:
    if current_step >= options.max_steps:
        logger.warning(
            "limit_hit_max_steps",
            current_step=current_step,
            max_steps=options.max_steps,
            run_id=state.run_id,
        )
        return TerminationReason.MAX_STEPS

    if options.run_timeout and state.elapsed > options.run_timeout:
        logger.warning(
            "limit_hit_timeout",
            elapsed=state.elapsed,
            run_timeout=options.run_timeout,
            run_id=state.run_id,
        )
        return TerminationReason.TIMEOUT

    if state.timeout_at and time.time() >= state.timeout_at:
        logger.warning(
            "limit_hit_context_timeout",
            timeout_at=state.timeout_at,
            run_id=state.run_id,
        )
        return TerminationReason.TIMEOUT

    return None


def check_post_llm_limits(
    state: RunContext,
    step: StepRecord,
    llm_context: LLMCallContext,
    *,
    options: AgentOptions,
    max_input_tokens_per_call: int,
) -> TerminationReason | None:
    input_tokens = step.metrics.input_tokens if step.metrics else 0
    if input_tokens and input_tokens > max_input_tokens_per_call:
        logger.warning(
            "limit_hit_max_input_tokens_per_call",
            input_tokens=input_tokens,
            max_input_tokens_per_call=max_input_tokens_per_call,
            run_id=state.run_id,
        )
        return TerminationReason.MAX_INPUT_TOKENS_PER_CALL

    if (
        options.max_run_cost is not None
        and state.ledger.token_cost >= options.max_run_cost
    ):
        logger.warning(
            "limit_hit_max_run_cost",
            token_cost=state.ledger.token_cost,
            max_run_cost=options.max_run_cost,
            run_id=state.run_id,
        )
        return TerminationReason.MAX_RUN_COST

    finish_reason = llm_context.finish_reason
    if finish_reason and finish_reason.strip().lower() in {"length", "max_tokens"}:
        logger.warning(
            "limit_hit_max_output_tokens",
            finish_reason=finish_reason,
            run_id=state.run_id,
        )
        return TerminationReason.MAX_OUTPUT_TOKENS

    return None


__all__ = [
    "check_non_recoverable_limits",
    "check_post_llm_limits",
]
