"""Shared mutation of RunContext step-tracking state."""

from agiwo.agent.run_state import RunContext
from agiwo.agent.types import StepRecord, step_to_message


def _track_step_metrics(state: RunContext, step: StepRecord) -> None:
    ledger = state.ledger
    metrics = step.metrics
    if metrics is None:
        return
    if metrics.token_cost is not None:
        ledger.token_cost += metrics.token_cost
    if metrics.total_tokens is not None:
        ledger.total_tokens += metrics.total_tokens
    if metrics.input_tokens is not None:
        ledger.input_tokens += metrics.input_tokens
    if metrics.output_tokens is not None:
        ledger.output_tokens += metrics.output_tokens
    if metrics.cache_read_tokens is not None:
        ledger.cache_read_tokens += metrics.cache_read_tokens
    if metrics.cache_creation_tokens is not None:
        ledger.cache_creation_tokens += metrics.cache_creation_tokens


def _track_assistant_step(state: RunContext, step: StepRecord) -> None:
    ledger = state.ledger
    if not step.is_assistant_step():
        return
    ledger.assistant_steps_count += 1
    if step.content is not None:
        ledger.response_content = step.content
    if step.tool_calls:
        ledger.tool_calls_count += len(step.tool_calls)


def track_step_state(
    state: RunContext,
    step: StepRecord,
    *,
    append_message: bool = True,
) -> None:
    ledger = state.ledger
    ledger.steps_count += 1
    _track_step_metrics(state, step)
    _track_assistant_step(state, step)
    if append_message:
        ledger.messages.append(step_to_message(step))
