"""Mutable run-state helpers for messages, counters, and termination state."""

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from agiwo.agent.models.compact import CompactMetadata
from agiwo.agent.models.run import TerminationReason
from agiwo.agent.models.step import StepRecord
from agiwo.agent.runtime.context import RunContext


def replace_messages(state: RunContext, messages: Sequence[dict[str, Any]]) -> None:
    state.ledger.messages = copy.deepcopy(list(messages))


def append_message(state: RunContext, message: Mapping[str, Any]) -> None:
    state.ledger.messages.append(copy.deepcopy(dict(message)))


def set_tool_schemas(
    state: RunContext,
    tool_schemas: Sequence[dict[str, Any]] | None,
) -> None:
    state.ledger.tool_schemas = (
        copy.deepcopy(list(tool_schemas)) if tool_schemas is not None else None
    )


def record_compaction_metadata(
    state: RunContext,
    metadata: CompactMetadata | None,
) -> None:
    state.ledger.last_compact_metadata = metadata


def set_termination_reason(
    state: RunContext,
    reason: TerminationReason,
) -> None:
    state.ledger.termination_reason = reason


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
        state.ledger.messages.append(copy.deepcopy(step.to_message()))


__all__ = [
    "append_message",
    "record_compaction_metadata",
    "replace_messages",
    "set_termination_reason",
    "set_tool_schemas",
    "track_step_state",
]
