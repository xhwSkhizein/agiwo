"""
LLM caller — streaming LLM calls + step building.

Consolidates the LLM-call and step-building logic into one module.
"""

import asyncio
import time
from datetime import datetime, timezone

from agiwo.agent.models.step import (
    LLMCallContext,
    StepDelta,
    StepMetrics,
    StepRecord,
)
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.models.stream import StepDeltaEvent
from agiwo.llm.base import Model, StreamChunk
from agiwo.llm.event_normalizer import normalize_usage_metrics
from agiwo.llm.usage_resolver import ModelUsageEstimator, UsageEstimate
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

_CHUNK_TIMEOUT_SECONDS = 120


async def stream_assistant_step(
    model: Model,
    state: RunContext,
    abort_signal: AbortSignal | None,
    *,
    messages: list[dict] | None = None,
    tools: list[dict] | None = ...,
) -> tuple[StepRecord, LLMCallContext]:
    """Stream call LLM and build Step."""
    messages = messages if messages is not None else state.copy_messages()
    tools_resolved = state.copy_tool_schemas() if tools is ... else tools

    metrics_resolver = ModelUsageEstimator(model)
    request_estimate = metrics_resolver.estimate_request(messages, tools_resolved)

    llm_context = LLMCallContext(
        messages=list(messages),
        tools=list(tools_resolved) if tools_resolved else None,
        request_params=_get_request_params(model),
    )

    step_start_time = time.time()
    sequence = await state.session_runtime.allocate_sequence()
    step = StepRecord.assistant(
        state,
        sequence=sequence,
        content="",
        tool_calls=None,
        metrics=StepMetrics(start_at=datetime.now(timezone.utc)),
    )
    if step.metrics is not None:
        step.metrics.model_name = getattr(model, "model_name", None)
        step.metrics.provider = getattr(model, "provider", None)

    first_token_received = False
    finish_reason: str | None = None
    tool_calls_acc: dict[int, dict] = {}

    stream = model.arun_stream(messages, tools=tools_resolved)
    while True:
        try:
            async with asyncio.timeout(_CHUNK_TIMEOUT_SECONDS):
                chunk = await stream.__anext__()
        except StopAsyncIteration:
            break
        except TimeoutError as exc:
            raise TimeoutError(
                f"LLM stream stalled: no chunk received for {_CHUNK_TIMEOUT_SECONDS}s"
            ) from exc

        _check_abort(abort_signal)
        delta, has_content, chunk_finish_reason = _apply_chunk_to_step(
            step=step,
            chunk=chunk,
            tool_calls_acc=tool_calls_acc,
        )

        if has_content and not first_token_received:
            first_token_received = True
            if step.metrics:
                step.metrics.first_token_latency_ms = (
                    time.time() - step_start_time
                ) * 1000

        if chunk_finish_reason:
            finish_reason = chunk_finish_reason

        if has_content or delta.usage:
            await state.session_runtime.publish(
                StepDeltaEvent.from_context(state, step_id=step.id, delta=delta),
            )

    step.content = step.content or None
    step.reasoning_content = step.reasoning_content or None
    step.tool_calls = _finalize_tool_calls(tool_calls_acc) or None

    if step.metrics:
        step.metrics.end_at = datetime.now(timezone.utc)
        step.metrics.duration_ms = (time.time() - step_start_time) * 1000

    _resolve_step_metrics(step, metrics_resolver, request_estimate)
    llm_context.finish_reason = finish_reason
    return step, llm_context


def _accumulate_tool_calls(
    tool_calls_acc: dict[int, dict], delta_calls: list[dict]
) -> None:
    """Accumulate streaming tool calls."""
    for tc in delta_calls:
        idx = tc.get("index", 0)

        if idx not in tool_calls_acc:
            tool_calls_acc[idx] = {
                "id": None,
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }

        acc = tool_calls_acc[idx]

        if tc.get("id"):
            acc["id"] = tc["id"]

        if tc.get("type"):
            acc["type"] = tc["type"]

        if tc.get("function"):
            fn = tc["function"]
            if fn.get("name"):
                acc["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                acc["function"]["arguments"] += fn["arguments"]


def _finalize_tool_calls(tool_calls_acc: dict[int, dict]) -> list[dict]:
    """Finalize accumulated tool calls."""
    return [call for call in tool_calls_acc.values() if call["id"] is not None]


def _apply_chunk_to_step(
    *,
    step: StepRecord,
    chunk: StreamChunk,
    tool_calls_acc: dict[int, dict],
) -> tuple[StepDelta, bool, str | None]:
    delta = StepDelta()
    has_content = bool(chunk.content or chunk.reasoning_content or chunk.tool_calls)

    if chunk.content:
        step.content = (step.content or "") + chunk.content
        delta.content = chunk.content

    if chunk.reasoning_content:
        step.reasoning_content = (
            step.reasoning_content or ""
        ) + chunk.reasoning_content
        delta.reasoning_content = chunk.reasoning_content

    if chunk.tool_calls:
        _accumulate_tool_calls(tool_calls_acc, chunk.tool_calls)
        delta.tool_calls = chunk.tool_calls

    if chunk.usage and step.metrics:
        normalized = normalize_usage_metrics(chunk.usage)
        step.metrics.input_tokens = normalized["input_tokens"]
        step.metrics.output_tokens = normalized["output_tokens"]
        step.metrics.total_tokens = normalized["total_tokens"]
        step.metrics.cache_read_tokens = normalized["cache_read_tokens"]
        step.metrics.cache_creation_tokens = normalized["cache_creation_tokens"]
        step.metrics.usage_source = "provider"
        delta.usage = normalized

    return delta, has_content, chunk.finish_reason


def _resolve_step_metrics(
    step: StepRecord,
    metrics_resolver: ModelUsageEstimator,
    request_estimate: UsageEstimate | None,
) -> None:
    """Fill missing metrics on StepRecord and compute cost."""
    if step.metrics is None:
        return

    had_provider_usage = any(
        value is not None
        for value in (
            step.metrics.input_tokens,
            step.metrics.output_tokens,
            step.metrics.total_tokens,
            step.metrics.cache_read_tokens,
            step.metrics.cache_creation_tokens,
        )
    )

    estimated_output = metrics_resolver.estimate_assistant_output(
        content=step.content if isinstance(step.content, str) else None,
        reasoning_content=step.reasoning_content,
        tool_calls=step.tool_calls,
    )

    if step.metrics.input_tokens is None and request_estimate is not None:
        step.metrics.input_tokens = request_estimate.input_tokens
    if step.metrics.output_tokens is None:
        step.metrics.output_tokens = estimated_output
    if step.metrics.total_tokens is None:
        step.metrics.total_tokens = (step.metrics.input_tokens or 0) + (
            step.metrics.output_tokens or 0
        )
    if step.metrics.cache_read_tokens is None:
        step.metrics.cache_read_tokens = (
            request_estimate.cache_read_tokens if request_estimate else 0
        )
    if step.metrics.cache_creation_tokens is None:
        step.metrics.cache_creation_tokens = (
            request_estimate.cache_creation_tokens if request_estimate else 0
        )

    if had_provider_usage:
        if request_estimate is not None and (
            step.metrics.input_tokens == request_estimate.input_tokens
            or step.metrics.output_tokens == estimated_output
        ):
            step.metrics.usage_source = "mixed"
        else:
            step.metrics.usage_source = "provider"
    else:
        step.metrics.usage_source = "estimated"

    step.metrics.token_cost = metrics_resolver.compute_cost(
        input_tokens=step.metrics.input_tokens,
        output_tokens=step.metrics.output_tokens,
        cache_read_tokens=step.metrics.cache_read_tokens,
        cache_creation_tokens=step.metrics.cache_creation_tokens,
    )


def _get_request_params(model: Model) -> dict:
    """Get LLM request parameters."""
    params: dict = {
        "model_id": getattr(model, "id", None),
        "model_name": getattr(model, "name", None),
    }
    if hasattr(model, "temperature"):
        params["temperature"] = model.temperature
        params["max_output_tokens"] = model.max_output_tokens
        params["top_p"] = model.top_p
    return params


def _check_abort(abort_signal: AbortSignal | None) -> None:
    """Check abort signal."""
    if abort_signal and abort_signal.is_aborted():
        raise asyncio.CancelledError(abort_signal.reason)


__all__ = ["stream_assistant_step"]
