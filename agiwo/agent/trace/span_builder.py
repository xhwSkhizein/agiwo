"""
Helpers for building agent trace spans from step events.
"""

import json
from datetime import datetime, timezone
from typing import Any

from agiwo.agent.runtime import LLMCallContext, StepRecord
from agiwo.observability.trace import Span, SpanKind, SpanStatus


def build_tool_step_span(
    *,
    trace_id: str,
    step: StepRecord,
    assistant_step_cache: dict[str, StepRecord],
    span_stack: dict[str, Span],
    current_span: Span | None,
) -> Span:
    start_time, end_time, duration_ms = extract_step_times(step)
    parent_span_id, _, parent_depth = resolve_parent_span(
        step, span_stack, current_span
    )
    tool_input_args = extract_tool_input_args(step, assistant_step_cache)
    tool_details = build_tool_details(step, tool_input_args)

    is_error = isinstance(step.content, str) and step.content.startswith("Error:")
    status = SpanStatus.ERROR if is_error else SpanStatus.OK
    error_message = step.content if is_error else None

    span = Span(
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        kind=SpanKind.TOOL_CALL,
        name=step.name or "tool",
        depth=parent_depth + 1,
        attributes={
            "tool_name": step.name,
            "tool_call_id": step.tool_call_id,
            "tool_id": step.tool_call_id,
        },
        tool_details=tool_details,
        step_id=step.id,
        run_id=step.run_id,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms or (step.metrics.duration_ms if step.metrics else None),
        status=status,
        error_message=error_message,
    )

    if step.metrics:
        span.metrics = {
            "tool.exec_time_ms": step.metrics.duration_ms,
            "duration_ms": span.duration_ms,
        }

    return span


def build_assistant_step_span(
    *,
    trace_id: str,
    step: StepRecord,
    llm: LLMCallContext | None,
    span_stack: dict[str, Span],
    current_span: Span | None,
    preview_length: int,
) -> Span:
    start_time, end_time, duration_ms = extract_step_times(step)
    parent_span_id, _, parent_depth = resolve_parent_span(
        step, span_stack, current_span
    )

    name = (
        step.metrics.model_name if step.metrics and step.metrics.model_name else "llm"
    )

    span = Span(
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        kind=SpanKind.LLM_CALL,
        name=name,
        depth=parent_depth + 1,
        attributes={
            "model_name": step.metrics.model_name if step.metrics else None,
            "provider": step.metrics.provider if step.metrics else None,
            "usage_source": step.metrics.usage_source if step.metrics else None,
            "has_tool_calls": bool(step.tool_calls),
        },
        step_id=step.id,
        run_id=step.run_id,
        start_time=start_time,
        end_time=end_time,
        duration_ms=duration_ms,
        status=SpanStatus.OK,
        output_preview=(
            step.content[:preview_length]
            if isinstance(step.content, str) and step.content
            else None
        ),
        llm_details=build_llm_details(step, llm),
    )

    if step.metrics:
        span.metrics = {
            "tokens.input": step.metrics.input_tokens,
            "tokens.output": step.metrics.output_tokens,
            "tokens.total": step.metrics.total_tokens,
            "tokens.cache_read": step.metrics.cache_read_tokens,
            "tokens.cache_creation": step.metrics.cache_creation_tokens,
            "token_cost": step.metrics.token_cost,
            "first_token_ms": step.metrics.first_token_latency_ms,
            "duration_ms": duration_ms,
            "model": step.metrics.model_name,
            "provider": step.metrics.provider,
            "usage_source": step.metrics.usage_source,
        }

    return span


def extract_tool_input_args(
    tool_step: StepRecord,
    assistant_step_cache: dict[str, StepRecord],
) -> dict[str, Any]:
    parsed_args: dict[str, Any] = {}
    if not tool_step.tool_call_id:
        return parsed_args

    assistant_step = assistant_step_cache.get(tool_step.tool_call_id)
    if not assistant_step or not assistant_step.tool_calls:
        return parsed_args

    for tool_call in assistant_step.tool_calls:
        if tool_call.get("id") != tool_step.tool_call_id:
            continue

        raw_args = tool_call.get("function", {}).get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {}
        elif isinstance(raw_args, dict):
            parsed_args = raw_args
        break

    return parsed_args


def extract_step_times(
    step: StepRecord,
) -> tuple[datetime | None, datetime | None, float | None]:
    start_time = (
        step.metrics.start_at if step.metrics and step.metrics.start_at else None
    ) or step.created_at
    end_time = step.metrics.end_at if step.metrics else None
    duration_ms = step.metrics.duration_ms if step.metrics else None
    return (
        normalize_time(start_time),
        normalize_time(end_time),
        duration_ms,
    )


def resolve_parent_span(
    step: StepRecord,
    span_stack: dict[str, Span],
    current_span: Span | None,
) -> tuple[str | None, Span | None, int]:
    parent = span_stack.get(step.run_id)
    if not parent and step.parent_run_id:
        parent = span_stack.get(step.parent_run_id)
    if not parent:
        parent = current_span

    parent_span_id = parent.span_id if parent else None
    parent_depth = parent.depth if parent else 0
    return parent_span_id, parent, parent_depth


def build_tool_details(
    tool_step: StepRecord,
    input_args: dict[str, Any],
) -> dict[str, Any]:
    """Build complete tool call details from Tool Step and input arguments."""
    is_error = isinstance(tool_step.content, str) and tool_step.content.startswith(
        "Error:"
    )

    details: dict[str, Any] = {
        "tool_name": tool_step.name,
        "tool_id": tool_step.tool_call_id,
        "tool_call_id": tool_step.tool_call_id,
        "input_args": input_args,
        "output": tool_step.content,
        "content_for_user": tool_step.content_for_user,
        "error": tool_step.content if is_error else None,
        "status": "error" if is_error else "completed",
    }

    if tool_step.metrics:
        details["metrics"] = {
            "duration_ms": tool_step.metrics.duration_ms,
        }

    return details


def build_llm_details(
    step: StepRecord,
    llm: LLMCallContext | None,
) -> dict[str, Any]:
    """Build complete LLM call details from Step and LLM context."""
    if not llm:
        return {}

    details: dict[str, Any] = {
        "request": llm.request_params or {},
        "messages": llm.messages,
        "tools": llm.tools,
        "response_content": step.content,
        "response_tool_calls": step.tool_calls,
        "finish_reason": llm.finish_reason,
        "status": "completed",
    }

    if step.metrics:
        details["metrics"] = {
            "duration_ms": step.metrics.duration_ms,
            "first_token_ms": step.metrics.first_token_latency_ms,
            "input_tokens": step.metrics.input_tokens,
            "output_tokens": step.metrics.output_tokens,
            "total_tokens": step.metrics.total_tokens,
            "cache_read_tokens": step.metrics.cache_read_tokens,
            "cache_creation_tokens": step.metrics.cache_creation_tokens,
            "usage_source": step.metrics.usage_source,
        }

    return details


def normalize_time(value: datetime | None) -> datetime | None:
    if value and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


__all__ = [
    "build_assistant_step_span",
    "build_tool_step_span",
    "build_llm_details",
    "build_tool_details",
    "extract_step_times",
    "normalize_time",
    "resolve_parent_span",
]
