"""Transport serialization for observability-domain models."""

from typing import Any

from agiwo.observability.trace import Span, Trace
from agiwo.utils.serialization import serialize_enum_value, serialize_optional_datetime


def serialize_span_payload(span: Span) -> dict[str, Any]:
    return {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "parent_span_id": span.parent_span_id,
        "kind": serialize_enum_value(span.kind),
        "name": span.name,
        "start_time": serialize_optional_datetime(span.start_time),
        "end_time": serialize_optional_datetime(span.end_time),
        "duration_ms": span.duration_ms,
        "status": serialize_enum_value(span.status),
        "error_message": span.error_message,
        "depth": span.depth,
        "attributes": span.attributes or {},
        "input_preview": span.input_preview,
        "output_preview": span.output_preview,
        "metrics": span.metrics or {},
        "llm_details": span.llm_details,
        "tool_details": span.tool_details,
        "run_id": span.run_id,
        "step_id": span.step_id,
    }


def serialize_trace_base_payload(trace: Trace) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "agent_id": trace.agent_id,
        "session_id": trace.session_id,
        "user_id": trace.user_id,
        "start_time": serialize_optional_datetime(trace.start_time),
        "end_time": serialize_optional_datetime(trace.end_time),
        "duration_ms": trace.duration_ms,
        "status": serialize_enum_value(trace.status),
        "root_span_id": trace.root_span_id,
        "total_tokens": trace.total_tokens,
        "total_input_tokens": trace.total_input_tokens,
        "total_output_tokens": trace.total_output_tokens,
        "total_llm_calls": trace.total_llm_calls,
        "total_tool_calls": trace.total_tool_calls,
        "total_cache_read_tokens": trace.total_cache_read_tokens,
        "total_cache_creation_tokens": trace.total_cache_creation_tokens,
        "total_token_cost": trace.total_token_cost,
        "max_depth": trace.max_depth,
        "input_query": trace.input_query,
        "final_output": trace.final_output,
    }


def serialize_trace_payload(
    trace: Trace,
    *,
    include_spans: bool = True,
) -> dict[str, Any]:
    payload = serialize_trace_base_payload(trace)
    if include_spans:
        payload["spans"] = [serialize_span_payload(span) for span in (trace.spans or [])]
    return payload


__all__ = [
    "serialize_span_payload",
    "serialize_trace_base_payload",
    "serialize_trace_payload",
]
