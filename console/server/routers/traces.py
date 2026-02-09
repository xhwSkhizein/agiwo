"""
Traces API router.
"""

import asyncio
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, HTTPException
from sse_starlette.sse import EventSourceResponse

from server.dependencies import get_storage_manager
from server.schemas import TraceListItem, TraceResponse, SpanResponse

router = APIRouter(prefix="/api/traces", tags=["traces"])


def _safe_isoformat(val: Any) -> str | None:
    if val is None:
        return None
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def _span_to_response(span: Any) -> SpanResponse:
    return SpanResponse(
        span_id=span.span_id,
        trace_id=span.trace_id,
        parent_span_id=span.parent_span_id,
        kind=span.kind.value if hasattr(span.kind, "value") else str(span.kind),
        name=span.name,
        start_time=_safe_isoformat(span.start_time),
        end_time=_safe_isoformat(span.end_time),
        duration_ms=span.duration_ms,
        status=span.status.value if hasattr(span.status, "value") else str(span.status),
        error_message=span.error_message,
        depth=span.depth,
        attributes=span.attributes or {},
        input_preview=span.input_preview,
        output_preview=span.output_preview,
        metrics=span.metrics or {},
        llm_details=span.llm_details,
        tool_details=span.tool_details,
        run_id=span.run_id,
        step_id=span.step_id,
    )


def _trace_to_list_item(trace: Any) -> TraceListItem:
    return TraceListItem(
        trace_id=trace.trace_id,
        agent_id=trace.agent_id,
        session_id=trace.session_id,
        user_id=trace.user_id,
        start_time=_safe_isoformat(trace.start_time),
        duration_ms=trace.duration_ms,
        status=trace.status.value if hasattr(trace.status, "value") else str(trace.status),
        total_tokens=trace.total_tokens,
        total_llm_calls=trace.total_llm_calls,
        total_tool_calls=trace.total_tool_calls,
        input_query=trace.input_query[:200] if trace.input_query else None,
        final_output=trace.final_output[:200] if trace.final_output else None,
    )


def _trace_to_response(trace: Any) -> TraceResponse:
    return TraceResponse(
        trace_id=trace.trace_id,
        agent_id=trace.agent_id,
        session_id=trace.session_id,
        user_id=trace.user_id,
        start_time=_safe_isoformat(trace.start_time),
        end_time=_safe_isoformat(trace.end_time),
        duration_ms=trace.duration_ms,
        status=trace.status.value if hasattr(trace.status, "value") else str(trace.status),
        root_span_id=trace.root_span_id,
        total_tokens=trace.total_tokens,
        total_llm_calls=trace.total_llm_calls,
        total_tool_calls=trace.total_tool_calls,
        total_cache_read_tokens=trace.total_cache_read_tokens,
        total_cache_creation_tokens=trace.total_cache_creation_tokens,
        max_depth=trace.max_depth,
        input_query=trace.input_query,
        final_output=trace.final_output,
        spans=[_span_to_response(s) for s in (trace.spans or [])],
    )


@router.get("", response_model=list[TraceListItem])
async def list_traces(
    agent_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[TraceListItem]:
    """Query traces with optional filters."""
    store = get_storage_manager().trace_storage

    query: dict[str, Any] = {"limit": limit, "offset": offset}
    if agent_id:
        query["agent_id"] = agent_id
    if session_id:
        query["session_id"] = session_id
    if user_id:
        query["user_id"] = user_id
    if status:
        query["status"] = status

    traces = await store.query_traces(query)
    return [_trace_to_list_item(t) for t in traces]


@router.get("/stream")
async def stream_traces() -> EventSourceResponse:
    """SSE endpoint for real-time trace updates."""
    store = get_storage_manager().trace_storage

    queue = store.subscribe()

    async def event_generator():
        try:
            while True:
                try:
                    trace = await asyncio.wait_for(queue.get(), timeout=30.0)
                    item = _trace_to_list_item(trace)
                    yield {"event": "trace", "data": item.model_dump_json()}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        except asyncio.CancelledError:
            pass
        finally:
            store.unsubscribe(queue)

    return EventSourceResponse(event_generator())


@router.get("/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str) -> TraceResponse:
    """Get a single trace with full span tree."""
    store = get_storage_manager().trace_storage
    trace = await store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return _trace_to_response(trace)
