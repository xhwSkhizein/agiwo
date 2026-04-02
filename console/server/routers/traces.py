"""
Traces API router.
"""

from typing import Any

from fastapi import APIRouter, Query, HTTPException

from server.dependencies import ConsoleRuntimeDep
from server.response_serialization import (
    trace_list_item_from_sdk,
    trace_response_from_sdk,
)
from server.models.view import PageResponse, TraceListItem, TraceResponse

router = APIRouter(prefix="/api/traces", tags=["traces"])


@router.get("", response_model=PageResponse[TraceListItem])
async def list_traces(
    runtime: ConsoleRuntimeDep,
    agent_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> PageResponse[TraceListItem]:
    """Query traces with optional filters."""
    store = runtime.trace_storage

    query: dict[str, Any] = {"limit": limit + 1, "offset": offset}
    if agent_id:
        query["agent_id"] = agent_id
    if session_id:
        query["session_id"] = session_id
    if user_id:
        query["user_id"] = user_id
    if status:
        query["status"] = status

    traces = await store.query_traces(query)
    has_more = len(traces) > limit
    page = traces[:limit]
    return PageResponse(
        items=[trace_list_item_from_sdk(t) for t in page],
        limit=limit,
        offset=offset,
        has_more=has_more,
        total=None,
    )


@router.get("/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str, runtime: ConsoleRuntimeDep) -> TraceResponse:
    """Get a single trace with full span tree."""
    store = runtime.trace_storage
    trace = await store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace_response_from_sdk(trace)
