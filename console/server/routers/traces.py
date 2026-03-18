"""
Traces API router.
"""

from typing import Any

from fastapi import APIRouter, Query, HTTPException

from server.dependencies import ConsoleRuntimeDep
from server.response_serialization import trace_to_list_item, trace_to_response
from server.schemas import TraceListItem, TraceResponse

router = APIRouter(prefix="/api/traces", tags=["traces"])


@router.get("", response_model=list[TraceListItem])
async def list_traces(
    runtime: ConsoleRuntimeDep,
    agent_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[TraceListItem]:
    """Query traces with optional filters."""
    store = runtime.trace_storage

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
    return [trace_to_list_item(t) for t in traces]


@router.get("/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str, runtime: ConsoleRuntimeDep) -> TraceResponse:
    """Get a single trace with full span tree."""
    store = runtime.trace_storage
    trace = await store.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace_to_response(trace)
