"""Traces API router."""

from fastapi import APIRouter, Query, HTTPException

from server.dependencies import ConsoleRuntimeDep, get_trace_query_service
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
    page = await get_trace_query_service(runtime).list_traces(
        agent_id=agent_id,
        session_id=session_id,
        user_id=user_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return PageResponse(
        items=[trace_list_item_from_sdk(t) for t in page.items],
        limit=limit,
        offset=offset,
        has_more=page.has_more,
        total=page.total,
    )


@router.get("/{trace_id}", response_model=TraceResponse)
async def get_trace(trace_id: str, runtime: ConsoleRuntimeDep) -> TraceResponse:
    """Get a single trace with full span tree."""
    trace = await get_trace_query_service(runtime).get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace_response_from_sdk(trace)
