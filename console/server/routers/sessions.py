"""Sessions and Runs query router (list/detail/steps; control is sessions_lifecycle)."""

from fastapi import APIRouter, HTTPException, Query

from server.dependencies import (
    ConsoleRuntimeDep,
    get_run_query_service,
    get_session_view_service,
)
from server.response_serialization import (
    run_response_from_sdk,
    session_detail_response_from_record,
    session_summary_response_from_record,
    step_response_from_sdk,
)
from server.models.view import (
    PageResponse,
    RunResponse,
    SessionDetailResponse,
    SessionSummaryResponse,
    StepResponse,
)

router = APIRouter(prefix="/api", tags=["sessions"])
_STEPS_MAX_LIMIT = 5000


@router.get("/runs", response_model=PageResponse[RunResponse])
async def list_runs(
    runtime: ConsoleRuntimeDep,
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> PageResponse[RunResponse]:
    """List all runs with optional filtering."""
    page = await get_run_query_service(runtime).list_runs(
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        offset=offset,
    )
    return PageResponse(
        items=[run_response_from_sdk(r) for r in page.items],
        limit=page.limit,
        offset=page.offset,
        has_more=page.has_more,
        total=page.total,
    )


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, runtime: ConsoleRuntimeDep) -> RunResponse:
    """Get a single run by ID."""
    run = await get_run_query_service(runtime).get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_response_from_sdk(run)


@router.get("/sessions", response_model=PageResponse[SessionSummaryResponse])
async def list_sessions(
    runtime: ConsoleRuntimeDep,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_archived: bool = Query(default=False),
) -> PageResponse[SessionSummaryResponse]:
    """List sessions from the session store with lightweight enrichment."""
    page = await get_session_view_service(runtime).list_sessions(
        limit=limit, offset=offset, include_archived=include_archived
    )
    return PageResponse(
        items=[session_summary_response_from_record(item) for item in page.items],
        limit=page.limit,
        offset=page.offset,
        has_more=page.has_more,
        total=page.total,
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session_detail(
    session_id: str,
    runtime: ConsoleRuntimeDep,
) -> SessionDetailResponse:
    detail = await get_session_view_service(runtime).get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_detail_response_from_record(detail)


@router.get("/sessions/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(
    session_id: str,
    runtime: ConsoleRuntimeDep,
) -> SessionSummaryResponse:
    """Get full aggregated metrics for one session."""
    detail = await get_session_view_service(runtime).get_session_detail(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_summary_response_from_record(detail.summary)


@router.get("/sessions/{session_id}/steps", response_model=PageResponse[StepResponse])
async def get_session_steps(
    session_id: str,
    runtime: ConsoleRuntimeDep,
    start_seq: int | None = Query(default=None, ge=1),
    end_seq: int | None = Query(default=None, ge=1),
    run_id: str | None = None,
    agent_id: str | None = None,
    limit: int = Query(default=1000, ge=1, le=_STEPS_MAX_LIMIT),
    order: str = Query(default="asc", pattern="^(asc|desc)$"),
) -> PageResponse[StepResponse]:
    """Get all steps for a session."""
    page = await get_run_query_service(runtime).list_session_steps(
        session_id,
        start_seq=start_seq,
        end_seq=end_seq,
        run_id=run_id,
        agent_id=agent_id,
        limit=limit,
        order=order,
    )
    return PageResponse(
        items=[step_response_from_sdk(s) for s in page.items],
        limit=page.limit,
        offset=page.offset,
        has_more=page.has_more,
        total=page.total,
    )
