"""Sessions and Runs API router."""

from fastapi import APIRouter, HTTPException, Query

from server.dependencies import ConsoleRuntimeDep
from server.domain.sessions import session_aggregate_to_summary_data
from server.domain.sessions import SessionSummaryData
from server.schemas import RunResponse, StepResponse
from server.services.metrics import collect_session_aggregates

router = APIRouter(prefix="/api", tags=["sessions"])


@router.get("/runs", response_model=list[RunResponse])
async def list_runs(
    runtime: ConsoleRuntimeDep,
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[RunResponse]:
    """List all runs with optional filtering."""
    storage = runtime.run_step_storage
    runs = await storage.list_runs(
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        offset=offset,
    )
    return [RunResponse.from_sdk(r) for r in runs]


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str, runtime: ConsoleRuntimeDep) -> RunResponse:
    """Get a single run by ID."""
    storage = runtime.run_step_storage
    run = await storage.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse.from_sdk(run)


@router.get("/sessions", response_model=list[SessionSummaryData])
async def list_sessions(
    runtime: ConsoleRuntimeDep,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[SessionSummaryData]:
    """List sessions by aggregating runs.

    .. note:: Pagination is applied in-memory after fetching all aggregates.
       See ``collect_session_aggregates`` for performance considerations.
    """
    storage = runtime.run_step_storage
    sessions = await collect_session_aggregates(storage)
    page = sessions[offset : offset + limit]
    return [session_aggregate_to_summary_data(session) for session in page]


@router.get("/sessions/{session_id}/summary", response_model=SessionSummaryData)
async def get_session_summary(
    session_id: str,
    runtime: ConsoleRuntimeDep,
) -> SessionSummaryData:
    """Get full aggregated metrics for one session."""
    storage = runtime.run_step_storage
    sessions = await collect_session_aggregates(storage, session_id=session_id)
    if not sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_aggregate_to_summary_data(sessions[0])


@router.get("/sessions/{session_id}/steps", response_model=list[StepResponse])
async def get_session_steps(
    session_id: str,
    runtime: ConsoleRuntimeDep,
    agent_id: str | None = None,
    limit: int = Query(default=1000, ge=1, le=5000),
) -> list[StepResponse]:
    """Get all steps for a session."""
    storage = runtime.run_step_storage
    steps = await storage.get_steps(
        session_id=session_id,
        agent_id=agent_id,
        limit=limit,
    )
    return [StepResponse.from_sdk(s) for s in steps]
