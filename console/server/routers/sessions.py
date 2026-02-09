"""
Sessions and Runs API router.
"""

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Query

from server.dependencies import get_storage_manager
from server.schemas import RunResponse, SessionSummary, StepResponse

router = APIRouter(prefix="/api", tags=["sessions"])


def _step_to_response(step: Any) -> StepResponse:
    """Convert StepRecord dataclass to StepResponse."""
    created_at = step.created_at
    if created_at is not None and hasattr(created_at, "isoformat"):
        created_at = created_at.isoformat()
    elif created_at is not None:
        created_at = str(created_at)

    return StepResponse(
        id=step.id,
        session_id=step.session_id,
        run_id=step.run_id,
        sequence=step.sequence,
        role=step.role.value if hasattr(step.role, "value") else str(step.role),
        agent_id=step.agent_id,
        content=step.content,
        content_for_user=step.content_for_user,
        reasoning_content=step.reasoning_content,
        tool_calls=step.tool_calls,
        tool_call_id=step.tool_call_id,
        name=step.name,
        metrics=asdict(step.metrics) if step.metrics else None,
        created_at=created_at,
        parent_run_id=step.parent_run_id,
        depth=step.depth,
    )


def _run_to_response(run: Any) -> RunResponse:
    """Convert Run dataclass to RunResponse."""
    return RunResponse(
        id=run.id,
        agent_id=run.agent_id,
        session_id=run.session_id,
        user_id=run.user_id,
        user_input=run.user_input,
        status=run.status.value if hasattr(run.status, "value") else str(run.status),
        response_content=run.response_content,
        metrics=asdict(run.metrics) if run.metrics else None,
        created_at=run.created_at.isoformat() if hasattr(run.created_at, "isoformat") else str(run.created_at) if run.created_at else None,
        updated_at=run.updated_at.isoformat() if hasattr(run.updated_at, "isoformat") else str(run.updated_at) if run.updated_at else None,
        parent_run_id=run.parent_run_id,
    )


@router.get("/runs", response_model=list[RunResponse])
async def list_runs(
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[RunResponse]:
    """List all runs with optional filtering."""
    storage = get_storage_manager().run_step_storage
    runs = await storage.list_runs(
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        offset=offset,
    )
    return [_run_to_response(r) for r in runs]


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str) -> RunResponse:
    """Get a single run by ID."""
    storage = get_storage_manager().run_step_storage
    run = await storage.get_run(run_id)
    if run is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_response(run)


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[SessionSummary]:
    """
    List sessions by aggregating runs.

    Groups runs by session_id and returns summary info.
    """
    storage = get_storage_manager().run_step_storage
    runs = await storage.list_runs(limit=500, offset=0)

    session_map: dict[str, dict[str, Any]] = {}
    for run in runs:
        sid = run.session_id
        if sid not in session_map:
            session_map[sid] = {
                "session_id": sid,
                "agent_id": run.agent_id,
                "runs": [],
                "created_at": run.created_at,
                "updated_at": run.updated_at,
            }
        session_map[sid]["runs"].append(run)
        if run.updated_at and (
            session_map[sid]["updated_at"] is None
            or run.updated_at > session_map[sid]["updated_at"]
        ):
            session_map[sid]["updated_at"] = run.updated_at

    sessions = sorted(
        session_map.values(),
        key=lambda s: s["updated_at"] or "",
        reverse=True,
    )
    page = sessions[offset : offset + limit]

    results = []
    for s in page:
        runs_list = s["runs"]
        last_run = runs_list[0] if runs_list else None
        user_input_str = None
        if last_run and last_run.user_input:
            user_input_str = (
                last_run.user_input
                if isinstance(last_run.user_input, str)
                else str(last_run.user_input)
            )

        results.append(
            SessionSummary(
                session_id=s["session_id"],
                agent_id=s["agent_id"],
                last_user_input=user_input_str[:200] if user_input_str else None,
                last_response=last_run.response_content[:200] if last_run and last_run.response_content else None,
                run_count=len(runs_list),
                created_at=s["created_at"].isoformat() if hasattr(s["created_at"], "isoformat") else str(s["created_at"]) if s["created_at"] else None,
                updated_at=s["updated_at"].isoformat() if hasattr(s["updated_at"], "isoformat") else str(s["updated_at"]) if s["updated_at"] else None,
            )
        )

    return results


@router.get("/sessions/{session_id}/steps", response_model=list[StepResponse])
async def get_session_steps(
    session_id: str,
    agent_id: str | None = None,
    limit: int = Query(default=1000, ge=1, le=5000),
) -> list[StepResponse]:
    """Get all steps for a session."""
    storage = get_storage_manager().run_step_storage
    steps = await storage.get_steps(
        session_id=session_id,
        agent_id=agent_id,
        limit=limit,
    )
    return [_step_to_response(s) for s in steps]
