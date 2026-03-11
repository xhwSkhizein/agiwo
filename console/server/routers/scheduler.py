"""Scheduler agent-state query router."""

from fastapi import APIRouter, HTTPException, Query

from agiwo.scheduler.models import AgentStateStatus

from server.dependencies import ConsoleRuntimeDep
from server.response_serialization import (
    state_to_list_item as _state_to_list_item,
    state_to_response as _state_to_response,
)
from server.schemas import (
    AgentStateListItem,
    AgentStateResponse,
    SchedulerStatsResponse,
)
from server.services.scheduler_state_metrics import build_metrics_by_state

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


@router.get("/states", response_model=list[AgentStateListItem])
async def list_agent_states(
    runtime: ConsoleRuntimeDep,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[AgentStateListItem]:
    """List all agent states with optional status filter."""
    scheduler = runtime.scheduler
    assert scheduler is not None
    storage = scheduler.store
    status_enum = None
    if status is not None:
        try:
            status_enum = AgentStateStatus(status)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}") from exc
    states = await storage.list_all(status=status_enum, limit=limit, offset=offset)
    metrics_by_state = await build_metrics_by_state(states, runtime)
    return [
        _state_to_list_item(
            s,
            metrics_by_state.get((s.session_id, s.id)),
        )
        for s in states
    ]


@router.get("/states/{state_id}", response_model=AgentStateResponse)
async def get_agent_state(
    state_id: str,
    runtime: ConsoleRuntimeDep,
) -> AgentStateResponse:
    """Get a single agent state by ID."""
    scheduler = runtime.scheduler
    assert scheduler is not None
    storage = scheduler.store
    state = await storage.get_state(state_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Agent state not found")
    metrics_by_state = await build_metrics_by_state([state], runtime)
    return _state_to_response(
        state,
        metrics_by_state.get((state.session_id, state.id)),
    )


@router.get("/states/{state_id}/children", response_model=list[AgentStateListItem])
async def get_children(
    state_id: str,
    runtime: ConsoleRuntimeDep,
) -> list[AgentStateListItem]:
    """Get child agent states for a given parent state."""
    scheduler = runtime.scheduler
    assert scheduler is not None
    storage = scheduler.store
    children = await storage.get_states_by_parent(state_id)
    children.sort(key=lambda s: s.created_at, reverse=True)
    metrics_by_state = await build_metrics_by_state(children, runtime)
    return [
        _state_to_list_item(
            s,
            metrics_by_state.get((s.session_id, s.id)),
        )
        for s in children
    ]


@router.get("/stats", response_model=SchedulerStatsResponse)
async def get_stats(runtime: ConsoleRuntimeDep) -> SchedulerStatsResponse:
    """Get scheduler statistics — count of states by status."""
    scheduler = runtime.scheduler
    assert scheduler is not None
    storage = scheduler.store
    all_states = await storage.list_all(limit=10000)
    counts: dict[str, int] = {
        "pending": 0,
        "running": 0,
        "sleeping": 0,
        "completed": 0,
        "failed": 0,
    }
    for s in all_states:
        key = s.status.value
        if key in counts:
            counts[key] += 1
    return SchedulerStatsResponse(
        total=len(all_states),
        **counts,
    )
