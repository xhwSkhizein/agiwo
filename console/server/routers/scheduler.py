"""
Scheduler Agent States API router.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from agiwo.scheduler.models import AgentState, AgentStateStatus

from server.dependencies import get_storage_manager
from server.schemas import (
    AgentStateListItem,
    AgentStateResponse,
    SchedulerStatsResponse,
    WakeConditionResponse,
)

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


def _wake_condition_to_response(wc: Any) -> WakeConditionResponse | None:
    if wc is None:
        return None
    return WakeConditionResponse(
        type=wc.type.value,
        wait_for=wc.wait_for,
        wait_mode=wc.wait_mode.value,
        completed_ids=wc.completed_ids,
        time_value=wc.time_value,
        time_unit=wc.time_unit.value if wc.time_unit else None,
        wakeup_at=wc.wakeup_at.isoformat() if wc.wakeup_at else None,
        submitted_task=wc.submitted_task,
        timeout_at=wc.timeout_at.isoformat() if wc.timeout_at else None,
    )


def _fmt_dt(dt: Any) -> str | None:
    if dt is None:
        return None
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


def _state_to_list_item(state: AgentState) -> AgentStateListItem:
    return AgentStateListItem(
        id=state.id,
        status=state.status.value,
        task=state.task[:200] if state.task else "",
        parent_id=state.parent_id,
        wake_condition=_wake_condition_to_response(state.wake_condition),
        result_summary=state.result_summary[:200] if state.result_summary else None,
        is_persistent=state.is_persistent,
        depth=state.depth,
        wake_count=state.wake_count,
        created_at=_fmt_dt(state.created_at),
        updated_at=_fmt_dt(state.updated_at),
    )


def _state_to_response(state: AgentState) -> AgentStateResponse:
    return AgentStateResponse(
        id=state.id,
        session_id=state.session_id,
        status=state.status.value,
        task=state.task,
        parent_id=state.parent_id,
        config_overrides=state.config_overrides,
        wake_condition=_wake_condition_to_response(state.wake_condition),
        result_summary=state.result_summary,
        signal_propagated=state.signal_propagated,
        is_persistent=state.is_persistent,
        depth=state.depth,
        wake_count=state.wake_count,
        created_at=_fmt_dt(state.created_at),
        updated_at=_fmt_dt(state.updated_at),
    )


@router.get("/states", response_model=list[AgentStateListItem])
async def list_agent_states(
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[AgentStateListItem]:
    """List all agent states with optional status filter."""
    storage = get_storage_manager().agent_state_storage
    status_enum = None
    if status is not None:
        try:
            status_enum = AgentStateStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    states = await storage.list_all(status=status_enum, limit=limit, offset=offset)
    return [_state_to_list_item(s) for s in states]


@router.get("/states/{state_id}", response_model=AgentStateResponse)
async def get_agent_state(state_id: str) -> AgentStateResponse:
    """Get a single agent state by ID."""
    storage = get_storage_manager().agent_state_storage
    state = await storage.get_state(state_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Agent state not found")
    return _state_to_response(state)


@router.get("/states/{state_id}/children", response_model=list[AgentStateListItem])
async def get_children(state_id: str) -> list[AgentStateListItem]:
    """Get child agent states for a given parent state."""
    storage = get_storage_manager().agent_state_storage
    children = await storage.get_states_by_parent(state_id)
    children.sort(key=lambda s: s.created_at, reverse=True)
    return [_state_to_list_item(s) for s in children]


@router.get("/stats", response_model=SchedulerStatsResponse)
async def get_stats() -> SchedulerStatsResponse:
    """Get scheduler statistics — count of states by status."""
    storage = get_storage_manager().agent_state_storage
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
