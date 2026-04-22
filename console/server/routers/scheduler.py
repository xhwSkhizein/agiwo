"""Scheduler router — state queries, control operations, and scheduler chat SSE."""

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from agiwo.scheduler.commands import RouteStreamMode
from agiwo.scheduler.models import AgentStateStatus

from server.dependencies import ConsoleRuntimeDep, SchedulerDep
from server.response_serialization import (
    agent_state_list_item_from_sdk,
    agent_state_response_from_sdk,
    pending_event_response_from_sdk,
    scheduler_tree_response_from_record,
)
from server.models.view import (
    AgentStateListItem,
    AgentStateResponse,
    CancelRequest,
    CreateAgentRequest,
    PageResponse,
    PendingEventResponse,
    ResumeRequest,
    SchedulerTreeResponse,
    SchedulerStatsResponse,
    SteerRequest,
)
from server.services.runtime import (
    PersistentAgentNotFoundError,
    PersistentAgentValidationError,
    SchedulerTreeNotFoundError,
    SchedulerTreeTooLargeError,
    SchedulerTreeValidationError,
    SchedulerTreeViewService,
    materialize_agent,
    resume_persistent_agent,
)

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])
SCHEDULER_TREE_MAX_NODES = 500


def _scheduler_tree_service(
    runtime: ConsoleRuntimeDep,
    scheduler: SchedulerDep,
) -> SchedulerTreeViewService:
    return SchedulerTreeViewService(
        scheduler=scheduler,
        session_store=runtime.session_store,
        max_nodes=SCHEDULER_TREE_MAX_NODES,
    )


# ── State Queries ────────────────────────────────────────────────────────────


@router.get("/states", response_model=PageResponse[AgentStateListItem])
async def list_agent_states(
    scheduler: SchedulerDep,
    runtime: ConsoleRuntimeDep,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> PageResponse[AgentStateListItem]:
    """List all agent states with optional status filter."""
    status_enum = None
    if status is not None:
        try:
            status_enum = AgentStateStatus(status)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid status: {status}"
            ) from exc
    states = await scheduler.list_states(
        statuses=(status_enum,) if status_enum is not None else None,
        limit=limit + 1,
        offset=offset,
    )
    has_more = len(states) > limit
    page = states[:limit]
    tree_service = _scheduler_tree_service(runtime, scheduler)
    root_ids = await tree_service.resolve_root_state_ids([state.id for state in page])
    return PageResponse(
        items=[
            agent_state_list_item_from_sdk(s, root_state_id=root_ids.get(s.id))
            for s in page
        ],
        limit=limit,
        offset=offset,
        has_more=has_more,
        total=None,
    )


@router.get("/states/{state_id}", response_model=AgentStateResponse)
async def get_agent_state(
    state_id: str,
    scheduler: SchedulerDep,
    runtime: ConsoleRuntimeDep,
) -> AgentStateResponse:
    """Get a single agent state by ID."""
    state = await scheduler.get_state(state_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Agent state not found")
    tree_service = _scheduler_tree_service(runtime, scheduler)
    root_state_id = await tree_service.resolve_root_state_id(state_id)
    return agent_state_response_from_sdk(state, root_state_id=root_state_id)


@router.get("/states/{state_id}/children", response_model=list[AgentStateListItem])
async def get_children(
    state_id: str,
    scheduler: SchedulerDep,
    runtime: ConsoleRuntimeDep,
) -> list[AgentStateListItem]:
    """Get child agent states for a given parent state."""
    children = await scheduler.list_states(parent_id=state_id, limit=1000)
    children.sort(key=lambda s: s.created_at, reverse=True)
    tree_service = _scheduler_tree_service(runtime, scheduler)
    root_state_id = await tree_service.resolve_root_state_id(state_id)
    return [
        agent_state_list_item_from_sdk(child, root_state_id=root_state_id)
        for child in children
    ]


@router.get("/states/{state_id}/tree", response_model=SchedulerTreeResponse)
async def get_state_tree(
    state_id: str,
    scheduler: SchedulerDep,
    runtime: ConsoleRuntimeDep,
) -> SchedulerTreeResponse:
    tree_service = _scheduler_tree_service(runtime, scheduler)
    try:
        tree = await tree_service.get_tree(state_id)
    except SchedulerTreeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SchedulerTreeTooLargeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SchedulerTreeValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return scheduler_tree_response_from_record(tree)


@router.get("/stats", response_model=SchedulerStatsResponse)
async def get_stats(scheduler: SchedulerDep) -> SchedulerStatsResponse:
    """Get scheduler statistics — count of states by status."""
    return SchedulerStatsResponse(**await scheduler.get_stats())


# ── State Control ────────────────────────────────────────────────────────────


@router.post("/states/{state_id}/steer")
async def steer_agent(
    state_id: str,
    body: SteerRequest,
    scheduler: SchedulerDep,
) -> dict[str, bool]:
    """Send a steering message to an agent (root only)."""
    ok = await scheduler.steer(state_id, body.message, urgent=body.urgent)
    if not ok:
        raise HTTPException(
            status_code=404, detail="Agent not found or steering unavailable"
        )
    return {"ok": True}


@router.post("/states/{state_id}/cancel")
async def cancel_agent(
    state_id: str,
    body: CancelRequest,
    scheduler: SchedulerDep,
) -> dict[str, bool]:
    """Cancel an agent and all its descendants."""
    ok = await scheduler.cancel(state_id, reason=body.reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Agent not found or not active")
    return {"ok": True}


@router.post("/states/{state_id}/resume")
async def resume_agent(
    state_id: str,
    body: ResumeRequest,
    scheduler: SchedulerDep,
    runtime: ConsoleRuntimeDep,
) -> dict[str, bool]:
    """Resume a persistent root agent by submitting a new task."""
    try:
        await resume_persistent_agent(
            scheduler,
            state_id=state_id,
            message=body.message,
            registry=runtime.agent_registry,
            console_config=runtime.config,
        )
    except PersistentAgentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PersistentAgentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@router.get(
    "/states/{state_id}/pending-events", response_model=list[PendingEventResponse]
)
async def list_pending_events(
    state_id: str,
    scheduler: SchedulerDep,
) -> list[PendingEventResponse]:
    """List pending events for an agent."""
    state = await scheduler.get_state(state_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Agent state not found")
    events = await scheduler.list_events(
        target_agent_id=state_id,
        session_id=state.session_id,
    )
    return [pending_event_response_from_sdk(event) for event in events]


@router.post("/states/create")
async def create_persistent_agent(
    body: CreateAgentRequest,
    scheduler: SchedulerDep,
    runtime: ConsoleRuntimeDep,
) -> dict[str, Any]:
    """Create and submit a new persistent root agent from a config template."""
    config_id = body.agent_config_id
    if config_id is None:
        raise HTTPException(status_code=400, detail="agent_config_id is required")

    config = await runtime.agent_registry.get_agent(config_id)
    if config is None:
        raise HTTPException(
            status_code=404, detail=f"Agent config '{config_id}' not found"
        )

    instance_id = f"{config.id}--{uuid4()}"
    agent = await materialize_agent(
        config,
        runtime.config,
        runtime.agent_registry,
        id=instance_id,
    )
    route_result = await scheduler.route_root_input(
        body.initial_task or "",
        agent=agent,
        session_id=body.session_id or str(uuid4()),
        persistent=True,
        agent_config_id=config_id,
        stream_mode=RouteStreamMode.UNTIL_SETTLED,
    )
    return {"ok": True, "state_id": route_result.state_id}
