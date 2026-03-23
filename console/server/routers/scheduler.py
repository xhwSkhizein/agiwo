"""Scheduler router — state queries, control operations, and scheduler chat SSE."""

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agiwo.scheduler.models import AgentStateStatus

from server.dependencies import ConsoleRuntimeDep, SchedulerDep
from server.response_serialization import (
    pending_event_to_response,
    state_to_list_item as _state_to_list_item,
    state_to_response as _state_to_response,
)
from server.schemas import (
    AgentStateListItem,
    AgentStateResponse,
    CancelRequest,
    ChatRequest,
    CreateAgentRequest,
    PendingEventResponse,
    ResumeRequest,
    SchedulerStatsResponse,
    SteerRequest,
)
from server.services.agent_lifecycle import (
    PersistentAgentNotFoundError,
    PersistentAgentValidationError,
    build_agent,
    resume_persistent_agent,
)
from server.services.chat_sse import (
    create_conversation_response,
    scheduler_error_message,
    stream_scheduler_events,
)
from server.services.metrics import build_metrics_by_state

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


class SchedulerChatCancelRequest(BaseModel):
    state_id: str


# ── State Queries ────────────────────────────────────────────────────────────


@router.get("/states", response_model=list[AgentStateListItem])
async def list_agent_states(
    scheduler: SchedulerDep,
    runtime: ConsoleRuntimeDep,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[AgentStateListItem]:
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
        limit=limit,
        offset=offset,
    )
    run_storage = runtime.run_step_storage
    metrics_by_state = await build_metrics_by_state(states, run_storage)
    return [
        _state_to_list_item(
            s,
            metrics_by_state.get((s.resolve_runtime_session_id(), s.id)),
        )
        for s in states
    ]


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
    run_storage = runtime.run_step_storage
    metrics_by_state = await build_metrics_by_state([state], run_storage)
    return _state_to_response(
        state,
        metrics_by_state.get((state.resolve_runtime_session_id(), state.id)),
    )


@router.get("/states/{state_id}/children", response_model=list[AgentStateListItem])
async def get_children(
    state_id: str,
    scheduler: SchedulerDep,
    runtime: ConsoleRuntimeDep,
) -> list[AgentStateListItem]:
    """Get child agent states for a given parent state."""
    children = await scheduler.list_states(parent_id=state_id, limit=1000)
    children.sort(key=lambda s: s.created_at, reverse=True)
    run_storage = runtime.run_step_storage
    metrics_by_state = await build_metrics_by_state(children, run_storage)
    return [
        _state_to_list_item(
            s,
            metrics_by_state.get((s.resolve_runtime_session_id(), s.id)),
        )
        for s in children
    ]


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
    return [pending_event_to_response(event) for event in events]


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

    agent = await build_agent(config, runtime.config, runtime.agent_registry)
    state_id = await scheduler.submit(
        agent,
        body.initial_task or "",
        session_id=body.session_id or str(uuid4()),
        persistent=True,
        agent_config_id=config_id,
    )
    return {"ok": True, "state_id": state_id}


# ── Scheduler Chat SSE ──────────────────────────────────────────────────────


@router.post("/chat/{agent_id}")
async def scheduler_chat(
    agent_id: str,
    body: ChatRequest,
    runtime: ConsoleRuntimeDep,
) -> EventSourceResponse:
    """Send a message to an agent via the Scheduler and stream SSE events."""
    return await create_conversation_response(
        agent_id,
        body.message,
        body.session_id,
        runtime,
        stream_scheduler_events,
        unexpected_error_builder=scheduler_error_message,
    )


@router.post("/chat/{agent_id}/cancel")
async def cancel_orchestration(
    agent_id: str,
    body: SchedulerChatCancelRequest,
    scheduler: SchedulerDep,
):
    """Cancel a running scheduler orchestration."""
    del agent_id
    success = await scheduler.cancel(body.state_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"No active orchestration found for state_id={body.state_id}",
        )
    return {"ok": True, "state_id": body.state_id}
