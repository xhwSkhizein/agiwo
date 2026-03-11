"""Scheduler control API router."""

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from server.dependencies import ConsoleRuntimeDep
from server.response_serialization import pending_event_to_response
from server.schemas import (
    CancelRequest,
    CreateAgentRequest,
    PendingEventResponse,
    ResumeRequest,
    SteerRequest,
)
from server.services.agent_lifecycle import (
    PersistentAgentNotFoundError,
    PersistentAgentValidationError,
    build_agent,
    resume_persistent_agent,
)

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])
@router.post("/states/{state_id}/steer")
async def steer_agent(
    state_id: str,
    body: SteerRequest,
    runtime: ConsoleRuntimeDep,
) -> dict[str, bool]:
    """Send a steering message to an agent (root only)."""
    scheduler = runtime.scheduler
    assert scheduler is not None
    ok = await scheduler.steer(state_id, body.message, urgent=body.urgent)
    if not ok:
        raise HTTPException(status_code=404, detail="Agent not found or steering unavailable")
    return {"ok": True}
@router.post("/states/{state_id}/cancel")
async def cancel_agent(
    state_id: str,
    body: CancelRequest,
    runtime: ConsoleRuntimeDep,
) -> dict[str, bool]:
    """Cancel an agent and all its descendants."""
    scheduler = runtime.scheduler
    assert scheduler is not None
    ok = await scheduler.cancel(state_id, reason=body.reason)
    if not ok:
        raise HTTPException(status_code=404, detail="Agent not found or not active")
    return {"ok": True}


@router.post("/states/{state_id}/resume")
async def resume_agent(
    state_id: str,
    body: ResumeRequest,
    runtime: ConsoleRuntimeDep,
) -> dict[str, bool]:
    """Resume a persistent root agent by submitting a new task."""
    scheduler = runtime.scheduler
    assert scheduler is not None
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


@router.get("/states/{state_id}/pending-events", response_model=list[PendingEventResponse])
async def list_pending_events(
    state_id: str,
    runtime: ConsoleRuntimeDep,
) -> list[PendingEventResponse]:
    """List pending events for an agent."""
    scheduler = runtime.scheduler
    assert scheduler is not None
    storage = scheduler.store
    state = await storage.get_state(state_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Agent state not found")
    events = await storage.get_pending_events(state_id, state.session_id)
    return [pending_event_to_response(event) for event in events]


@router.post("/states/create")
async def create_persistent_agent(
    body: CreateAgentRequest,
    runtime: ConsoleRuntimeDep,
) -> dict[str, Any]:
    """Create and submit a new persistent root agent from a config template."""
    scheduler = runtime.scheduler
    assert scheduler is not None

    config_id = body.agent_config_id
    if config_id is None:
        raise HTTPException(status_code=400, detail="agent_config_id is required")

    config = await runtime.agent_registry.get_agent(config_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Agent config '{config_id}' not found")

    agent = await build_agent(config, runtime.config, runtime.agent_registry)
    state_id = await scheduler.submit(
        agent,
        body.initial_task or "",
        session_id=body.session_id or str(uuid4()),
        persistent=True,
        agent_config_id=config_id,
    )
    return {"ok": True, "state_id": state_id}
