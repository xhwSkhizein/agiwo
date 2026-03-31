"""Chat API router — real-time Agent conversation via SSE."""

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from server.channels.session import SessionContextService
from server.dependencies import ConsoleRuntime, ConsoleRuntimeDep, SchedulerDep
from server.models import (
    ChatRequest,
    CreateSessionRequest,
    ForkSessionRequest,
    SchedulerChatCancelRequest,
    SwitchSessionRequest,
)
from server.services.chat_sse import create_scheduler_chat_response
from server.services.metrics import (
    collect_session_aggregates,
    session_aggregate_to_chat_summary,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/{agent_id}")
async def chat(
    agent_id: str,
    body: ChatRequest,
    runtime: ConsoleRuntimeDep,
) -> EventSourceResponse:
    """Send a message to an agent via the Scheduler and stream SSE events."""
    return await create_scheduler_chat_response(
        agent_id,
        body.message,
        body.session_id,
        runtime,
    )


@router.post("/{agent_id}/cancel")
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


@router.get("/{agent_id}/sessions")
async def list_agent_sessions(
    agent_id: str,
    runtime: ConsoleRuntimeDep,
):
    """Get conversation sessions for a specific agent."""
    storage = runtime.run_step_storage
    sessions = await collect_session_aggregates(storage, agent_id=agent_id)
    return [session_aggregate_to_chat_summary(session) for session in sessions]


# ── Session Management ──────────────────────────────────────────────────


def _get_session_service(runtime: ConsoleRuntime) -> SessionContextService:
    """Build a SessionContextService from the console runtime."""
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")
    return SessionContextService(
        store=runtime.session_store,
        agent_registry=runtime.agent_registry,
        default_agent_name=runtime.config.default_agent.name,
    )


@router.post("/{agent_id}/sessions/create")
async def create_session(
    agent_id: str,
    body: CreateSessionRequest,
    runtime: ConsoleRuntimeDep,
):
    """Create a new session for an agent."""
    service = _get_session_service(runtime)
    result = await service.create_new_session(
        chat_context_scope_id=body.chat_context_scope_id,
        channel_instance_id=body.channel_instance_id,
        chat_id=body.chat_context_scope_id,
        chat_type="dm",
        user_open_id=body.user_open_id,
        base_agent_id=agent_id,
        created_by="CONSOLE_CREATE",
    )
    return {
        "session_id": result.session.id,
        "task_id": result.session.current_task_id,
        "source_session_id": result.session.source_session_id,
    }


@router.post("/{agent_id}/sessions/switch")
async def switch_session(
    agent_id: str,
    body: SwitchSessionRequest,
    runtime: ConsoleRuntimeDep,
):
    """Switch to a different session."""
    del agent_id
    service = _get_session_service(runtime)
    result = await service.switch_session(
        chat_context_scope_id=body.chat_context_scope_id,
        target_session_id=body.target_session_id,
    )
    return {
        "session_id": result.current_session.id,
        "task_id": result.current_session.current_task_id,
        "previous_session_id": (
            result.previous_session.id if result.previous_session else None
        ),
    }


@router.post("/{agent_id}/sessions/{session_id}/fork")
async def fork_session(
    agent_id: str,
    session_id: str,
    body: ForkSessionRequest,
    runtime: ConsoleRuntimeDep,
):
    """Fork a session into a new one with weak lineage."""
    del agent_id
    service = _get_session_service(runtime)
    result = await service.fork_session_by_id(
        session_id=session_id,
        context_summary=body.context_summary,
        created_by="CONSOLE_FORK",
    )
    return {
        "session_id": result.session.id,
        "task_id": result.session.current_task_id,
        "source_session_id": result.session.source_session_id,
    }
