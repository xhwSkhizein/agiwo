"""Chat API router — real-time Agent conversation via SSE."""

from collections.abc import AsyncIterator
from datetime import datetime, timezone
import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from server.dependencies import ConsoleRuntime, ConsoleRuntimeDep, SchedulerDep
from server.models.session import Session
from server.response_serialization import stream_event_to_sse_message
from server.models.view import (
    ChatRequest,
    CreateSessionRequest,
    ForkSessionRequest,
    SchedulerChatCancelRequest,
    SwitchSessionRequest,
)
from server.services.metrics import (
    collect_session_aggregates,
    session_aggregate_to_chat_summary,
)
from server.services.runtime import (
    SessionContextService,
    SessionRuntimeService,
    build_agent,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/{agent_id}")
async def chat(
    agent_id: str,
    body: ChatRequest,
    runtime: ConsoleRuntimeDep,
) -> EventSourceResponse:
    """Send a message to an agent via the shared runtime service and stream SSE."""
    registry = runtime.agent_registry
    agent_config = await registry.get_agent(agent_id)
    if agent_config is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    try:
        agent = await build_agent(agent_config, runtime.config, registry)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    scheduler = runtime.scheduler
    if scheduler is None:
        raise RuntimeError("Scheduler not initialized")
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")

    session = await _get_or_create_chat_session(
        session_id=body.session_id or str(uuid4()),
        agent_id=agent_id,
        runtime=runtime,
    )
    runtime_service = SessionRuntimeService(
        scheduler=scheduler,
        session_store=runtime.session_store,
    )
    dispatch = await runtime_service.execute(agent, session, body.message)

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        try:
            if dispatch.stream is not None:
                async for item in dispatch.stream:
                    yield stream_event_to_sse_message(item)
                return

            state = await runtime_service.get_state(session.scheduler_state_id)
            if state is not None and state.result_summary:
                yield {
                    "event": "scheduler_ack",
                    "data": json.dumps(
                        {
                            "type": "scheduler_ack",
                            "result_summary": state.result_summary,
                        },
                        default=str,
                    ),
                }
                return
            yield {
                "event": "scheduler_ack",
                "data": json.dumps(
                    {
                        "type": "scheduler_ack",
                        "message": "消息已收到，正在继续处理。",
                    },
                    default=str,
                ),
            }
        finally:
            await agent.close()

    return EventSourceResponse(event_generator())


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


async def _get_or_create_chat_session(
    *,
    session_id: str,
    agent_id: str,
    runtime: ConsoleRuntime,
) -> Session:
    assert runtime.session_store is not None
    session = await runtime.session_store.get_session(session_id)
    if session is not None:
        if session.base_agent_id != agent_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Session '{session_id}' belongs to agent "
                    f"'{session.base_agent_id}', not '{agent_id}'"
                ),
            )
        return session

    now = datetime.now(timezone.utc)
    session = Session(
        id=session_id,
        chat_context_scope_id=session_id,
        base_agent_id=agent_id,
        runtime_agent_id="",
        scheduler_state_id="",
        created_by="CONSOLE_CHAT",
        created_at=now,
        updated_at=now,
    )
    await runtime.session_store.upsert_session(session)
    return session


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
