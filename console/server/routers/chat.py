"""Chat API router — real-time Agent conversation via SSE."""

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from server.dependencies import ConsoleRuntimeDep
from server.domain.sessions import session_aggregate_to_chat_summary
from server.schemas import ChatRequest
from server.services.chat_sse import (
    create_conversation_response,
    scheduler_error_message,
    stream_scheduler_events,
)
from server.services.metrics import collect_session_aggregates

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/{agent_id}")
async def chat(
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


@router.get("/{agent_id}/sessions")
async def list_agent_sessions(
    agent_id: str,
    runtime: ConsoleRuntimeDep,
):
    """Get conversation sessions for a specific agent."""
    storage = runtime.run_step_storage
    sessions = await collect_session_aggregates(storage, agent_id=agent_id)
    return [session_aggregate_to_chat_summary(session) for session in sessions]
