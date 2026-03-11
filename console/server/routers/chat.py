"""Chat API router — real-time Agent conversation via SSE."""

from collections.abc import AsyncIterator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from agiwo.agent.agent import Agent

from server.dependencies import ConsoleRuntime, ConsoleRuntimeDep
from server.domain.sessions import session_aggregate_to_chat_summary
from server.schemas import ChatRequest
from server.services.conversation_sse import (
    create_conversation_response,
    stream_event_message,
)
from server.services.session_summary import collect_session_aggregates

router = APIRouter(prefix="/api/chat", tags=["chat"])


async def _stream_chat_events(
    runtime: ConsoleRuntime,
    agent: Agent,
    message: str,
    session_id: str,
) -> AsyncIterator[dict[str, str]]:
    del runtime
    async for event in agent.run_stream(message, session_id=session_id):
        yield stream_event_message(event)


@router.post("/{agent_id}")
async def chat(
    agent_id: str,
    body: ChatRequest,
    runtime: ConsoleRuntimeDep,
) -> EventSourceResponse:
    """Send a message to an agent and stream the response via SSE."""
    return await create_conversation_response(
        agent_id,
        body.message,
        body.session_id,
        runtime,
        _stream_chat_events,
    )


@router.get("/{agent_id}/sessions")
async def list_agent_sessions(
    agent_id: str,
    runtime: ConsoleRuntimeDep,
):
    """Get conversation sessions for a specific agent."""
    storage = runtime.storage_manager.run_step_storage
    sessions = await collect_session_aggregates(storage, agent_id=agent_id)
    return [session_aggregate_to_chat_summary(session) for session in sessions]
