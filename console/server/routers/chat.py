"""Chat API router — real-time Agent conversation via SSE."""

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from agiwo.agent import Agent
from agiwo.utils.abort_signal import AbortSignal

from server.dependencies import ConsoleRuntime, ConsoleRuntimeDep
from server.domain.sessions import session_aggregate_to_chat_summary
from server.schemas import ChatRequest
from server.services.chat_sse import (
    create_conversation_response,
    stream_event_message,
)
from server.services.metrics import collect_session_aggregates

router = APIRouter(prefix="/api/chat", tags=["chat"])


async def _stream_chat_events(
    _runtime: ConsoleRuntime,
    agent: Agent,
    message: str,
    session_id: str,
) -> AsyncIterator[dict[str, str]]:
    abort_signal = AbortSignal()
    handle = agent.start(
        message,
        session_id=session_id,
        abort_signal=abort_signal,
    )

    completed = False
    try:
        async for event in handle.stream():
            yield stream_event_message(event)
        await handle.wait()
        completed = True
    finally:
        if not completed:
            handle.cancel("SSE connection closed")
            try:
                await handle.wait()
            except asyncio.CancelledError:
                pass


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
    storage = runtime.run_step_storage
    sessions = await collect_session_aggregates(storage, agent_id=agent_id)
    return [session_aggregate_to_chat_summary(session) for session in sessions]
