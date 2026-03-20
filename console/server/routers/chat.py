"""Chat API router — real-time Agent conversation via SSE."""

from collections.abc import AsyncIterator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from agiwo.agent.agent import Agent
from agiwo.agent.streaming import consume_execution_stream
from agiwo.utils.abort_signal import AbortSignal

from server.channels.web_consent_notifier import WebChatToolConsentNotifier
from server.dependencies import ConsoleRuntime, ConsoleRuntimeDep
from server.domain.sessions import session_aggregate_to_chat_summary
from server.schemas import ChatRequest
from server.services.chat_sse import (
    create_conversation_response,
    stream_event_message,
)
from server.services.consent_notifier_injection import inject_consent_notifier
from server.services.metrics import collect_session_aggregates

router = APIRouter(prefix="/api/chat", tags=["chat"])


async def _stream_chat_events(
    _runtime: ConsoleRuntime,
    agent: Agent,
    message: str,
    session_id: str,
) -> AsyncIterator[dict[str, str]]:
    abort_signal = AbortSignal()
    handle = agent.start(message, session_id=session_id, abort_signal=abort_signal)
    
    # Inject web chat consent notifier with session runtime publish callback
    # Only inject if session_runtime is accessible (not in test mocks)
    if hasattr(handle, "_session_runtime"):
        web_consent_notifier = WebChatToolConsentNotifier(
            session_id=handle.session_id,
            agent_id=agent.id,
            parent_run_id=None,
            depth=0,
            publish_callback=handle._session_runtime.publish,
        )
        inject_consent_notifier(agent, web_consent_notifier)
    
    async for event in consume_execution_stream(
        handle,
        cancel_reason="SSE connection closed",
    ):
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
    storage = runtime.run_step_storage
    sessions = await collect_session_aggregates(storage, agent_id=agent_id)
    return [session_aggregate_to_chat_summary(session) for session in sessions]
