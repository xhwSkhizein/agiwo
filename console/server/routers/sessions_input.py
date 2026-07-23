"""Session input SSE router (MainAgent.accept streaming)."""

from collections.abc import AsyncIterator
import json
import uuid

from fastapi import APIRouter, Header, HTTPException
from sse_starlette.sse import EventSourceResponse

from agiwo.agent.models.input import UserMessage
from agiwo.agent.models.stream import RunCompletedEvent, RunFailedEvent

from server.channels.exceptions import BaseAgentNotFoundError
from server.dependencies import ConsoleRuntimeDep, get_session_gateway
from server.models.view import ChatRequest
from server.response_serialization import stream_event_to_sse_message

router = APIRouter(prefix="/api", tags=["sessions"])


async def _session_input_event_stream(
    *,
    session_id: str,
    body: ChatRequest,
    idempotency_key: str,
    gateway,
) -> AsyncIterator[dict[str, str]]:
    message = UserMessage.from_value(body.message)
    try:
        started = await gateway.start_user_message(
            session_id,
            message,
            idempotency_key=idempotency_key,
        )
    except (ValueError, BaseAgentNotFoundError) as exc:
        yield {
            "event": "session_error",
            "data": json.dumps({"message": str(exc)}, default=str),
        }
        return

    yield {
        "event": "session_accepted",
        "data": json.dumps(
            {
                "session_id": started.session_id,
                "run_id": started.run_id,
            },
            default=str,
        ),
    }

    final_response: str | None = None
    final_status = "completed"
    async for item in started.main_agent.subscribe():
        yield stream_event_to_sse_message(item)
        if isinstance(item, RunCompletedEvent):
            final_response = item.response
            final_status = "completed"
        elif isinstance(item, RunFailedEvent):
            final_status = "failed"
            final_response = item.error

    yield {
        "event": "session_turn",
        "data": json.dumps(
            {
                "kind": "session",
                "session_id": started.session_id,
                "run_id": started.run_id,
                "status": final_status,
                "response": final_response,
            },
            default=str,
        ),
    }


@router.post("/sessions/{session_id}/input")
async def send_session_input(
    session_id: str,
    body: ChatRequest,
    runtime: ConsoleRuntimeDep,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> EventSourceResponse:
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")
    session = await runtime.session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    gateway = get_session_gateway(runtime)
    key = idempotency_key or uuid.uuid4().hex

    return EventSourceResponse(
        _session_input_event_stream(
            gateway=gateway,
            session_id=session_id,
            body=body,
            idempotency_key=key,
        )
    )
