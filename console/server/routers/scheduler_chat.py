"""
Scheduler Chat API router — real-time Scheduler orchestration via SSE.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from server.dependencies import ConsoleRuntimeDep
from server.schemas import ChatRequest
from server.services.conversation_sse import (
    create_conversation_response,
)
from server.services.scheduler_conversation_sse import (
    scheduler_error_message,
    stream_scheduler_events,
)

router = APIRouter(prefix="/api/scheduler/chat", tags=["scheduler-chat"])


class CancelRequest(BaseModel):
    state_id: str


@router.post("/{agent_id}")
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


@router.post("/{agent_id}/cancel")
async def cancel_orchestration(
    agent_id: str,
    body: CancelRequest,
    runtime: ConsoleRuntimeDep,
):
    """Cancel a running scheduler orchestration."""
    del agent_id
    scheduler = runtime.scheduler
    assert scheduler is not None
    success = await scheduler.cancel(body.state_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"No active orchestration found for state_id={body.state_id}",
        )
    return {"ok": True, "state_id": body.state_id}
