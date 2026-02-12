"""
Scheduler Chat API router — real-time Scheduler orchestration via SSE.

Uses AgentHooks.on_event to forward all agent events (root + children)
through a shared asyncio.Queue to the SSE connection.
"""

import asyncio
import json
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from agiwo.agent.hooks import AgentHooks
from agiwo.agent.schema import StreamEvent
from agiwo.utils.abort_signal import AbortSignal

from server.dependencies import get_agent_registry, get_console_config, get_scheduler
from server.schemas import ChatRequest
from server.services.agent_builder import build_agent, serialize_event

router = APIRouter(prefix="/api/scheduler/chat", tags=["scheduler-chat"])

_SENTINEL = object()


class CancelRequest(BaseModel):
    state_id: str


@router.post("/{agent_id}")
async def scheduler_chat(agent_id: str, body: ChatRequest) -> EventSourceResponse:
    """
    Send a message to an agent via the Scheduler and stream all orchestration
    events (root + children) via SSE.

    Events include standard agent events (step_delta, step_completed, run_started,
    run_completed) plus scheduler-level events (scheduler_completed, scheduler_failed).
    """
    registry = get_agent_registry()
    agent_config = await registry.get_agent(agent_id)
    if agent_config is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    console_config = get_console_config()
    scheduler = get_scheduler()

    agent = await build_agent(agent_config, console_config, registry)
    session_id = body.session_id or str(uuid4())

    event_queue: asyncio.Queue[StreamEvent | object] = asyncio.Queue()
    abort_signal = AbortSignal()

    async def on_event(event: StreamEvent) -> None:
        await event_queue.put(event)

    agent.hooks = AgentHooks(on_event=on_event)

    async def run_orchestration() -> None:
        try:
            state_id = await scheduler.submit(
                agent, body.message, session_id=session_id, abort_signal=abort_signal
            )
            result = await scheduler.wait_for(state_id, timeout=600)
            completion_data = {
                "type": "scheduler_completed",
                "state_id": state_id,
                "response": result.response,
                "termination_reason": result.termination_reason.value if result.termination_reason else None,
            }
            await event_queue.put(completion_data)
        except Exception as e:
            error_data = {"type": "scheduler_failed", "error": str(e)}
            await event_queue.put(error_data)
        finally:
            await event_queue.put(_SENTINEL)

    async def event_generator():
        task = asyncio.create_task(run_orchestration())
        try:
            while True:
                item = await event_queue.get()
                if item is _SENTINEL:
                    break
                if isinstance(item, StreamEvent):
                    yield {"event": item.type.value, "data": serialize_event(item)}
                elif isinstance(item, dict):
                    event_type = item.get("type", "scheduler_event")
                    yield {"event": event_type, "data": json.dumps(item, default=str)}
        except asyncio.CancelledError:
            abort_signal.abort("SSE connection closed")
        finally:
            if not task.done():
                task.cancel()
            await agent.close()

    return EventSourceResponse(event_generator())


@router.post("/{agent_id}/cancel")
async def cancel_orchestration(agent_id: str, body: CancelRequest):
    """Cancel a running scheduler orchestration."""
    scheduler = get_scheduler()
    success = await scheduler.cancel(body.state_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"No active orchestration found for state_id={body.state_id}",
        )
    return {"ok": True, "state_id": body.state_id}
