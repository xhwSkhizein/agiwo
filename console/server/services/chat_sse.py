"""Chat SSE: build agent, stream scheduler events, emit SSE messages."""

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import HTTPException
from sse_starlette.sse import EventSourceResponse

from agiwo.agent import AgentStreamItem
from agiwo.utils.abort_signal import AbortSignal

from server.dependencies import ConsoleRuntime
from server.services.agent_lifecycle import build_agent

SseMessage = dict[str, str]


def stream_event_message(event: AgentStreamItem) -> SseMessage:
    return {
        "event": event.type,
        "data": json.dumps(event.to_dict(), default=str),
    }


async def create_scheduler_chat_response(
    agent_id: str,
    message: str,
    session_id: str | None,
    runtime: ConsoleRuntime,
) -> EventSourceResponse:
    registry = runtime.agent_registry
    agent_config = await registry.get_agent(agent_id)
    if agent_config is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    try:
        agent = await build_agent(agent_config, runtime.config, registry)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    scheduler = runtime.scheduler
    assert scheduler is not None
    resolved_session_id = session_id or str(uuid4())

    async def event_generator() -> AsyncIterator[SseMessage]:
        abort_signal = AbortSignal()
        try:
            async for item in scheduler.stream(
                message,
                agent=agent,
                session_id=resolved_session_id,
                abort_signal=abort_signal,
                timeout=600,
            ):
                yield stream_event_message(item)
        except (asyncio.CancelledError, GeneratorExit):
            abort_signal.abort("SSE connection closed")
            raise
        except Exception as exc:  # noqa: BLE001
            yield {
                "event": "scheduler_failed",
                "data": json.dumps(
                    {"type": "scheduler_failed", "error": str(exc)}, default=str
                ),
            }
        finally:
            await agent.close()

    return EventSourceResponse(event_generator())
