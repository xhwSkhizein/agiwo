"""Chat SSE: shared helpers, direct chat strategy, and scheduler chat strategy."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from uuid import uuid4

from fastapi import HTTPException
from sse_starlette.sse import EventSourceResponse

from agiwo.agent import Agent, StreamEvent
from agiwo.agent.hooks import AgentHooks
from agiwo.utils.abort_signal import AbortSignal

from server.dependencies import ConsoleRuntime
from server.domain.scheduler_events import (
    scheduler_completed_payload,
    scheduler_failed_payload,
)
from server.services.agent_lifecycle import build_agent, serialize_event

SseMessage = dict[str, str]
ConversationEventStrategy = Callable[
    [ConsoleRuntime, Agent, str, str],
    AsyncIterator[SseMessage],
]
UnexpectedErrorBuilder = Callable[[Exception], SseMessage]

_SENTINEL = object()


# ── Shared helpers ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PreparedConversation:
    agent: Agent
    message: str
    session_id: str


def stream_event_message(event: StreamEvent) -> SseMessage:
    return {"event": event.type.value, "data": serialize_event(event)}


def json_event_message(event_name: str, payload: object) -> SseMessage:
    return {"event": event_name, "data": json.dumps(payload, default=str)}


def error_event_message(error: Exception) -> SseMessage:
    return json_event_message("error", {"error": str(error)})


async def prepare_conversation(
    agent_id: str,
    message: str,
    session_id: str | None,
    runtime: ConsoleRuntime,
) -> PreparedConversation:
    registry = runtime.agent_registry
    agent_config = await registry.get_agent(agent_id)
    if agent_config is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    try:
        agent = await build_agent(agent_config, runtime.config, registry)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PreparedConversation(
        agent=agent,
        message=message,
        session_id=session_id or str(uuid4()),
    )


async def create_conversation_response(
    agent_id: str,
    message: str,
    session_id: str | None,
    runtime: ConsoleRuntime,
    strategy: ConversationEventStrategy,
    *,
    unexpected_error_builder: UnexpectedErrorBuilder | None = None,
) -> EventSourceResponse:
    prepared = await prepare_conversation(agent_id, message, session_id, runtime)
    error_builder = unexpected_error_builder or error_event_message

    async def event_generator() -> AsyncIterator[SseMessage]:
        try:
            async for sse_msg in strategy(
                runtime,
                prepared.agent,
                prepared.message,
                prepared.session_id,
            ):
                yield sse_msg
        except Exception as exc:  # noqa: BLE001
            yield error_builder(exc)
        finally:
            await prepared.agent.close()

    return EventSourceResponse(event_generator())


# ── Scheduler chat strategy ─────────────────────────────────────────────────


def scheduler_error_message(error: Exception) -> SseMessage:
    return json_event_message(
        "scheduler_failed",
        scheduler_failed_payload(str(error)),
    )


async def stream_scheduler_events(
    runtime: ConsoleRuntime,
    agent: Agent,
    message: str,
    session_id: str,
) -> AsyncIterator[SseMessage]:
    scheduler = runtime.scheduler
    assert scheduler is not None
    event_queue: asyncio.Queue[StreamEvent | dict[str, object] | object] = asyncio.Queue()
    abort_signal = AbortSignal()

    async def on_event(event: StreamEvent) -> None:
        await event_queue.put(event)

    agent.hooks = AgentHooks(on_event=on_event)

    async def run_orchestration() -> None:
        try:
            state_id = await scheduler.submit(
                agent,
                message,
                session_id=session_id,
                abort_signal=abort_signal,
            )
            result = await scheduler.wait_for(state_id, timeout=600)
            await event_queue.put(
                scheduler_completed_payload(
                    state_id=state_id,
                    response=result.response,
                    termination_reason=(
                        result.termination_reason.value
                        if result.termination_reason
                        else None
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            await event_queue.put(scheduler_failed_payload(str(exc)))
        finally:
            await event_queue.put(_SENTINEL)

    task = asyncio.create_task(run_orchestration())
    try:
        while True:
            item = await event_queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, StreamEvent):
                yield stream_event_message(item)
                continue
            if isinstance(item, dict):
                event_type = item.get("type")
                yield json_event_message(
                    event_type if isinstance(event_type, str) else "scheduler_event",
                    item,
                )
    except asyncio.CancelledError:
        abort_signal.abort("SSE connection closed")
        raise
    finally:
        if not task.done():
            task.cancel()
