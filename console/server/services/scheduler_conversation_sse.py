"""Scheduler-specific SSE event strategy for chat routes."""

import asyncio
from collections.abc import AsyncIterator

from agiwo.agent.agent import Agent
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.schema import StreamEvent
from agiwo.utils.abort_signal import AbortSignal

from server.dependencies import ConsoleRuntime
from server.domain.scheduler_events import (
    scheduler_completed_payload,
    scheduler_failed_payload,
)
from server.services.conversation_sse import json_event_message, stream_event_message

_SENTINEL = object()


def scheduler_error_message(error: Exception) -> dict[str, str]:
    return json_event_message(
        "scheduler_failed",
        scheduler_failed_payload(str(error)),
    )


async def stream_scheduler_events(
    runtime: ConsoleRuntime,
    agent: Agent,
    message: str,
    session_id: str,
) -> AsyncIterator[dict[str, str]]:
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


__all__ = [
    "scheduler_error_message",
    "stream_scheduler_events",
]
