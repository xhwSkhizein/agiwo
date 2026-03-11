"""Shared SSE conversation runner for chat-style routes."""

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from uuid import uuid4

from fastapi import HTTPException
from sse_starlette.sse import EventSourceResponse

from agiwo.agent.agent import Agent
from agiwo.agent.schema import StreamEvent

from server.dependencies import ConsoleRuntime
from server.services.agent_lifecycle import build_agent, serialize_event

SseMessage = dict[str, str]
ConversationEventStrategy = Callable[
    [ConsoleRuntime, Agent, str, str],
    AsyncIterator[SseMessage],
]
UnexpectedErrorBuilder = Callable[[Exception], SseMessage]


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
            async for message in strategy(
                runtime,
                prepared.agent,
                prepared.message,
                prepared.session_id,
            ):
                yield message
        except Exception as exc:  # noqa: BLE001
            yield error_builder(exc)
        finally:
            await prepared.agent.close()

    return EventSourceResponse(event_generator())


__all__ = [
    "ConversationEventStrategy",
    "PreparedConversation",
    "SseMessage",
    "create_conversation_response",
    "error_event_message",
    "json_event_message",
    "prepare_conversation",
    "stream_event_message",
]
