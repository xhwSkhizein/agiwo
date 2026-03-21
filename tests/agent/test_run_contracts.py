import asyncio
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, AgentConfig
from agiwo.llm.base import Model, StreamChunk


class FixedResponseModel(Model):
    def __init__(self, response: str = "final answer") -> None:
        super().__init__(id="fixed-model", name="fixed-model", temperature=0.0)
        self._response = response

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


@pytest.mark.asyncio
async def test_handle_wait_is_idempotent() -> None:
    agent = Agent(
        AgentConfig(name="run-contract", description="run contract test"),
        model=FixedResponseModel(),
    )

    handle = agent.start("hello", session_id="run-contract-session")
    first = await handle.wait()
    second = await handle.wait()

    assert first.response == "final answer"
    assert second.response == first.response
    assert second.run_id == first.run_id


@pytest.mark.asyncio
async def test_handle_supports_multiple_stream_subscribers() -> None:
    agent = Agent(
        AgentConfig(name="stream-contract", description="stream contract test"),
        model=FixedResponseModel(),
    )

    handle = agent.start("hello", session_id="stream-contract-session")
    stream_one = handle.stream()
    stream_two = handle.stream()

    events_one, events_two = await asyncio.gather(
        _collect_stream(stream_one),
        _collect_stream(stream_two),
    )
    result = await handle.wait()

    assert result.response == "final answer"
    assert [event.type for event in events_one] == [event.type for event in events_two]


async def _collect_stream(stream):
    events = []
    async for event in stream:
        events.append(event)
        if event.type in {"run_completed", "run_failed"}:
            break
    return events
