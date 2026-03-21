import asyncio
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, AgentConfig
from agiwo.llm.base import Model, StreamChunk


class FixedResponseModel(Model):
    def __init__(
        self,
        response: str = "final answer",
        *,
        start_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__(id="fixed-model", name="fixed-model", temperature=0.0)
        self._response = response
        self._start_event = start_event

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        if self._start_event is not None:
            await self._start_event.wait()
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
    start_event = asyncio.Event()
    agent = Agent(
        AgentConfig(name="stream-contract", description="stream contract test"),
        model=FixedResponseModel(start_event=start_event),
    )

    handle = agent.start("hello", session_id="stream-contract-session")
    stream_one = handle.stream()
    stream_two = handle.stream()

    collector_one = asyncio.create_task(_collect_stream(stream_one))
    collector_two = asyncio.create_task(_collect_stream(stream_two))
    await asyncio.sleep(0)
    start_event.set()

    events_one, events_two = await asyncio.gather(collector_one, collector_two)
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
