"""Integration tests for the Chat API streaming endpoint."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from httpx import ASGITransport, AsyncClient

from agiwo.agent import StepDelta, StepDeltaEvent

from server.app import create_app
from server.config import ConsoleConfig
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
    get_console_runtime_from_app,
)
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.storage_wiring import create_run_step_storage, create_trace_storage


def _runtime(client: AsyncClient) -> ConsoleRuntime:
    return get_console_runtime_from_app(client._transport.app)  # type: ignore[attr-defined]


@pytest.fixture
async def client():
    app = create_app()

    config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
    )
    run_step_storage = create_run_step_storage(config)
    trace_storage = create_trace_storage(config)
    registry = AgentRegistry(config)
    await registry.initialize()
    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_step_storage=run_step_storage,
            trace_storage=trace_storage,
            agent_registry=registry,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    clear_console_runtime(app)
    await registry.close()
    await run_step_storage.close()


class FakeStreamingAgent:
    def __init__(self, events: list[StepDeltaEvent]) -> None:
        self._events = events
        self.closed = False

    class _Handle:
        def __init__(self, events: list[StepDeltaEvent]) -> None:
            self._events = events
            self.done = False

        async def wait(self):
            self.done = True
            return None

        async def steer(self, message: str) -> bool:
            del message
            return False

        def cancel(self, reason: str | None = None) -> None:
            del reason
            self.done = True

        async def _iterate(self):
            for event in self._events:
                yield event
            self.done = True

        def stream(self):
            return self._iterate()

    def start(self, message: str, *, session_id: str, abort_signal=None):
        assert message == "hello"
        assert session_id == "session-1"
        del abort_signal
        return self._Handle(self._events)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_chat_streams_agent_events_and_closes_agent(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _runtime(client).agent_registry
    await registry.create_agent(
        AgentConfigRecord(
            id="agent-1",
            name="chat-agent",
            model_provider="openai",
            model_name="gpt-test",
        )
    )

    fake_agent = FakeStreamingAgent(
        [
            StepDeltaEvent(
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                parent_run_id=None,
                depth=0,
                step_id="step-1",
                delta=StepDelta(content="hello"),
                timestamp=datetime(2026, 3, 10),
            )
        ]
    )
    monkeypatch.setattr(
        "server.services.chat_sse.build_agent",
        AsyncMock(return_value=fake_agent),
    )

    async with client.stream(
        "POST",
        "/api/chat/agent-1",
        json={"message": "hello", "session_id": "session-1"},
    ) as response:
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]

    assert any(line == "event: step_delta" for line in lines)
    assert any('"run_id": "run-1"' in line for line in lines)
    assert fake_agent.closed is True
