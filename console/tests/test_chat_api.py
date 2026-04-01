"""Integration tests for the Chat API streaming endpoint (scheduler-mediated)."""

from unittest.mock import AsyncMock

import pytest

from httpx import ASGITransport, AsyncClient

from agiwo.agent import RunCompletedEvent, StepDelta, StepDeltaEvent
from agiwo.scheduler.models import AgentStateStorageConfig, SchedulerConfig
from agiwo.scheduler.engine import Scheduler

from server.app import create_app
from server.channels.feishu.store.memory import InMemoryFeishuChannelStore
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

    scheduler_config = SchedulerConfig(
        state_storage=AgentStateStorageConfig(storage_type="memory"),
    )
    scheduler = Scheduler(scheduler_config)
    await scheduler.start()
    session_store = InMemoryFeishuChannelStore()
    await session_store.connect()

    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_step_storage=run_step_storage,
            trace_storage=trace_storage,
            agent_registry=registry,
            scheduler=scheduler,
            session_store=session_store,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    clear_console_runtime(app)
    await scheduler.stop()
    await registry.close()
    await run_step_storage.close()


@pytest.mark.asyncio
async def test_chat_streams_scheduler_events(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The /api/chat endpoint routes through SessionRuntimeService + route_root_input."""
    registry = _runtime(client).agent_registry
    await registry.create_agent(
        AgentConfigRecord(
            id="agent-1",
            name="chat-agent",
            model_provider="openai",
            model_name="gpt-test",
        )
    )

    class FakeAgent:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    fake_agent = FakeAgent()
    monkeypatch.setattr(
        "server.routers.chat.build_agent",
        AsyncMock(return_value=fake_agent),
    )

    scheduler = _runtime(client).scheduler
    assert scheduler is not None

    async def _stream(session_id: str):
        yield StepDeltaEvent(
            session_id=session_id,
            run_id="run-1",
            agent_id="agent-1",
            parent_run_id=None,
            depth=0,
            step_id="step-1",
            delta=StepDelta(content="hello"),
        )
        yield RunCompletedEvent(
            session_id=session_id,
            run_id="run-1",
            agent_id="agent-1",
            parent_run_id=None,
            depth=0,
            response="done",
        )

    async def fake_route_root_input(
        message: str,
        *,
        agent,
        state_id: str | None,
        session_id: str,
        persistent: bool,
        timeout: int,
    ):
        assert agent is fake_agent
        assert message == "hello"
        assert state_id is None
        assert persistent is True
        assert timeout == 600
        return type(
            "RouteResultStub",
            (),
            {
                "action": "submitted",
                "state_id": "state-1",
                "stream": _stream(session_id),
            },
        )()

    monkeypatch.setattr(scheduler, "route_root_input", fake_route_root_input)

    async with client.stream(
        "POST",
        "/api/chat/agent-1",
        json={"message": "hello", "session_id": "session-1"},
    ) as response:
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]

    assert any(line == "event: step_delta" for line in lines)
    assert any(line == "event: run_completed" for line in lines)
    assert any('"response": "done"' in line for line in lines)
    assert fake_agent.closed is True


@pytest.mark.asyncio
async def test_chat_agent_not_found(client) -> None:
    """Chat returns 404 when agent is not registered."""
    resp = await client.post(
        "/api/chat/nonexistent-agent",
        json={"message": "hello"},
    )
    assert resp.status_code == 404
