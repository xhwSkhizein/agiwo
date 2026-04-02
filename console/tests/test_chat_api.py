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


@pytest.mark.asyncio
async def test_chat_without_session_id_creates_agent_scoped_session_context(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Console chat should create sessions inside the agent-scoped chat context."""
    registry = _runtime(client).agent_registry
    await registry.create_agent(
        AgentConfigRecord(
            id="agent-ctx",
            name="ctx-agent",
            model_provider="openai",
            model_name="gpt-test",
        )
    )

    class FakeAgent:
        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "server.routers.chat.build_agent",
        AsyncMock(return_value=FakeAgent()),
    )

    scheduler = _runtime(client).scheduler
    assert scheduler is not None

    async def fake_route_root_input(
        message: str,
        *,
        agent,
        state_id: str | None,
        session_id: str,
        persistent: bool,
        timeout: int,
    ):
        del message, agent, state_id, persistent, timeout
        return type(
            "RouteResultStub",
            (),
            {
                "action": "submitted",
                "state_id": "state-created",
                "stream": None,
            },
        )()

    monkeypatch.setattr(scheduler, "route_root_input", fake_route_root_input)

    async with client.stream(
        "POST",
        "/api/chat/agent-ctx",
        json={"message": "hello"},
    ) as response:
        assert response.status_code == 200
        _ = [line async for line in response.aiter_lines()]

    runtime = _runtime(client)
    assert runtime.session_store is not None
    chat_context = await runtime.session_store.get_chat_context("agent-ctx")
    assert chat_context is not None
    session = await runtime.session_store.get_session(chat_context.current_session_id)
    assert session is not None
    assert session.chat_context_scope_id == "agent-ctx"
    assert session.base_agent_id == "agent-ctx"


@pytest.mark.asyncio
async def test_list_agent_sessions_includes_store_backed_sessions_without_runs(
    client,
) -> None:
    """Session listing should surface session-store records even before any run exists."""
    registry = _runtime(client).agent_registry
    await registry.create_agent(
        AgentConfigRecord(
            id="agent-sessions",
            name="sessions-agent",
            model_provider="openai",
            model_name="gpt-test",
        )
    )

    create_resp = await client.post(
        "/api/chat/agent-sessions/sessions/create",
        json={
            "chat_context_scope_id": "agent-sessions",
            "channel_instance_id": "console-web",
            "user_open_id": "console-user",
        },
    )
    assert create_resp.status_code == 200
    created_session_id = create_resp.json()["session_id"]

    list_resp = await client.get("/api/chat/agent-sessions/sessions")
    assert list_resp.status_code == 200

    data = list_resp.json()
    assert [item["session_id"] for item in data] == [created_session_id]
    assert data[0]["current_task_id"] is None
    assert data[0]["task_message_count"] == 0
