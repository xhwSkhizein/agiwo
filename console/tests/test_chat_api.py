"""Integration tests for session-driven scheduler chat APIs."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.agent import (
    RunCompletedEvent,
    RunStartedEvent,
    StepDelta,
    StepDeltaEvent,
    TerminationReason,
)
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import AgentStateStorageConfig, SchedulerConfig

from server.app import create_app
from server.channels.feishu.store.memory import InMemoryFeishuChannelStore
from server.config import ConsoleConfig
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
    get_console_runtime_from_app,
)
from server.models.session import Session
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.runtime import AgentRuntimeCache
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

    scheduler = Scheduler(
        SchedulerConfig(
            state_storage=AgentStateStorageConfig(storage_type="memory"),
        )
    )
    await scheduler.start()
    session_store = InMemoryFeishuChannelStore()
    await session_store.connect()

    agent_runtime_cache = AgentRuntimeCache(
        scheduler=scheduler,
        agent_registry=registry,
        console_config=config,
        session_store=session_store,
    )

    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_step_storage=run_step_storage,
            trace_storage=trace_storage,
            agent_registry=registry,
            scheduler=scheduler,
            session_store=session_store,
            agent_runtime_cache=agent_runtime_cache,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    clear_console_runtime(app)
    await agent_runtime_cache.close()
    await scheduler.stop()
    await registry.close()
    await run_step_storage.close()


@pytest.mark.asyncio
async def test_create_and_list_agent_sessions_are_base_agent_scoped(client) -> None:
    runtime = _runtime(client)
    await runtime.agent_registry.create_agent(
        AgentConfigRecord(
            id="agent-1",
            name="agent-one",
            model_provider="openai",
            model_name="gpt-test",
        )
    )

    create_resp = await client.post("/api/agents/agent-1/sessions")

    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["session_id"]
    assert created["source_session_id"] is None

    list_resp = await client.get("/api/agents/agent-1/sessions")

    assert list_resp.status_code == 200
    payload = list_resp.json()
    assert payload["items"][0]["session_id"] == created["session_id"]
    assert payload["items"][0]["base_agent_id"] == "agent-1"
    assert payload["items"][0]["chat_context_scope_id"] is None
    assert "runtime_agent_id" not in payload["items"][0]
    assert "scheduler_state_id" not in payload["items"][0]
    assert "current_task_id" not in payload["items"][0]
    assert "task_message_count" not in payload["items"][0]


@pytest.mark.asyncio
async def test_session_input_streams_scheduler_events_and_uses_session_identity(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(client)
    await runtime.agent_registry.create_agent(
        AgentConfigRecord(
            id="agent-1",
            name="agent-one",
            model_provider="openai",
            model_name="gpt-test",
        )
    )
    create_resp = await client.post("/api/agents/agent-1/sessions")
    session_id = create_resp.json()["session_id"]

    class FakeAgent:
        def __init__(self, agent_id: str) -> None:
            self.id = agent_id

        async def close(self) -> None:
            pass

    fake_agent = FakeAgent(session_id)
    assert runtime.agent_runtime_cache is not None
    monkeypatch.setattr(
        runtime.agent_runtime_cache,
        "get_or_create_runtime_agent",
        AsyncMock(return_value=fake_agent),
    )

    scheduler = runtime.scheduler
    assert scheduler is not None

    async def _stream() -> object:
        yield StepDeltaEvent(
            session_id=session_id,
            run_id="run-1",
            agent_id=session_id,
            parent_run_id=None,
            depth=0,
            step_id="step-1",
            delta=StepDelta(content="hello"),
        )
        yield RunCompletedEvent(
            session_id=session_id,
            run_id="run-1",
            agent_id=session_id,
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
        timeout: int | None,
        stream_mode: str,
    ):
        assert agent is fake_agent
        assert message == "hello"
        assert state_id == session_id
        assert session_id == fake_agent.id
        assert persistent is True
        assert timeout == 600
        assert stream_mode == "until_settled"
        return type(
            "RouteResultStub",
            (),
            {
                "action": "submitted",
                "state_id": session_id,
                "stream": _stream(),
            },
        )()

    monkeypatch.setattr(
        scheduler,
        "route_root_input",
        fake_route_root_input,
    )

    async with client.stream(
        "POST",
        f"/api/sessions/{session_id}/input",
        json={"message": "hello"},
    ) as response:
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]

    assert any(line == "event: step_delta" for line in lines)
    assert any(line == "event: run_completed" for line in lines)


@pytest.mark.asyncio
async def test_session_input_continues_stream_after_root_sleeping(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(client)
    await runtime.agent_registry.create_agent(
        AgentConfigRecord(
            id="agent-1",
            name="agent-one",
            model_provider="openai",
            model_name="gpt-test",
        )
    )
    create_resp = await client.post("/api/agents/agent-1/sessions")
    session_id = create_resp.json()["session_id"]

    class FakeAgent:
        def __init__(self, agent_id: str) -> None:
            self.id = agent_id

        async def close(self) -> None:
            pass

    fake_agent = FakeAgent(session_id)
    assert runtime.agent_runtime_cache is not None
    monkeypatch.setattr(
        runtime.agent_runtime_cache,
        "get_or_create_runtime_agent",
        AsyncMock(return_value=fake_agent),
    )

    scheduler = runtime.scheduler
    assert scheduler is not None

    async def _continuous_stream() -> object:
        yield RunStartedEvent(
            session_id=session_id,
            run_id="run-1",
            agent_id=session_id,
            parent_run_id=None,
            depth=0,
        )
        yield StepDeltaEvent(
            session_id=session_id,
            run_id="run-1",
            agent_id=session_id,
            parent_run_id=None,
            depth=0,
            step_id="step-1",
            delta=StepDelta(content="Waiting on child"),
        )
        yield RunCompletedEvent(
            session_id=session_id,
            run_id="run-1",
            agent_id=session_id,
            parent_run_id=None,
            depth=0,
            response="sleep",
            termination_reason=TerminationReason.SLEEPING,
        )
        yield RunStartedEvent(
            session_id=session_id,
            run_id="run-2",
            agent_id=session_id,
            parent_run_id=None,
            depth=0,
        )
        yield StepDeltaEvent(
            session_id=session_id,
            run_id="run-2",
            agent_id=session_id,
            parent_run_id=None,
            depth=0,
            step_id="step-2",
            delta=StepDelta(content="Child done"),
        )
        yield RunCompletedEvent(
            session_id=session_id,
            run_id="run-2",
            agent_id=session_id,
            parent_run_id=None,
            depth=0,
            response="final",
        )

    async def fake_route_root_input(
        message: str,
        *,
        agent,
        state_id: str | None,
        session_id: str,
        persistent: bool,
        timeout: int | None,
        stream_mode: str,
    ):
        assert agent is fake_agent
        assert message == "hello"
        assert state_id == session_id
        assert session_id == fake_agent.id
        assert persistent is True
        assert timeout == 600
        assert stream_mode == "until_settled"
        return type(
            "RouteResultStub",
            (),
            {
                "action": "submitted",
                "state_id": session_id,
                "stream": _continuous_stream(),
            },
        )()

    monkeypatch.setattr(
        scheduler,
        "route_root_input",
        fake_route_root_input,
    )

    async with client.stream(
        "POST",
        f"/api/sessions/{session_id}/input",
        json={"message": "hello"},
    ) as response:
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]

    assert sum(1 for line in lines if line == "event: run_started") == 2
    assert sum(1 for line in lines if line == "event: run_completed") == 2
    assert any("run-2" in line for line in lines)


@pytest.mark.asyncio
async def test_session_input_returns_404_for_missing_session(client) -> None:
    response = await client.post(
        "/api/sessions/missing-session/input",
        json={"message": "hello"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_session_input_returns_404_when_session_base_agent_missing(
    client,
) -> None:
    runtime = _runtime(client)
    assert runtime.session_store is not None
    await runtime.session_store.upsert_session(
        Session(
            id="session-1",
            chat_context_scope_id=None,
            base_agent_id="missing-agent",
            created_by="TEST",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )

    response = await client.post(
        "/api/sessions/session-1/input",
        json={"message": "hello"},
    )
    assert response.status_code == 404
