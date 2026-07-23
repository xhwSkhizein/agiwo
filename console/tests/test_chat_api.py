"""Integration tests for session-driven chat APIs."""

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import (
    AgentStateStorageConfig,
    SchedulerConfig,
)

from server.channels.exceptions import BaseAgentNotFoundError
from server.app import create_app
from server.services.session_store import InMemorySessionStore
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
from server.services.storage_wiring import (
    create_run_log_storage,
    create_trace_storage,
)
from server.services.session_gateway import SessionGateway, SessionStreamStart
from tests.test_agent_runtime_components import FakeMainAgent


def _runtime(client: AsyncClient) -> ConsoleRuntime:
    return get_console_runtime_from_app(client._transport.app)  # type: ignore[attr-defined]


async def _stub_start_user_message(self, session_id, user_message, *, idempotency_key):
    del self, user_message, idempotency_key
    return SessionStreamStart(
        session_id=session_id,
        run_id=f"run_{session_id}",
        main_agent=FakeMainAgent(session_id),
    )


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        SessionGateway,
        "start_user_message",
        _stub_start_user_message,
    )
    app = create_app()

    config = ConsoleConfig(
        storage={
            "run_log_type": "memory",
            "trace_type": "memory",
            "metadata_type": "memory",
        }
    )
    run_log_storage = create_run_log_storage(config)
    trace_storage = create_trace_storage(config)
    registry = AgentRegistry(config)
    await registry.initialize()

    scheduler = Scheduler(
        SchedulerConfig(
            state_storage=AgentStateStorageConfig(storage_type="memory"),
        )
    )
    await scheduler.start()
    session_store = InMemorySessionStore()
    await session_store.connect()

    agent_runtime_cache = AgentRuntimeCache(
        agent_registry=registry,
        console_config=config,
        session_store=session_store,
    )

    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_log_storage=run_log_storage,
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
    await run_log_storage.close()
    await trace_storage.close()
    await session_store.close()


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
async def test_session_input_streams_plain_turn(client) -> None:
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

    async with client.stream(
        "POST",
        f"/api/sessions/{session_id}/input",
        json={"message": "hello"},
        headers={"Idempotency-Key": "chat-1"},
        timeout=5.0,
    ) as response:
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]

    assert any(line == "event: session_turn" for line in lines)
    assert any("stub reply" in line for line in lines if line.startswith("data:"))
    assert any(
        '"kind": "session"' in line for line in lines if line.startswith("data:")
    )


@pytest.mark.asyncio
async def test_session_input_continues_as_plain_turns(client) -> None:
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

    async with client.stream(
        "POST",
        f"/api/sessions/{session_id}/input",
        json={"message": "first"},
        headers={"Idempotency-Key": "chat-a"},
        timeout=5.0,
    ) as response:
        assert response.status_code == 200
        first_lines = [line async for line in response.aiter_lines()]

    async with client.stream(
        "POST",
        f"/api/sessions/{session_id}/input",
        json={"message": "second"},
        headers={"Idempotency-Key": "chat-b"},
        timeout=5.0,
    ) as response:
        assert response.status_code == 200
        second_lines = [line async for line in response.aiter_lines()]

    assert any(line == "event: session_turn" for line in first_lines)
    assert any(line == "event: session_turn" for line in second_lines)


@pytest.mark.asyncio
async def test_session_input_returns_404_for_missing_session(client) -> None:
    response = await client.post(
        "/api/sessions/missing-session/input",
        json={"message": "hello"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_session_input_accepts_session_even_if_base_agent_missing(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway owns the turn; missing agent fails later at dispatch, not at accept."""

    async def _raise_missing(self, session_id, user_message, *, idempotency_key):
        del self, session_id, user_message, idempotency_key
        raise BaseAgentNotFoundError("missing-agent")

    monkeypatch.setattr(
        SessionGateway,
        "start_user_message",
        _raise_missing,
    )
    runtime = _runtime(client)
    assert runtime.session_store is not None
    await runtime.session_store.upsert_session(
        Session(
            id="session-1",
            chat_context_scope_id=None,
            base_agent_id="missing-agent",
            created_by="test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )

    async with client.stream(
        "POST",
        "/api/sessions/session-1/input",
        json={"message": "hello"},
        headers={"Idempotency-Key": "orphan-agent"},
        timeout=5.0,
    ) as response:
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines()]
    assert any(line == "event: session_error" for line in lines)
