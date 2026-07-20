"""Session archive / restore tests (P5-06)."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.objective import CommandResult, ObjectiveStatus
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import AgentStateStorageConfig, SchedulerConfig

from server.app import create_app
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
from server.services.session_store import InMemorySessionStore
from server.services.storage_wiring import (
    create_objective_store,
    create_run_log_storage,
    create_trace_storage,
)
from tests.objective_test_support import build_test_objective_service


def _runtime(client: AsyncClient) -> ConsoleRuntime:
    return get_console_runtime_from_app(client._transport.app)  # type: ignore[attr-defined]


@pytest.fixture
async def client():
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
    objective_store = create_objective_store(config)
    registry = AgentRegistry(config)
    await registry.initialize()
    scheduler = Scheduler(
        SchedulerConfig(state_storage=AgentStateStorageConfig(storage_type="memory"))
    )
    await scheduler.start()
    session_store = InMemorySessionStore()
    await session_store.connect()
    agent_runtime_cache = AgentRuntimeCache(
        scheduler=scheduler,
        agent_registry=registry,
        console_config=config,
        session_store=session_store,
    )
    test_agent, objective_service = build_test_objective_service(
        objective_store,
        scheduler,
    )
    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_log_storage=run_log_storage,
            trace_storage=trace_storage,
            objective_store=objective_store,
            agent_registry=registry,
            objective_service=objective_service,
            scheduler=scheduler,
            session_store=session_store,
            agent_runtime_cache=agent_runtime_cache,
        ),
    )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    clear_console_runtime(app)
    await test_agent.close()
    await agent_runtime_cache.close()
    await scheduler.stop()
    await registry.close()
    await run_log_storage.close()
    await trace_storage.close()
    close = getattr(objective_store, "close", None)
    if close is not None:
        await close()
    await session_store.close()


@pytest.mark.asyncio
async def test_archive_and_restore_idle_session(client) -> None:
    runtime = _runtime(client)
    await runtime.agent_registry.create_agent(
        AgentConfigRecord(
            id="agent-1",
            name="agent-one",
            model_provider="openai",
            model_name="gpt-test",
        )
    )
    now = datetime.now(timezone.utc)
    await runtime.session_store.upsert_session(
        Session(
            id="sess-arch",
            chat_context_scope_id=None,
            base_agent_id="agent-1",
            created_by="TEST",
            created_at=now,
            updated_at=now,
        )
    )

    listed = await client.get("/api/sessions")
    assert any(item["session_id"] == "sess-arch" for item in listed.json()["items"])

    archive = await client.post("/api/sessions/sess-arch/archive")
    assert archive.status_code == 200
    assert archive.json()["archived_at"]

    listed_after = await client.get("/api/sessions")
    assert not any(
        item["session_id"] == "sess-arch" for item in listed_after.json()["items"]
    )

    with_archived = await client.get("/api/sessions", params={"include_archived": True})
    assert any(
        item["session_id"] == "sess-arch" for item in with_archived.json()["items"]
    )

    restore = await client.post("/api/sessions/sess-arch/restore")
    assert restore.status_code == 200
    listed_restored = await client.get("/api/sessions")
    assert any(
        item["session_id"] == "sess-arch" for item in listed_restored.json()["items"]
    )


@pytest.mark.asyncio
async def test_archive_pauses_active_objective_first(client, monkeypatch) -> None:
    runtime = _runtime(client)
    service = runtime.objective_service
    assert service is not None
    await runtime.agent_registry.create_agent(
        AgentConfigRecord(
            id="agent-1",
            name="agent-one",
            model_provider="openai",
            model_name="gpt-test",
        )
    )
    now = datetime.now(timezone.utc)
    await runtime.session_store.upsert_session(
        Session(
            id="sess-active",
            chat_context_scope_id=None,
            base_agent_id="agent-1",
            created_by="TEST",
            created_at=now,
            updated_at=now,
        )
    )
    create = await client.post(
        "/api/objectives",
        headers={"Idempotency-Key": "arch-obj"},
        json={
            "session_id": "sess-active",
            "message": "busy",
            "budget": {
                "handoffs": 5,
                "verification_attempts": 3,
                "llm_cost_usd": 2.0,
                "active_seconds": 600,
            },
        },
    )
    objective_id = create.json()["objective_id"]

    pause_mock = AsyncMock(
        return_value=CommandResult(
            objective_id=objective_id,
            status=ObjectiveStatus.USER_PAUSED.value,
        )
    )
    monkeypatch.setattr(service, "pause", pause_mock)
    monkeypatch.setattr(
        service,
        "get_view",
        AsyncMock(
            return_value=SimpleNamespace(
                objective_id=objective_id,
                is_terminal=False,
                status=ObjectiveStatus.USER_PAUSED,
            )
        ),
    )

    archive = await client.post("/api/sessions/sess-active/archive")
    assert archive.status_code == 200
    pause_mock.assert_awaited_once()
    session = await runtime.session_store.get_session("sess-active")
    assert session is not None
    assert session.archived_at is not None
