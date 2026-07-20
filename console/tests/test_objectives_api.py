"""Objective gateway HTTP API tests (P5-01)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.objective import (
    CommandResult,
    InvariantViolation,
    ObjectiveStatus,
)
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


async def _seed_session(client: AsyncClient) -> str:
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
    session = Session(
        id="sess-obj-1",
        chat_context_scope_id="scope-1",
        base_agent_id="agent-1",
        created_by="TEST",
        created_at=now,
        updated_at=now,
    )
    await runtime.session_store.upsert_session(session)
    return session.id


@pytest.mark.asyncio
async def test_create_objective_requires_idempotency_key(client) -> None:
    session_id = await _seed_session(client)
    resp = await client.post(
        "/api/objectives",
        json={"session_id": session_id, "message": "hello"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_create_get_and_list_objectives(client) -> None:
    session_id = await _seed_session(client)
    create = await client.post(
        "/api/objectives",
        headers={"Idempotency-Key": "create-1"},
        json={
            "session_id": session_id,
            "message": "plan my week",
            "budget": {
                "handoffs": 5,
                "verification_attempts": 3,
                "llm_cost_usd": 2.0,
                "active_seconds": 600,
            },
        },
    )
    assert create.status_code == 201, create.text
    body = create.json()
    objective_id = body["objective_id"]
    assert body["replayed"] is False

    get_resp = await client.get(f"/api/objectives/{objective_id}")
    assert get_resp.status_code == 200
    view = get_resp.json()
    assert view["session_id"] == session_id
    assert view["status"] in {s.value for s in ObjectiveStatus}
    assert view["budget"]["handoffs"]["limit"] == 5

    listed = await client.get(f"/api/sessions/{session_id}/objectives")
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert listed.json()[0]["objective_id"] == objective_id


@pytest.mark.asyncio
async def test_create_objective_idempotent_replay(client) -> None:
    session_id = await _seed_session(client)
    payload = {
        "session_id": session_id,
        "message": "same request",
        "budget": {
            "handoffs": 5,
            "verification_attempts": 3,
            "llm_cost_usd": 2.0,
            "active_seconds": 600,
        },
    }
    first = await client.post(
        "/api/objectives",
        headers={"Idempotency-Key": "idem-1"},
        json=payload,
    )
    second = await client.post(
        "/api/objectives",
        headers={"Idempotency-Key": "idem-1"},
        json=payload,
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["objective_id"] == second.json()["objective_id"]
    assert second.json()["replayed"] is True


@pytest.mark.asyncio
async def test_idempotency_conflict_on_different_body(client) -> None:
    session_id = await _seed_session(client)
    budget = {
        "handoffs": 5,
        "verification_attempts": 3,
        "llm_cost_usd": 2.0,
        "active_seconds": 600,
    }
    first = await client.post(
        "/api/objectives",
        headers={"Idempotency-Key": "conflict-1"},
        json={"session_id": session_id, "message": "a", "budget": budget},
    )
    assert first.status_code == 201
    second = await client.post(
        "/api/objectives",
        headers={"Idempotency-Key": "conflict-1"},
        json={"session_id": session_id, "message": "b", "budget": budget},
    )
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_second_active_objective_rejected(client, monkeypatch) -> None:
    session_id = await _seed_session(client)
    runtime = _runtime(client)
    service = runtime.objective_service
    assert service is not None

    async def fake_create(*_args, **_kwargs):
        raise InvariantViolation("session already has an active objective")

    monkeypatch.setattr(service, "create_objective", fake_create)
    resp = await client.post(
        "/api/objectives",
        headers={"Idempotency-Key": "slot-1"},
        json={
            "session_id": session_id,
            "message": "hello",
            "budget": {
                "handoffs": 5,
                "verification_attempts": 3,
                "llm_cost_usd": 2.0,
                "active_seconds": 600,
            },
        },
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_pause_resume_via_api(client, monkeypatch) -> None:
    session_id = await _seed_session(client)
    runtime = _runtime(client)
    service = runtime.objective_service
    assert service is not None

    create = await client.post(
        "/api/objectives",
        headers={"Idempotency-Key": "pause-create"},
        json={
            "session_id": session_id,
            "message": "work",
            "budget": {
                "handoffs": 5,
                "verification_attempts": 3,
                "llm_cost_usd": 2.0,
                "active_seconds": 600,
            },
        },
    )
    objective_id = create.json()["objective_id"]

    monkeypatch.setattr(
        service,
        "pause",
        AsyncMock(
            return_value=CommandResult(
                objective_id=objective_id,
                status=ObjectiveStatus.USER_PAUSED.value,
            )
        ),
    )
    monkeypatch.setattr(
        service,
        "resume",
        AsyncMock(
            return_value=CommandResult(
                objective_id=objective_id,
                status=ObjectiveStatus.RUNNING.value,
            )
        ),
    )

    pause = await client.post(
        f"/api/objectives/{objective_id}/pause",
        headers={"Idempotency-Key": "pause-1"},
        json={"reason": "user_pause"},
    )
    assert pause.status_code == 200
    assert pause.json()["status"] == ObjectiveStatus.USER_PAUSED.value

    resume = await client.post(
        f"/api/objectives/{objective_id}/resume",
        headers={"Idempotency-Key": "resume-1"},
        json={"reason": "user_resume"},
    )
    assert resume.status_code == 200
    assert resume.json()["status"] == ObjectiveStatus.RUNNING.value
