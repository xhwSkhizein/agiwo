"""
Integration tests for the Scheduler API endpoints.

Tests the /api/scheduler/* routes using httpx TestClient.
"""

import pytest
from datetime import datetime, timezone

from httpx import ASGITransport, AsyncClient

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    WaitMode,
    WakeCondition,
    WakeType,
    TimeUnit,
)
from agiwo.scheduler.store import InMemoryAgentStateStorage

from server.app import create_app
from server.dependencies import (
    get_storage_manager,
    set_storage_manager,
    set_console_config,
    set_agent_registry,
)
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry
from server.services.storage_manager import StorageManager


@pytest.fixture
async def client():
    """Create test client with mocked in-memory storage."""
    app = create_app()

    # Manually initialize dependencies with in-memory storage
    config = ConsoleConfig(storage_type="sqlite", sqlite_db_path=":memory:")
    set_console_config(config)
    sm = StorageManager(config)
    # Override agent_state_storage with in-memory for isolation
    sm.agent_state_storage = InMemoryAgentStateStorage()
    set_storage_manager(sm)

    registry = AgentRegistry(config)
    await registry.initialize()
    set_agent_registry(registry)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await registry.close()
    await sm.close()


async def _seed_states(client: AsyncClient) -> None:
    """Seed agent states into the in-memory storage."""
    storage = get_storage_manager().agent_state_storage

    states = [
        AgentState(
            id="parent-1",
            session_id="sess-1",
            agent_id="parent-1",
            parent_agent_id="parent-1",
            status=AgentStateStatus.SLEEPING,
            task="Orchestrate research",
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=["child-1", "child-2"],
                completed_ids=["child-1"],
            ),
        ),
        AgentState(
            id="child-1",
            session_id="sess-1",
            agent_id="child-1",
            parent_agent_id="parent-1",
            parent_state_id="parent-1",
            status=AgentStateStatus.COMPLETED,
            task="Research topic A",
            result_summary="Topic A is about X.",
            signal_propagated=True,
        ),
        AgentState(
            id="child-2",
            session_id="sess-1",
            agent_id="child-2",
            parent_agent_id="parent-1",
            parent_state_id="parent-1",
            status=AgentStateStatus.RUNNING,
            task="Research topic B",
        ),
        AgentState(
            id="delayed-1",
            session_id="sess-2",
            agent_id="delayed-1",
            parent_agent_id="delayed-1",
            status=AgentStateStatus.SLEEPING,
            task="Wait and retry",
            wake_condition=WakeCondition(
                type=WakeType.TIMER,
                time_value=30,
                time_unit=TimeUnit.MINUTES,
                wakeup_at=datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
            ),
        ),
        AgentState(
            id="failed-1",
            session_id="sess-3",
            agent_id="failed-1",
            parent_agent_id="failed-1",
            status=AgentStateStatus.FAILED,
            task="Broken task",
        ),
    ]
    for s in states:
        await storage.save_state(s)


class TestListAgentStates:
    @pytest.mark.asyncio
    async def test_list_empty(self, client):
        resp = await client.get("/api/scheduler/states")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_all(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5
        assert all("id" in item for item in data)

    @pytest.mark.asyncio
    async def test_filter_by_status(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states?status=sleeping")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(item["status"] == "sleeping" for item in data)

    @pytest.mark.asyncio
    async def test_filter_invalid_status(self, client):
        resp = await client.get("/api/scheduler/states?status=invalid")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states?limit=2&offset=0")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

        resp2 = await client.get("/api/scheduler/states?limit=2&offset=2")
        assert resp2.status_code == 200
        assert len(resp2.json()) == 2


class TestGetAgentState:
    @pytest.mark.asyncio
    async def test_get_existing(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states/parent-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "parent-1"
        assert data["status"] == "sleeping"
        assert data["task"] == "Orchestrate research"
        assert data["wake_condition"] is not None
        assert data["wake_condition"]["type"] == "waitset"
        assert data["wake_condition"]["wait_for"] == ["child-1", "child-2"]
        assert data["wake_condition"]["completed_ids"] == ["child-1"]

    @pytest.mark.asyncio
    async def test_get_with_delay_wake(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states/delayed-1")
        assert resp.status_code == 200
        data = resp.json()
        wc = data["wake_condition"]
        assert wc["type"] == "timer"
        assert wc["time_value"] == 30
        assert wc["time_unit"] == "minutes"
        assert wc["wakeup_at"] is not None

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, client):
        resp = await client.get("/api/scheduler/states/nonexistent")
        assert resp.status_code == 404


class TestGetChildren:
    @pytest.mark.asyncio
    async def test_get_children(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states/parent-1/children")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        ids = {item["id"] for item in data}
        assert ids == {"child-1", "child-2"}

    @pytest.mark.asyncio
    async def test_get_children_empty(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states/child-1/children")
        assert resp.status_code == 200
        assert resp.json() == []


class TestSchedulerStats:
    @pytest.mark.asyncio
    async def test_stats_empty(self, client):
        resp = await client.get("/api/scheduler/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["pending"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_data(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert data["sleeping"] == 2
        assert data["running"] == 1
        assert data["completed"] == 1
        assert data["failed"] == 1
        assert data["pending"] == 0
