"""
Integration tests for the Scheduler Chat API endpoints.

Tests the /api/scheduler/chat/* routes using httpx TestClient.
"""

import pytest

from httpx import ASGITransport, AsyncClient

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    AgentStateStorageConfig,
    SchedulerConfig,
)
from agiwo.scheduler.scheduler import Scheduler
from agiwo.scheduler.store import InMemoryAgentStateStorage

from server.app import create_app
from server.dependencies import (
    get_storage_manager,
    set_storage_manager,
    set_console_config,
    set_agent_registry,
    set_scheduler,
    get_scheduler,
)
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry
from server.services.storage_manager import StorageManager


@pytest.fixture
async def client():
    """Create test client with mocked in-memory storage and Scheduler."""
    app = create_app()

    config = ConsoleConfig(storage_type="sqlite", sqlite_db_path=":memory:")
    set_console_config(config)
    sm = StorageManager(config)
    sm.agent_state_storage = InMemoryAgentStateStorage()
    set_storage_manager(sm)

    registry = AgentRegistry(config)
    await registry.initialize()
    set_agent_registry(registry)

    scheduler_config = SchedulerConfig(
        state_storage=AgentStateStorageConfig(storage_type="memory"),
    )
    scheduler = Scheduler(scheduler_config)
    await scheduler.start()
    set_scheduler(scheduler)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    await scheduler.stop()
    await registry.close()
    await sm.close()


class TestSchedulerChatCancel:
    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, client):
        """Cancel returns 404 when state_id doesn't exist."""
        resp = await client.post(
            "/api/scheduler/chat/some-agent/cancel",
            json={"state_id": "nonexistent-state"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_completed_state(self, client):
        """Cancel returns 404 when state is already completed."""
        scheduler = get_scheduler()
        state = AgentState(
            id="completed-state",
            session_id="sess-1",
            agent_id="completed-state",
            parent_agent_id="completed-state",
            status=AgentStateStatus.COMPLETED,
            task="Done task",
            result_summary="All done.",
        )
        await scheduler.store.save_state(state)

        resp = await client.post(
            "/api/scheduler/chat/some-agent/cancel",
            json={"state_id": "completed-state"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_running_state(self, client):
        """Cancel succeeds for a running state."""
        scheduler = get_scheduler()
        state = AgentState(
            id="running-state",
            session_id="sess-2",
            agent_id="running-state",
            parent_agent_id="running-state",
            status=AgentStateStatus.RUNNING,
            task="Long task",
        )
        await scheduler.store.save_state(state)

        resp = await client.post(
            "/api/scheduler/chat/some-agent/cancel",
            json={"state_id": "running-state"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["state_id"] == "running-state"

        updated = await scheduler.store.get_state("running-state")
        assert updated is not None
        assert updated.status == AgentStateStatus.FAILED

    @pytest.mark.asyncio
    async def test_cancel_sleeping_state(self, client):
        """Cancel succeeds for a sleeping state."""
        scheduler = get_scheduler()
        state = AgentState(
            id="sleeping-state",
            session_id="sess-3",
            agent_id="sleeping-state",
            parent_agent_id="sleeping-state",
            status=AgentStateStatus.SLEEPING,
            task="Waiting task",
        )
        await scheduler.store.save_state(state)

        resp = await client.post(
            "/api/scheduler/chat/some-agent/cancel",
            json={"state_id": "sleeping-state"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_cancel_cascades_to_children(self, client):
        """Cancel cascades to active child states."""
        scheduler = get_scheduler()

        parent = AgentState(
            id="parent-cancel",
            session_id="sess-4",
            agent_id="parent-cancel",
            parent_agent_id="parent-cancel",
            status=AgentStateStatus.SLEEPING,
            task="Parent task",
        )
        child1 = AgentState(
            id="child-cancel-1",
            session_id="sess-4",
            agent_id="child-cancel-1",
            parent_agent_id="parent-cancel",
            parent_state_id="parent-cancel",
            status=AgentStateStatus.RUNNING,
            task="Child 1",
        )
        child2 = AgentState(
            id="child-cancel-2",
            session_id="sess-4",
            agent_id="child-cancel-2",
            parent_agent_id="parent-cancel",
            parent_state_id="parent-cancel",
            status=AgentStateStatus.COMPLETED,
            task="Child 2",
            result_summary="Done",
        )
        for s in [parent, child1, child2]:
            await scheduler.store.save_state(s)

        resp = await client.post(
            "/api/scheduler/chat/some-agent/cancel",
            json={"state_id": "parent-cancel"},
        )
        assert resp.status_code == 200

        parent_state = await scheduler.store.get_state("parent-cancel")
        assert parent_state is not None
        assert parent_state.status == AgentStateStatus.FAILED

        child1_state = await scheduler.store.get_state("child-cancel-1")
        assert child1_state is not None
        assert child1_state.status == AgentStateStatus.FAILED

        child2_state = await scheduler.store.get_state("child-cancel-2")
        assert child2_state is not None
        assert child2_state.status == AgentStateStatus.COMPLETED


class TestSchedulerChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_agent_not_found(self, client):
        """Chat returns 404 when agent_id is not registered."""
        resp = await client.post(
            "/api/scheduler/chat/nonexistent-agent",
            json={"message": "hello"},
        )
        assert resp.status_code == 404
