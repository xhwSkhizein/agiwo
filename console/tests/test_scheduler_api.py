"""
Integration tests for the Scheduler API endpoints.

Tests the /api/scheduler/* routes using httpx TestClient.
"""

import os
import shutil

import pytest
from datetime import datetime, timezone

from httpx import ASGITransport, AsyncClient

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    AgentStateStorageConfig,
    PendingEvent,
    SchedulerRunResult,
    SchedulerConfig,
    SchedulerEventType,
    WakeCondition,
    WakeType,
    TimeUnit,
)
from agiwo.agent import TerminationReason, UserMessage
from agiwo.scheduler.engine import Scheduler

from server.app import create_app
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
    get_console_runtime_from_app,
)
from server.config import ConsoleConfig
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.storage_wiring import create_run_step_storage, create_trace_storage


def _runtime(client: AsyncClient) -> ConsoleRuntime:
    return get_console_runtime_from_app(client._transport.app)  # type: ignore[attr-defined]


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch):
    """Create test client with mocked in-memory storage."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    app = create_app()

    # Manually initialize dependencies with in-memory storage
    config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
    )
    run_step_storage = create_run_step_storage(config)
    trace_storage = create_trace_storage(config)
    scheduler = Scheduler(
        SchedulerConfig(
            state_storage=AgentStateStorageConfig(storage_type="memory"),
        )
    )
    await scheduler.start()

    registry = AgentRegistry(config)
    await registry.initialize()
    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_step_storage=run_step_storage,
            trace_storage=trace_storage,
            agent_registry=registry,
            scheduler=scheduler,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    clear_console_runtime(app)
    await registry.close()
    await scheduler.stop()
    await run_step_storage.close()

    # Clean up agent workspace directories created during tests
    _cleanup_agent_workspaces()


async def _seed_states(client: AsyncClient) -> None:
    """Seed agent states into the in-memory storage."""
    scheduler = _runtime(client).scheduler
    assert scheduler is not None
    storage = scheduler._store

    states = [
        AgentState(
            id="parent-1",
            session_id="sess-1",
            status=AgentStateStatus.WAITING,
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
            status=AgentStateStatus.COMPLETED,
            task="Research topic A",
            parent_id="parent-1",
            result_summary="Topic A is about X.",
            last_run_result=SchedulerRunResult(
                run_id="run-child-1",
                termination_reason=TerminationReason.COMPLETED,
                summary="Topic A is about X.",
            ),
            signal_propagated=True,
        ),
        AgentState(
            id="child-2",
            session_id="sess-1",
            status=AgentStateStatus.RUNNING,
            task="Research topic B",
            parent_id="parent-1",
        ),
        AgentState(
            id="delayed-1",
            session_id="sess-2",
            status=AgentStateStatus.WAITING,
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
            status=AgentStateStatus.FAILED,
            task="Broken task",
        ),
    ]
    for s in states:
        await storage.save_state(s)


async def _seed_tree_states(client: AsyncClient) -> None:
    scheduler = _runtime(client).scheduler
    assert scheduler is not None
    storage = scheduler._store

    base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    states = [
        AgentState(
            id="root-tree",
            session_id="sess-tree",
            status=AgentStateStatus.WAITING,
            task="Root orchestration",
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=["child-running", "child-cancelled", "child-failed"],
                completed_ids=["child-cancelled"],
            ),
            created_at=base_time,
            updated_at=base_time,
        ),
        AgentState(
            id="child-running",
            session_id="sess-tree",
            status=AgentStateStatus.RUNNING,
            task="Active child",
            parent_id="root-tree",
            depth=1,
            created_at=base_time.replace(minute=1),
            updated_at=base_time.replace(minute=1),
        ),
        AgentState(
            id="child-cancelled",
            session_id="sess-tree",
            status=AgentStateStatus.FAILED,
            task="Cancelled child",
            parent_id="root-tree",
            depth=1,
            result_summary="Cancelled by operator",
            last_run_result=SchedulerRunResult(
                run_id="run-cancelled",
                termination_reason=TerminationReason.CANCELLED,
                summary="Cancelled by operator",
                completed_at=base_time.replace(minute=2),
            ),
            created_at=base_time.replace(minute=2),
            updated_at=base_time.replace(minute=2),
        ),
        AgentState(
            id="child-failed",
            session_id="sess-tree",
            status=AgentStateStatus.FAILED,
            task="Broken child",
            parent_id="root-tree",
            depth=1,
            result_summary="Tool crashed",
            last_run_result=SchedulerRunResult(
                run_id="run-failed",
                termination_reason=TerminationReason.ERROR,
                summary="Tool crashed",
                error="Tool crashed",
                completed_at=base_time.replace(minute=3),
            ),
            created_at=base_time.replace(minute=3),
            updated_at=base_time.replace(minute=3),
        ),
        AgentState(
            id="grandchild-completed",
            session_id="sess-tree",
            status=AgentStateStatus.COMPLETED,
            task="Grandchild result",
            parent_id="child-running",
            depth=2,
            result_summary="Grandchild finished",
            last_run_result=SchedulerRunResult(
                run_id="run-grandchild",
                termination_reason=TerminationReason.COMPLETED,
                summary="Grandchild finished",
                completed_at=base_time.replace(minute=4),
            ),
            created_at=base_time.replace(minute=4),
            updated_at=base_time.replace(minute=4),
        ),
    ]
    for state in states:
        await storage.save_state(state)

    events = [
        PendingEvent(
            id="evt-root-1",
            target_agent_id="root-tree",
            session_id="sess-tree",
            event_type=SchedulerEventType.USER_HINT,
            payload={"user_input": UserMessage.to_storage_value("Root hint")},
            created_at=base_time.replace(minute=10),
        ),
        PendingEvent(
            id="evt-child-1",
            target_agent_id="child-running",
            session_id="sess-tree",
            event_type=SchedulerEventType.CHILD_COMPLETED,
            payload={"result": "Result A"},
            created_at=base_time.replace(minute=11),
            source_agent_id="grandchild-completed",
        ),
        PendingEvent(
            id="evt-child-2",
            target_agent_id="child-running",
            session_id="sess-tree",
            event_type=SchedulerEventType.USER_HINT,
            payload={"user_input": UserMessage.to_storage_value("Second hint")},
            created_at=base_time.replace(minute=12),
        ),
        PendingEvent(
            id="evt-grandchild-1",
            target_agent_id="grandchild-completed",
            session_id="sess-tree",
            event_type=SchedulerEventType.USER_HINT,
            payload={"user_input": UserMessage.to_storage_value("Nested hint")},
            created_at=base_time.replace(minute=13),
        ),
    ]
    for event in events:
        await storage.save_event(event)


class TestListAgentStates:
    @pytest.mark.asyncio
    async def test_list_empty(self, client):
        resp = await client.get("/api/scheduler/states")
        assert resp.status_code == 200
        assert resp.json() == {
            "items": [],
            "limit": 50,
            "offset": 0,
            "has_more": False,
            "total": None,
        }

    @pytest.mark.asyncio
    async def test_list_all_states(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 5
        assert all("id" in item for item in data["items"])

    @pytest.mark.asyncio
    async def test_filter_by_status(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states?status=waiting")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        assert all(item["status"] == "waiting" for item in data["items"])

    @pytest.mark.asyncio
    async def test_filter_invalid_status(self, client):
        resp = await client.get("/api/scheduler/states?status=invalid")
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states?limit=2&offset=0")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) == 2
        assert resp.json()["has_more"] is True

        resp2 = await client.get("/api/scheduler/states?limit=2&offset=2")
        assert resp2.status_code == 200
        assert len(resp2.json()["items"]) == 2


class TestGetAgentState:
    @pytest.mark.asyncio
    async def test_get_existing(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states/parent-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "parent-1"
        assert data["status"] == "waiting"
        assert data["task"] == "Orchestrate research"
        assert data["wake_condition"] is not None
        assert data["wake_condition"]["type"] == "waitset"
        assert data["wake_condition"]["wait_for"] == ["child-1", "child-2"]
        assert data["wake_condition"]["completed_ids"] == ["child-1"]
        assert data["last_run_result"] is None

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
    async def test_get_child_includes_root_state_id(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states/child-2")
        assert resp.status_code == 200
        assert resp.json()["root_state_id"] == "parent-1"

    @pytest.mark.asyncio
    async def test_get_completed_child_exposes_last_run_result(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/states/child-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["last_run_result"]["run_id"] == "run-child-1"
        assert data["last_run_result"]["termination_reason"] == "completed"

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


class TestSchedulerTree:
    @pytest.mark.asyncio
    async def test_get_tree_returns_parent_before_child_and_pending_counts(
        self, client
    ):
        await _seed_tree_states(client)

        resp = await client.get("/api/scheduler/states/root-tree/tree")

        assert resp.status_code == 200
        data = resp.json()
        assert data["root_state_id"] == "root-tree"
        assert data["root_session_id"] == "sess-tree"
        assert data["stats"] == {
            "total": 5,
            "running": 1,
            "waiting": 1,
            "queued": 0,
            "idle": 0,
            "completed": 1,
            "failed": 1,
            "cancelled": 1,
        }

        ids = [node["state_id"] for node in data["nodes"]]
        assert ids.index("root-tree") < ids.index("child-running")
        assert ids.index("root-tree") < ids.index("child-cancelled")
        assert ids.index("child-running") < ids.index("grandchild-completed")

        node_map = {node["state_id"]: node for node in data["nodes"]}
        assert node_map["root-tree"]["child_ids"] == [
            "child-running",
            "child-cancelled",
            "child-failed",
        ]
        assert node_map["root-tree"]["pending_event_count"] == 1
        assert node_map["child-running"]["pending_event_count"] == 2
        assert node_map["grandchild-completed"]["pending_event_count"] == 1
        assert node_map["child-cancelled"]["root_state_id"] == "root-tree"
        assert node_map["child-cancelled"]["last_error"] == "Cancelled by operator"
        assert (
            node_map["child-cancelled"]["last_run_result"]["termination_reason"]
            == "cancelled"
        )
        assert node_map["child-failed"]["last_run_result"]["error"] == "Tool crashed"
        assert "pending_events" not in node_map["child-running"]

    @pytest.mark.asyncio
    async def test_get_tree_requires_root_state(self, client):
        await _seed_tree_states(client)

        resp = await client.get("/api/scheduler/states/child-running/tree")

        assert resp.status_code == 422
        assert "root" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_tree_not_found(self, client):
        resp = await client.get("/api/scheduler/states/missing/tree")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_tree_too_large_returns_422(self, client, monkeypatch):
        await _seed_tree_states(client)
        monkeypatch.setattr("server.routers.scheduler.SCHEDULER_TREE_MAX_NODES", 3)

        resp = await client.get("/api/scheduler/states/root-tree/tree")

        assert resp.status_code == 422
        assert "tree too large" in resp.json()["detail"].lower()


class TestSchedulerStats:
    @pytest.mark.asyncio
    async def test_stats_empty(self, client):
        resp = await client.get("/api/scheduler/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["pending"] == 0


class TestPersistentAgentEndpoints:
    @pytest.mark.asyncio
    async def test_create_persistent_agent_generates_unique_state_ids_without_session_id(
        self, client
    ):
        registry = _runtime(client).agent_registry
        config = await registry.create_agent(
            AgentConfigRecord(
                id="agent-config-1",
                name="persistent-agent",
                model_provider="openai",
                model_name="gpt-test",
            )
        )

        first = await client.post(
            "/api/scheduler/states/create",
            json={"agent_config_id": config.id},
        )
        second = await client.post(
            "/api/scheduler/states/create",
            json={"agent_config_id": config.id},
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["state_id"] != second.json()["state_id"]

    @pytest.mark.asyncio
    async def test_resume_endpoint_rehydrates_missing_runtime_agent(self, client):
        registry = _runtime(client).agent_registry
        config = await registry.create_agent(
            AgentConfigRecord(
                id="agent-config-2",
                name="persistent-agent",
                model_provider="openai",
                model_name="gpt-test",
            )
        )
        scheduler = _runtime(client).scheduler
        assert scheduler is not None
        state = AgentState(
            id="agent-config-2--resume",
            session_id="sess-resume",
            status=AgentStateStatus.IDLE,
            task="Initial task",
            agent_config_id=config.id,
            is_persistent=True,
        )
        await scheduler._store.save_state(state)

        response = await client.post(
            f"/api/scheduler/states/{state.id}/resume",
            json={"message": "Resume work"},
        )

        assert response.status_code == 200
        assert scheduler.get_registered_agent(state.id) is not None

        updated = await scheduler.get_state(state.id)
        assert updated is not None
        assert updated.pending_input == "Resume work"
        assert updated.status == AgentStateStatus.QUEUED

    @pytest.mark.asyncio
    async def test_stats_with_data(self, client):
        await _seed_states(client)
        resp = await client.get("/api/scheduler/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert data["waiting"] == 2
        assert data["idle"] == 0
        assert data["queued"] == 0
        assert data["running"] == 1
        assert data["completed"] == 1
        assert data["failed"] == 1
        assert data["pending"] == 0


def _cleanup_agent_workspaces() -> None:
    """Remove agent workspace directories created by tests."""
    # Default root path is .agiwo (relative to cwd)
    root_path = os.path.expanduser(".agiwo")
    # Agent names used in tests that create workspace directories
    test_agent_names = ["persistent-agent"]
    for agent_name in test_agent_names:
        workspace_dir = os.path.join(root_path, agent_name)
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir, ignore_errors=True)
