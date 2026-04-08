"""Integration tests for session-scoped cancellation."""

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    AgentStateStorageConfig,
    SchedulerConfig,
)

from server.app import create_app
from server.services.session_store import InMemorySessionStore
from server.config import ConsoleConfig
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
    get_console_runtime_from_app,
)
from server.services.agent_registry import AgentRegistry
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
    session_store = InMemorySessionStore()
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


class TestSessionCancel:
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_session_state(self, client) -> None:
        resp = await client.post(
            "/api/sessions/nonexistent-session/cancel",
            json={"reason": "operator stop"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_completed_state_returns_404(self, client) -> None:
        scheduler = _runtime(client).scheduler
        assert scheduler is not None
        await scheduler._store.save_state(
            AgentState(
                id="session-1",
                session_id="session-1",
                status=AgentStateStatus.COMPLETED,
                task="Done task",
                result_summary="All done.",
            )
        )

        resp = await client.post(
            "/api/sessions/session-1/cancel",
            json={"reason": "operator stop"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_running_root_state_by_session_id(self, client) -> None:
        scheduler = _runtime(client).scheduler
        assert scheduler is not None
        await scheduler._store.save_state(
            AgentState(
                id="session-2",
                session_id="session-2",
                status=AgentStateStatus.RUNNING,
                task="Long task",
            )
        )

        resp = await client.post(
            "/api/sessions/session-2/cancel",
            json={"reason": "operator stop"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["session_id"] == "session-2"
        assert data["state_id"] == "session-2"

        updated = await scheduler.get_state("session-2")
        assert updated is not None
        assert updated.status == AgentStateStatus.FAILED

    @pytest.mark.asyncio
    async def test_cancel_cascades_to_children(self, client) -> None:
        scheduler = _runtime(client).scheduler
        assert scheduler is not None

        states = [
            AgentState(
                id="session-3",
                session_id="session-3",
                status=AgentStateStatus.WAITING,
                task="Parent task",
            ),
            AgentState(
                id="child-cancel-1",
                session_id="session-3",
                status=AgentStateStatus.RUNNING,
                task="Child 1",
                parent_id="session-3",
            ),
            AgentState(
                id="child-cancel-2",
                session_id="session-3",
                status=AgentStateStatus.COMPLETED,
                task="Child 2",
                parent_id="session-3",
                result_summary="Done",
            ),
        ]
        for state in states:
            await scheduler._store.save_state(state)

        resp = await client.post(
            "/api/sessions/session-3/cancel",
            json={"reason": "operator stop"},
        )
        assert resp.status_code == 200

        parent_state = await scheduler.get_state("session-3")
        assert parent_state is not None
        assert parent_state.status == AgentStateStatus.FAILED

        child1 = await scheduler.get_state("child-cancel-1")
        assert child1 is not None
        assert child1.status == AgentStateStatus.FAILED

        child2 = await scheduler.get_state("child-cancel-2")
        assert child2 is not None
        assert child2.status == AgentStateStatus.COMPLETED
