"""
Integration tests for the Scheduler Chat API endpoints.

Tests the /api/scheduler/chat/* routes using httpx TestClient.
"""

from unittest.mock import AsyncMock

import pytest

from httpx import ASGITransport, AsyncClient

from agiwo.agent import RunCompletedEvent, StepDelta, StepDeltaEvent
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    AgentStateStorageConfig,
    SchedulerConfig,
)
from agiwo.scheduler.scheduler import Scheduler

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
async def client():
    """Create test client with mocked in-memory storage and Scheduler."""
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
    await scheduler.stop()
    await registry.close()
    await run_step_storage.close()


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
        scheduler = _runtime(client).scheduler
        assert scheduler is not None
        state = AgentState(
            id="completed-state",
            session_id="sess-1",
            status=AgentStateStatus.COMPLETED,
            task="Done task",
            result_summary="All done.",
        )
        await scheduler._store.save_state(state)

        resp = await client.post(
            "/api/scheduler/chat/some-agent/cancel",
            json={"state_id": "completed-state"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_running_state(self, client):
        """Cancel succeeds for a running state."""
        scheduler = _runtime(client).scheduler
        assert scheduler is not None
        state = AgentState(
            id="running-state",
            session_id="sess-2",
            status=AgentStateStatus.RUNNING,
            task="Long task",
        )
        await scheduler._store.save_state(state)

        resp = await client.post(
            "/api/scheduler/chat/some-agent/cancel",
            json={"state_id": "running-state"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["state_id"] == "running-state"

        updated = await scheduler.get_state("running-state")
        assert updated is not None
        assert updated.status == AgentStateStatus.FAILED

    @pytest.mark.asyncio
    async def test_cancel_waiting_state(self, client):
        """Cancel succeeds for a waiting state."""
        scheduler = _runtime(client).scheduler
        assert scheduler is not None
        state = AgentState(
            id="sleeping-state",
            session_id="sess-3",
            status=AgentStateStatus.WAITING,
            task="Waiting task",
        )
        await scheduler._store.save_state(state)

        resp = await client.post(
            "/api/scheduler/chat/some-agent/cancel",
            json={"state_id": "sleeping-state"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_cancel_cascades_to_children(self, client):
        """Cancel cascades to active child states."""
        scheduler = _runtime(client).scheduler
        assert scheduler is not None

        parent = AgentState(
            id="parent-cancel",
            session_id="sess-4",
            status=AgentStateStatus.WAITING,
            task="Parent task",
        )
        child1 = AgentState(
            id="child-cancel-1",
            session_id="sess-4",
            status=AgentStateStatus.RUNNING,
            task="Child 1",
            parent_id="parent-cancel",
        )
        child2 = AgentState(
            id="child-cancel-2",
            session_id="sess-4",
            status=AgentStateStatus.COMPLETED,
            task="Child 2",
            parent_id="parent-cancel",
            result_summary="Done",
        )
        for s in [parent, child1, child2]:
            await scheduler._store.save_state(s)

        resp = await client.post(
            "/api/scheduler/chat/some-agent/cancel",
            json={"state_id": "parent-cancel"},
        )
        assert resp.status_code == 200

        parent_state = await scheduler.get_state("parent-cancel")
        assert parent_state is not None
        assert parent_state.status == AgentStateStatus.FAILED

        child1_state = await scheduler.get_state("child-cancel-1")
        assert child1_state is not None
        assert child1_state.status == AgentStateStatus.FAILED

        child2_state = await scheduler.get_state("child-cancel-2")
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

    @pytest.mark.asyncio
    async def test_chat_streams_scheduler_events_and_completion(
        self,
        client,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        registry = _runtime(client).agent_registry
        await registry.create_agent(
            AgentConfigRecord(
                id="agent-1",
                name="scheduler-agent",
                model_provider="openai",
                model_name="gpt-test",
            )
        )

        class FakeStreamingAgent:
            def __init__(self) -> None:
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        fake_agent = FakeStreamingAgent()
        monkeypatch.setattr(
            "server.services.chat_sse.build_agent",
            AsyncMock(return_value=fake_agent),
        )

        scheduler = _runtime(client).scheduler
        assert scheduler is not None

        async def fake_stream(
            message: str,
            *,
            agent,
            session_id: str,
            abort_signal,
            timeout: int,
        ):
            assert agent is fake_agent
            assert message == "hello"
            assert session_id == "session-1"
            assert timeout == 600
            del abort_signal
            yield StepDeltaEvent(
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                parent_run_id=None,
                depth=0,
                step_id="step-1",
                delta=StepDelta(content="hello"),
            )
            yield RunCompletedEvent(
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                parent_run_id=None,
                depth=0,
                response="done",
            )

        monkeypatch.setattr(
            scheduler,
            "stream",
            fake_stream,
        )

        async with client.stream(
            "POST",
            "/api/scheduler/chat/agent-1",
            json={"message": "hello", "session_id": "session-1"},
        ) as response:
            assert response.status_code == 200
            lines = [line async for line in response.aiter_lines()]

        assert any(line == "event: step_delta" for line in lines)
        assert any(line == "event: run_completed" for line in lines)
        assert any('"response": "done"' in line for line in lines)
        assert fake_agent.closed is True
