"""Integration tests for session-scoped cancellation via MainAgent."""

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.agent import MainAgentState
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import (
    AgentStateStorageConfig,
    SchedulerConfig,
)

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
from server.services.runtime import AgentRuntimeCache, CachedAgent
from server.services.session_store import InMemorySessionStore
from server.services.storage_wiring import create_run_log_storage, create_trace_storage

from tests.test_agent_runtime_components import FakeMainAgent


def _runtime(client: AsyncClient) -> ConsoleRuntime:
    return get_console_runtime_from_app(client._transport.app)  # type: ignore[attr-defined]


async def _seed_session(
    runtime: ConsoleRuntime,
    *,
    session_id: str,
) -> None:
    assert runtime.session_store is not None
    now = datetime.now(timezone.utc)
    await runtime.session_store.upsert_session(
        Session(
            id=session_id,
            chat_context_scope_id=None,
            base_agent_id="agent-1",
            created_by="TEST",
            created_at=now,
            updated_at=now,
        )
    )


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch):
    async def fake_materialize_main_agent(*_args, **_kwargs):
        return FakeMainAgent("sess-stub")

    monkeypatch.setattr(
        "server.services.runtime.agent_runtime_cache.materialize_main_agent",
        fake_materialize_main_agent,
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


class TestSessionCancel:
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_session_returns_404(self, client) -> None:
        resp = await client.post(
            "/api/sessions/nonexistent-session/cancel",
            json={"reason": "operator stop"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_idle_main_agent_returns_404(self, client) -> None:
        runtime = _runtime(client)
        await runtime.agent_registry.create_agent(
            AgentConfigRecord(
                id="agent-1",
                name="agent-one",
                model_provider="openai",
                model_name="gpt-test",
            )
        )
        await _seed_session(runtime, session_id="session-1")

        resp = await client.post(
            "/api/sessions/session-1/cancel",
            json={"reason": "operator stop"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_running_main_agent_by_session_id(self, client) -> None:
        runtime = _runtime(client)
        assert runtime.agent_runtime_cache is not None
        await runtime.agent_registry.create_agent(
            AgentConfigRecord(
                id="agent-1",
                name="agent-one",
                model_provider="openai",
                model_name="gpt-test",
            )
        )
        await _seed_session(runtime, session_id="session-2")
        running_main_agent = FakeMainAgent(
            "session-2",
            state=MainAgentState.RUNNING,
        )
        runtime.agent_runtime_cache._cache["session-2"] = CachedAgent(
            main_agent=running_main_agent,
            config_snapshot=(
                "agent-one",
                "",
                "openai",
                "gpt-test",
                "",
                None,
                None,
                (),
                (),
                "",
            ),
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
        assert running_main_agent.state is MainAgentState.IDLE

        # After cancel the same MainAgent must accept a new turn (restart).
        restarted = await running_main_agent.accept("after cancel")
        assert restarted is not None
        assert running_main_agent.accept_calls == 1
        assert running_main_agent.state is MainAgentState.RUNNING
