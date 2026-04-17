"""Smoke test: create_app() succeeds and basic health endpoint responds."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import server.app as app_module
from server.app import create_app
from server.config import ConsoleConfig
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
)
from server.services.agent_registry import AgentRegistry
from server.services.storage_wiring import create_run_step_storage, create_trace_storage


@pytest.mark.asyncio
async def test_create_app_starts_and_lists_agents() -> None:
    """Smoke test: app creation, runtime binding, and a basic API call all succeed."""
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

    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_step_storage=run_step_storage,
            trace_storage=trace_storage,
            agent_registry=registry,
        ),
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/agents")
        assert resp.status_code == 200

    clear_console_runtime(app)
    await registry.close()
    await run_step_storage.close()


@pytest.mark.asyncio
async def test_lifespan_closes_session_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
    )
    run_step_storage = SimpleNamespace(close=AsyncMock())
    trace_storage = SimpleNamespace(close=AsyncMock())
    session_store = SimpleNamespace(connect=AsyncMock(), close=AsyncMock())
    agent_registry = SimpleNamespace(
        initialize=AsyncMock(),
        close=AsyncMock(),
        get_agent_by_name=AsyncMock(return_value=None),
    )
    scheduler = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    agent_runtime_cache = SimpleNamespace(close=AsyncMock())
    skill_manager = SimpleNamespace(initialize=AsyncMock())

    async def fake_safe_close_all(*closables: object) -> None:
        for obj in closables:
            close = getattr(obj, "close", None)
            if close is not None:
                await close()

    monkeypatch.setattr(app_module, "ConsoleConfig", lambda: config)
    monkeypatch.setattr(
        app_module, "create_run_step_storage", lambda _cfg: run_step_storage
    )
    monkeypatch.setattr(app_module, "create_trace_storage", lambda _cfg: trace_storage)
    monkeypatch.setattr(app_module, "AgentRegistry", lambda _cfg: agent_registry)
    monkeypatch.setattr(
        app_module, "RuntimeConfigService", lambda _cfg: SimpleNamespace()
    )
    monkeypatch.setattr(app_module, "Scheduler", lambda _cfg: scheduler)
    monkeypatch.setattr(app_module, "get_global_skill_manager", lambda: skill_manager)
    monkeypatch.setattr(
        app_module, "create_session_store", lambda **_kwargs: session_store
    )
    monkeypatch.setattr(
        app_module,
        "AgentRuntimeCache",
        lambda **_kwargs: agent_runtime_cache,
    )
    monkeypatch.setattr(
        app_module, "bind_console_runtime", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        app_module, "clear_console_runtime", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(app_module, "safe_close_all", fake_safe_close_all)

    async with app_module.lifespan(FastAPI()):
        pass

    session_store.connect.assert_awaited_once()
    session_store.close.assert_awaited_once()
    scheduler.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_closes_partial_startup_resources_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ConsoleConfig(
        run_step_storage_type="memory",
        trace_storage_type="memory",
        metadata_storage_type="memory",
        channels={
            "feishu": {
                "enabled": True,
            }
        },
    )
    run_step_storage = SimpleNamespace(close=AsyncMock())
    trace_storage = SimpleNamespace(close=AsyncMock())
    session_store = SimpleNamespace(connect=AsyncMock(), close=AsyncMock())
    agent_registry = SimpleNamespace(
        initialize=AsyncMock(),
        close=AsyncMock(),
        get_agent_by_name=AsyncMock(return_value=None),
    )
    scheduler = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    skill_manager = SimpleNamespace(initialize=AsyncMock())

    async def fake_safe_close_all(*closables: object) -> None:
        for obj in closables:
            close = getattr(obj, "close", None)
            if close is not None:
                await close()

    monkeypatch.setattr(app_module, "ConsoleConfig", lambda: config)
    monkeypatch.setattr(
        app_module, "create_run_step_storage", lambda _cfg: run_step_storage
    )
    monkeypatch.setattr(app_module, "create_trace_storage", lambda _cfg: trace_storage)
    monkeypatch.setattr(app_module, "AgentRegistry", lambda _cfg: agent_registry)
    monkeypatch.setattr(
        app_module, "RuntimeConfigService", lambda _cfg: SimpleNamespace()
    )
    monkeypatch.setattr(app_module, "Scheduler", lambda _cfg: scheduler)
    monkeypatch.setattr(app_module, "get_global_skill_manager", lambda: skill_manager)
    monkeypatch.setattr(
        app_module, "create_session_store", lambda **_kwargs: session_store
    )
    monkeypatch.setattr(app_module, "safe_close_all", fake_safe_close_all)

    with pytest.raises(RuntimeError, match="Feishu channel enabled but missing"):
        async with app_module.lifespan(FastAPI()):
            pass

    session_store.connect.assert_awaited_once()
    session_store.close.assert_awaited_once()
    agent_registry.close.assert_awaited_once()
    run_step_storage.close.assert_awaited_once()
    trace_storage.close.assert_awaited_once()
    scheduler.stop.assert_awaited_once()
