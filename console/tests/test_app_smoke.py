"""Smoke test: create_app() succeeds and basic health endpoint responds."""

import pytest
from httpx import ASGITransport, AsyncClient

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
