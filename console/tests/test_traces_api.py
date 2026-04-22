"""Integration tests for traces API envelopes."""

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.observability.trace import SpanStatus, Trace

from server.app import create_app
from server.config import ConsoleConfig
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
)
from server.services.agent_registry import AgentRegistry
from server.services.storage_wiring import create_run_log_storage, create_trace_storage


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
    registry = AgentRegistry(config)
    await registry.initialize()

    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_log_storage=run_log_storage,
            trace_storage=trace_storage,
            agent_registry=registry,
        ),
    )

    created_at = datetime(2026, 4, 2, tzinfo=timezone.utc)
    for idx in range(3):
        await trace_storage.save_trace(
            Trace(
                trace_id=f"trace-{idx}",
                agent_id="agent-alpha",
                session_id=f"session-{idx}",
                status=SpanStatus.OK,
                total_tokens=10 + idx,
                total_token_cost=0.1 + idx,
                start_time=created_at,
            )
        )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    clear_console_runtime(app)
    await registry.close()
    await run_log_storage.close()


@pytest.mark.asyncio
async def test_list_traces_returns_page_envelope(client) -> None:
    response = await client.get("/api/traces?limit=2&offset=0")

    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 2
    assert payload["offset"] == 0
    assert payload["has_more"] is True
    assert payload["total"] is None
    assert len(payload["items"]) == 2
