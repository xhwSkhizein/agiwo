"""Integration tests for dashboard overview API."""

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.agent.models.run import Run, RunMetrics, RunStatus
from agiwo.observability.trace import SpanStatus, Trace

from server.app import create_app
from server.config import ConsoleConfig
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
)
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.storage_wiring import create_run_step_storage, create_trace_storage


@pytest.mark.asyncio
async def test_overview_reports_real_totals_instead_of_recent_samples() -> None:
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

    created_at = datetime(2026, 4, 1, tzinfo=timezone.utc)
    for idx in range(3):
        await registry.create_agent(
            AgentConfigRecord(
                id=f"agent-{idx}",
                name=f"agent-{idx}",
                model_provider="openai",
                model_name="gpt-test",
            )
        )

    runs = [
        Run(
            id="run-1",
            agent_id="agent-0",
            session_id="session-a",
            user_input="hello",
            status=RunStatus.COMPLETED,
            response_content="done",
            metrics=RunMetrics(
                duration_ms=10.0,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                token_cost=0.1,
            ),
            created_at=created_at,
            updated_at=created_at,
        ),
        Run(
            id="run-2",
            agent_id="agent-0",
            session_id="session-a",
            user_input="follow-up",
            status=RunStatus.COMPLETED,
            response_content="done again",
            metrics=RunMetrics(
                duration_ms=12.0,
                input_tokens=8,
                output_tokens=4,
                total_tokens=12,
                token_cost=0.08,
            ),
            created_at=created_at,
            updated_at=created_at,
        ),
        Run(
            id="run-3",
            agent_id="agent-1",
            session_id="session-b",
            user_input="other",
            status=RunStatus.COMPLETED,
            response_content="done third",
            metrics=RunMetrics(
                duration_ms=5.0,
                input_tokens=4,
                output_tokens=2,
                total_tokens=6,
                token_cost=0.03,
            ),
            created_at=created_at,
            updated_at=created_at,
        ),
    ]
    for run in runs:
        await run_step_storage.save_run(run)

    traces = [
        Trace(
            trace_id="trace-1",
            agent_id="agent-0",
            session_id="session-a",
            status=SpanStatus.OK,
            total_tokens=15,
            total_llm_calls=1,
            total_tool_calls=0,
            total_token_cost=0.1,
            start_time=created_at,
        ),
        Trace(
            trace_id="trace-2",
            agent_id="agent-0",
            session_id="session-a",
            status=SpanStatus.OK,
            total_tokens=12,
            total_llm_calls=2,
            total_tool_calls=1,
            total_token_cost=0.08,
            start_time=created_at,
        ),
        Trace(
            trace_id="trace-3",
            agent_id="agent-1",
            session_id="session-b",
            status=SpanStatus.ERROR,
            total_tokens=6,
            total_llm_calls=1,
            total_tool_calls=1,
            total_token_cost=0.03,
            start_time=created_at,
        ),
    ]
    for trace in traces:
        await trace_storage.save_trace(trace)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/overview")

    assert response.status_code == 200
    assert response.json() == {
        "total_sessions": 2,
        "total_traces": 3,
        "total_agents": 4,
        "total_tokens": 33,
        "scheduler": {
            "total": 0,
            "pending": 0,
            "running": 0,
            "waiting": 0,
            "idle": 0,
            "queued": 0,
            "completed": 0,
            "failed": 0,
        },
    }

    clear_console_runtime(app)
    await registry.close()
    await run_step_storage.close()
