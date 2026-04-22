"""Integration tests for dashboard overview API."""

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.agent.models.log import RunFinished, RunStarted
from agiwo.agent.models.run import RunMetrics
from agiwo.agent import TerminationReason
from agiwo.observability.trace import SpanStatus, Trace

from server.app import create_app
from server.services.session_store import InMemorySessionStore
from server.config import ConsoleConfig
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
)
from server.models.session import Session
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
from server.services.storage_wiring import create_run_log_storage, create_trace_storage


@pytest.mark.asyncio
async def test_overview_reports_real_totals_instead_of_recent_samples() -> None:
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

    session_store = InMemorySessionStore()
    await session_store.connect()

    created_at = datetime(2026, 4, 1, tzinfo=timezone.utc)
    await session_store.upsert_session(
        Session(
            id="session-a",
            chat_context_scope_id=None,
            base_agent_id="agent-0",
            created_by="test",
            created_at=created_at,
            updated_at=created_at,
        )
    )
    await session_store.upsert_session(
        Session(
            id="session-b",
            chat_context_scope_id=None,
            base_agent_id="agent-1",
            created_by="test",
            created_at=created_at,
            updated_at=created_at,
        )
    )

    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_log_storage=run_log_storage,
            trace_storage=trace_storage,
            agent_registry=registry,
            session_store=session_store,
        ),
    )

    for idx in range(3):
        await registry.create_agent(
            AgentConfigRecord(
                id=f"agent-{idx}",
                name=f"agent-{idx}",
                model_provider="openai",
                model_name="gpt-test",
            )
        )

    await run_log_storage.append_entries(
        [
            RunStarted(
                sequence=1,
                session_id="session-a",
                run_id="run-1",
                agent_id="agent-0",
                user_input="hello",
                created_at=created_at,
            ),
            RunFinished(
                sequence=2,
                session_id="session-a",
                run_id="run-1",
                agent_id="agent-0",
                response="done",
                termination_reason=TerminationReason.COMPLETED,
                metrics=RunMetrics(
                    duration_ms=10.0,
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    token_cost=0.1,
                ).to_dict(),
                created_at=created_at,
            ),
            RunStarted(
                sequence=3,
                session_id="session-a",
                run_id="run-2",
                agent_id="agent-0",
                user_input="follow-up",
                created_at=created_at,
            ),
            RunFinished(
                sequence=4,
                session_id="session-a",
                run_id="run-2",
                agent_id="agent-0",
                response="done again",
                termination_reason=TerminationReason.COMPLETED,
                metrics=RunMetrics(
                    duration_ms=12.0,
                    input_tokens=8,
                    output_tokens=4,
                    total_tokens=12,
                    token_cost=0.08,
                ).to_dict(),
                created_at=created_at,
            ),
            RunStarted(
                sequence=1,
                session_id="session-b",
                run_id="run-3",
                agent_id="agent-1",
                user_input="other",
                created_at=created_at,
            ),
            RunFinished(
                sequence=2,
                session_id="session-b",
                run_id="run-3",
                agent_id="agent-1",
                response="done third",
                termination_reason=TerminationReason.COMPLETED,
                metrics=RunMetrics(
                    duration_ms=5.0,
                    input_tokens=4,
                    output_tokens=2,
                    total_tokens=6,
                    token_cost=0.03,
                ).to_dict(),
                created_at=created_at,
            ),
        ]
    )

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
    await run_log_storage.close()
