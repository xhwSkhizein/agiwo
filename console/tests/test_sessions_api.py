"""Integration tests for sessions and runs API envelopes and detail views."""

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.agent import TerminationReason
from agiwo.agent.models.log import (
    AssistantStepCommitted,
    RunFinished,
    RunStarted,
    ToolStepCommitted,
    UserStepCommitted,
)
from agiwo.agent.models.run import Run, RunMetrics, RunStatus
from agiwo.agent.models.step import MessageRole, StepMetrics, StepRecord
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    AgentStateStorageConfig,
    SchedulerRunResult,
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
from server.models.session import ChannelChatContext, Session
from server.services.agent_registry import AgentConfigRecord, AgentRegistry
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
    await session_store.close()
    await scheduler.stop()
    await registry.close()
    await run_step_storage.close()


async def _seed_session_context(client: AsyncClient) -> None:
    runtime = _runtime(client)
    assert runtime.session_store is not None
    created_at = datetime(2026, 4, 2, 8, 0, tzinfo=timezone.utc)
    await runtime.agent_registry.create_agent(
        AgentConfigRecord(
            id="agent-alpha",
            name="agent-alpha",
            model_provider="openai",
            model_name="gpt-test",
        )
    )
    await runtime.session_store.upsert_chat_context(
        ChannelChatContext(
            scope_id="agent-alpha",
            channel_instance_id="console-web",
            chat_id="agent-alpha",
            chat_type="dm",
            user_open_id="console-user",
            base_agent_id="agent-alpha",
            current_session_id="session-a",
            created_at=created_at,
            updated_at=created_at,
        )
    )
    await runtime.session_store.upsert_session(
        Session(
            id="session-a",
            chat_context_scope_id="agent-alpha",
            base_agent_id="agent-alpha",
            created_by="TEST",
            created_at=created_at,
            updated_at=created_at,
        )
    )
    await runtime.session_store.upsert_session(
        Session(
            id="session-b",
            chat_context_scope_id="agent-alpha",
            base_agent_id="agent-alpha",
            created_by="TEST",
            created_at=created_at,
            updated_at=created_at,
            source_session_id="session-a",
            fork_context_summary="forked for retry",
        )
    )


async def _seed_runs_and_steps(client: AsyncClient) -> None:
    runtime = _runtime(client)
    now = datetime(2026, 4, 2, 8, 0, tzinfo=timezone.utc)
    assert runtime.session_store is not None
    existing = await runtime.session_store.get_session("session-z")
    if existing is None:
        await runtime.session_store.upsert_session(
            Session(
                id="session-z",
                chat_context_scope_id=None,
                base_agent_id="agent-zeta",
                created_by="TEST",
                created_at=now,
                updated_at=now,
            )
        )
    await runtime.run_step_storage.save_run(
        Run(
            id="run-a1",
            agent_id="agent-alpha",
            session_id="session-a",
            user_input="hello",
            status=RunStatus.COMPLETED,
            response_content="done",
            metrics=RunMetrics(
                duration_ms=12.0,
                input_tokens=4,
                output_tokens=6,
                total_tokens=10,
                token_cost=0.2,
                steps_count=3,
                tool_calls_count=1,
            ),
            created_at=now,
            updated_at=now,
        )
    )
    await runtime.run_step_storage.append_entries(
        [
            RunStarted(
                sequence=1,
                session_id="session-a",
                run_id="run-a1",
                agent_id="agent-alpha",
                user_input="hello",
                created_at=now,
            ),
            RunFinished(
                sequence=2,
                session_id="session-a",
                run_id="run-a1",
                agent_id="agent-alpha",
                response="done",
                termination_reason=TerminationReason.COMPLETED,
                metrics=RunMetrics(
                    duration_ms=12.0,
                    input_tokens=4,
                    output_tokens=6,
                    total_tokens=10,
                    token_cost=0.2,
                    steps_count=3,
                    tool_calls_count=1,
                ).to_dict(),
                created_at=now,
            ),
        ]
    )
    await runtime.run_step_storage.save_run(
        Run(
            id="run-a2",
            agent_id="agent-alpha",
            session_id="session-a",
            user_input="again",
            status=RunStatus.COMPLETED,
            response_content="done again",
            metrics=RunMetrics(
                duration_ms=15.0,
                input_tokens=5,
                output_tokens=7,
                total_tokens=12,
                token_cost=0.3,
                steps_count=4,
                tool_calls_count=2,
            ),
            created_at=now,
            updated_at=now,
        )
    )
    await runtime.run_step_storage.append_entries(
        [
            RunStarted(
                sequence=3,
                session_id="session-a",
                run_id="run-a2",
                agent_id="agent-alpha",
                user_input="again",
                created_at=now,
            ),
            RunFinished(
                sequence=4,
                session_id="session-a",
                run_id="run-a2",
                agent_id="agent-alpha",
                response="done again",
                termination_reason=TerminationReason.COMPLETED,
                metrics=RunMetrics(
                    duration_ms=15.0,
                    input_tokens=5,
                    output_tokens=7,
                    total_tokens=12,
                    token_cost=0.3,
                    steps_count=4,
                    tool_calls_count=2,
                ).to_dict(),
                created_at=now,
            ),
        ]
    )
    await runtime.run_step_storage.save_run(
        Run(
            id="run-z1",
            agent_id="agent-zeta",
            session_id="session-z",
            user_input="zeta",
            status=RunStatus.COMPLETED,
            response_content="zeta done",
            metrics=RunMetrics(
                duration_ms=20.0,
                input_tokens=8,
                output_tokens=9,
                total_tokens=17,
                token_cost=0.4,
                steps_count=2,
                tool_calls_count=0,
            ),
            created_at=now,
            updated_at=now,
        )
    )
    await runtime.run_step_storage.append_entries(
        [
            RunStarted(
                sequence=1,
                session_id="session-z",
                run_id="run-z1",
                agent_id="agent-zeta",
                user_input="zeta",
                created_at=now,
            ),
            RunFinished(
                sequence=2,
                session_id="session-z",
                run_id="run-z1",
                agent_id="agent-zeta",
                response="zeta done",
                termination_reason=TerminationReason.COMPLETED,
                metrics=RunMetrics(
                    duration_ms=20.0,
                    input_tokens=8,
                    output_tokens=9,
                    total_tokens=17,
                    token_cost=0.4,
                    steps_count=2,
                    tool_calls_count=0,
                ).to_dict(),
                created_at=now,
            ),
        ]
    )

    for sequence, role, content, agent_id in [
        (1, MessageRole.USER, "hello", "agent-alpha"),
        (2, MessageRole.ASSISTANT, "thinking", "agent-alpha"),
        (3, MessageRole.TOOL, "tool output", "child-agent"),
    ]:
        await runtime.run_step_storage.save_step(
            StepRecord(
                id=f"step-{sequence}",
                session_id="session-a",
                run_id="run-a1",
                sequence=sequence,
                role=role,
                content=content,
                agent_id=agent_id,
                name="web_search" if role == MessageRole.TOOL else None,
                metrics=StepMetrics(duration_ms=float(sequence)),
                created_at=now,
            )
        )
    await runtime.run_step_storage.append_entries(
        [
            UserStepCommitted(
                sequence=1,
                session_id="session-a",
                run_id="run-a1",
                agent_id="agent-alpha",
                step_id="step-u1",
                role=MessageRole.USER,
                content="hello",
                user_input="hello",
                created_at=now,
            ),
            AssistantStepCommitted(
                sequence=2,
                session_id="session-a",
                run_id="run-a1",
                agent_id="agent-alpha",
                step_id="step-a2",
                role=MessageRole.ASSISTANT,
                content="thinking",
                metrics=StepMetrics(duration_ms=2.0),
                created_at=now,
            ),
            ToolStepCommitted(
                sequence=3,
                session_id="session-a",
                run_id="run-a1",
                agent_id="child-agent",
                step_id="step-t3",
                role=MessageRole.TOOL,
                content="tool output",
                tool_call_id="tc-3",
                name="web_search",
                metrics=StepMetrics(duration_ms=3.0),
                created_at=now,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_list_sessions_returns_paginated_envelope_without_runtime_binding_fields(
    client,
) -> None:
    await _seed_session_context(client)
    await _seed_runs_and_steps(client)

    response = await client.get("/api/sessions?limit=5&offset=0")

    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 5
    assert payload["offset"] == 0
    assert payload["has_more"] is False
    assert payload["total"] == 3
    items = payload["items"]
    assert {item["session_id"] for item in items} == {
        "session-a",
        "session-b",
        "session-z",
    }
    session_a = next(item for item in items if item["session_id"] == "session-a")
    assert session_a["base_agent_id"] == "agent-alpha"
    assert session_a["chat_context_scope_id"] == "agent-alpha"
    assert "runtime_agent_id" not in session_a
    assert "scheduler_state_id" not in session_a
    assert "current_task_id" not in session_a
    assert "task_message_count" not in session_a
    session_b = next(item for item in items if item["session_id"] == "session-b")
    assert session_b["run_count"] == 0
    assert session_b["source_session_id"] == "session-a"
    assert session_b["fork_context_summary"] == "forked for retry"


@pytest.mark.asyncio
async def test_get_session_detail_returns_summary_session_and_root_scheduler_state(
    client,
) -> None:
    await _seed_session_context(client)
    await _seed_runs_and_steps(client)
    scheduler = _runtime(client).scheduler
    assert scheduler is not None
    await scheduler._store.save_state(
        AgentState(
            id="session-a",
            session_id="session-a",
            status=AgentStateStatus.WAITING,
            task="wait for child",
            result_summary="waiting",
            last_run_result=SchedulerRunResult(
                run_id="run-last",
                termination_reason=TerminationReason.TIMEOUT,
                error="took too long",
            ),
        )
    )

    response = await client.get("/api/sessions/session-a")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["session_id"] == "session-a"
    assert payload["session"]["id"] == "session-a"
    assert "runtime_agent_id" not in payload["session"]
    assert "scheduler_state_id" not in payload["session"]
    assert payload["scheduler_state"]["id"] == "session-a"
    assert payload["scheduler_state"]["status"] == "waiting"
    assert payload["scheduler_state"]["last_run_result"]["run_id"] == "run-last"
    assert (
        payload["scheduler_state"]["last_run_result"]["termination_reason"] == "timeout"
    )


@pytest.mark.asyncio
async def test_list_runs_returns_page_envelope(client) -> None:
    await _seed_runs_and_steps(client)

    response = await client.get("/api/runs?limit=2&offset=0")

    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 2
    assert payload["offset"] == 0
    assert payload["has_more"] is True
    assert payload["total"] is None
    assert len(payload["items"]) == 2


@pytest.mark.asyncio
async def test_get_session_steps_supports_order_and_has_more(client) -> None:
    await _seed_runs_and_steps(client)

    response = await client.get(
        "/api/sessions/session-a/steps?limit=2&order=desc&agent_id=agent-alpha"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["has_more"] is False
    assert payload["total"] is None
    assert [item["sequence"] for item in payload["items"]] == [2, 1]
