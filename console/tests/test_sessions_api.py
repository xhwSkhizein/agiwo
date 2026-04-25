"""Integration tests for sessions and runs API envelopes and detail views."""

from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.agent import TerminationReason
from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CompactionFailed,
    RunFinished,
    RunStarted,
    StepBackApplied,
    TerminationDecided,
    ToolStepCommitted,
    UserStepCommitted,
)
from agiwo.agent.models.run import RunMetrics
from agiwo.agent.models.step import MessageRole, StepMetrics
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace
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
from server.services.storage_wiring import create_run_log_storage, create_trace_storage


def _runtime(client: AsyncClient) -> ConsoleRuntime:
    return get_console_runtime_from_app(client._transport.app)  # type: ignore[attr-defined]


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
            run_log_storage=run_log_storage,
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
    await run_log_storage.close()


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
    await runtime.run_log_storage.append_entries(
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
    await runtime.trace_storage.save_trace(
        Trace(
            trace_id="trace-a2",
            agent_id="agent-alpha",
            session_id="session-a",
            status=SpanStatus.OK,
            total_tokens=12,
            total_llm_calls=1,
            start_time=now + timedelta(seconds=1),
            duration_ms=15.0,
            input_query="again",
            final_output="done again",
        )
    )
    await runtime.run_log_storage.append_entries(
        [
            RunStarted(
                sequence=3,
                session_id="session-a",
                run_id="run-a2",
                agent_id="agent-alpha",
                user_input="again",
                created_at=now + timedelta(seconds=1),
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
                created_at=now + timedelta(seconds=1),
            ),
            TerminationDecided(
                sequence=5,
                session_id="session-a",
                run_id="run-a2",
                agent_id="agent-alpha",
                termination_reason=TerminationReason.COMPLETED,
                phase="after_tool_batch",
                source="finished",
                created_at=now + timedelta(seconds=1),
            ),
        ]
    )
    await runtime.run_log_storage.append_entries(
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
    await runtime.run_log_storage.append_entries(
        [
            UserStepCommitted(
                sequence=6,
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
                sequence=7,
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
                sequence=8,
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
    assert payload["observability"]["recent_traces"][0]["trace_id"] == "trace-a2"
    assert payload["observability"]["decision_events"][0]["kind"] == "termination"
    assert (
        payload["observability"]["decision_events"][0]["details"]["reason"]
        == "completed"
    )


@pytest.mark.asyncio
async def test_get_session_detail_includes_milestone_board_and_conversation_events(
    client,
) -> None:
    await _seed_session_context(client)
    await _seed_runs_and_steps(client)
    runtime = _runtime(client)
    created_at = datetime(2026, 4, 2, 8, 0, 5, tzinfo=timezone.utc)
    root_span = Span(
        trace_id="trace-mainline",
        kind=SpanKind.AGENT,
        name="agent-alpha",
        depth=0,
        run_id="run-a2",
        start_time=created_at,
        end_time=created_at.replace(second=9),
        duration_ms=4000.0,
        status=SpanStatus.OK,
        attributes={
            "agent_id": "agent-alpha",
            "session_id": "session-a",
            "nested": False,
            "parent_run_id": None,
            "start_sequence": 3,
            "end_sequence": 10,
        },
        output_preview="done again",
    )
    await runtime.trace_storage.save_trace(
        Trace(
            trace_id="trace-mainline",
            agent_id="agent-alpha",
            session_id="session-a",
            status=SpanStatus.OK,
            start_time=created_at,
            end_time=created_at.replace(second=9),
            duration_ms=4000.0,
            root_span_id=root_span.span_id,
            spans=[
                root_span,
                Span(
                    trace_id="trace-mainline",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.TOOL_CALL,
                    name="declare_milestones",
                    depth=1,
                    run_id="run-a2",
                    step_id="step-milestones",
                    start_time=created_at.replace(second=6),
                    end_time=created_at.replace(second=6),
                    duration_ms=5.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 6,
                        "agent_id": "agent-alpha",
                        "tool_name": "declare_milestones",
                    },
                    tool_details={
                        "tool_name": "declare_milestones",
                        "input_args": {
                            "milestones": [
                                {
                                    "id": "inspect",
                                    "description": "Inspect auth flow",
                                    "status": "active",
                                },
                                {
                                    "id": "fix",
                                    "description": "Apply auth fix",
                                    "status": "pending",
                                },
                            ]
                        },
                        "output": {
                            "milestones": [
                                {
                                    "id": "inspect",
                                    "description": "Inspect auth flow",
                                    "status": "active",
                                },
                                {
                                    "id": "fix",
                                    "description": "Apply auth fix",
                                    "status": "pending",
                                },
                            ]
                        },
                        "status": "completed",
                    },
                ),
                Span(
                    trace_id="trace-mainline",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.TOOL_CALL,
                    name="web_search",
                    depth=1,
                    run_id="run-a2",
                    step_id="step-1",
                    start_time=created_at.replace(second=7),
                    end_time=created_at.replace(second=7),
                    duration_ms=20.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 7,
                        "agent_id": "agent-alpha",
                        "tool_name": "web_search",
                    },
                    tool_details={
                        "tool_name": "web_search",
                        "output": (
                            "<system-review>\n"
                            'Active milestone: "Inspect auth flow"\n\n'
                            "Trigger: step_interval\n"
                            "Steps since last review: 4\n"
                            "</system-review>"
                        ),
                        "status": "completed",
                    },
                ),
                Span(
                    trace_id="trace-mainline",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.RUNTIME,
                    name="review_milestones",
                    depth=1,
                    run_id="run-a2",
                    start_time=created_at.replace(second=6),
                    end_time=created_at.replace(second=6),
                    duration_ms=0.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 6,
                        "agent_id": "agent-alpha",
                        "active_milestone_id": "inspect",
                        "source_tool_call_id": "tc-milestones",
                        "source_step_id": "step-milestones",
                        "milestones": [
                            {
                                "id": "inspect",
                                "description": "Inspect auth flow",
                                "status": "active",
                                "declared_at_seq": 6,
                                "completed_at_seq": None,
                            },
                            {
                                "id": "fix",
                                "description": "Apply auth fix",
                                "status": "pending",
                                "declared_at_seq": 6,
                                "completed_at_seq": None,
                            },
                        ],
                    },
                ),
                Span(
                    trace_id="trace-mainline",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.RUNTIME,
                    name="review_trigger",
                    depth=1,
                    run_id="run-a2",
                    start_time=created_at.replace(second=7),
                    end_time=created_at.replace(second=7),
                    duration_ms=0.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 7,
                        "agent_id": "agent-alpha",
                        "trigger_reason": "step_interval",
                        "active_milestone_id": "inspect",
                        "review_count_since_checkpoint": 4,
                        "trigger_tool_call_id": "tc-1",
                        "trigger_tool_step_id": "step-1",
                        "notice_step_id": "step-1",
                    },
                ),
                Span(
                    trace_id="trace-mainline",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.TOOL_CALL,
                    name="review_trajectory",
                    depth=1,
                    run_id="run-a2",
                    step_id="step-2",
                    start_time=created_at.replace(second=8),
                    end_time=created_at.replace(second=8),
                    duration_ms=10.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 8,
                        "agent_id": "agent-alpha",
                        "tool_name": "review_trajectory",
                    },
                    tool_details={
                        "tool_name": "review_trajectory",
                        "input_args": {
                            "aligned": False,
                            "experience": "Need to switch from search to code inspection.",
                        },
                        "output": (
                            "Trajectory review: aligned=false. "
                            "Need to switch from search to code inspection."
                        ),
                        "status": "completed",
                    },
                ),
                Span(
                    trace_id="trace-mainline",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.RUNTIME,
                    name="review_outcome",
                    depth=1,
                    run_id="run-a2",
                    start_time=created_at.replace(second=8),
                    end_time=created_at.replace(second=8),
                    duration_ms=0.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 8,
                        "agent_id": "agent-alpha",
                        "aligned": False,
                        "mode": "step_back",
                        "experience": "Switch to code inspection.",
                        "active_milestone_id": "inspect",
                        "review_tool_call_id": "tc-review",
                        "review_step_id": "step-2",
                        "condensed_step_ids": ["step-a", "step-b"],
                    },
                ),
                Span(
                    trace_id="trace-mainline",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.RUNTIME,
                    name="step_back",
                    depth=1,
                    run_id="run-a2",
                    start_time=created_at.replace(second=9),
                    end_time=created_at.replace(second=9),
                    duration_ms=0.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 9,
                        "agent_id": "agent-alpha",
                        "affected_count": 2,
                        "checkpoint_seq": 5,
                        "experience": "Switch to code inspection.",
                    },
                ),
            ],
        )
    )

    response = await client.get("/api/sessions/session-a")

    assert response.status_code == 200
    payload = response.json()
    assert payload["milestone_board"]["active_milestone_id"] == "inspect"
    assert (
        payload["milestone_board"]["latest_review_outcome"]["step_back_applied"] is True
    )
    assert payload["review_cycles"][0]["trigger_reason"] == "step_interval"
    assert payload["conversation_events"][0]["kind"] == "user_message"
    assert any(
        event["kind"] == "review_event" for event in payload["conversation_events"]
    )


@pytest.mark.asyncio
async def test_get_session_detail_lists_recent_runtime_decisions(client) -> None:
    await _seed_session_context(client)
    await _seed_runs_and_steps(client)
    runtime = _runtime(client)
    await runtime.run_log_storage.append_entries(
        [
            CompactionFailed(
                sequence=9,
                session_id="session-a",
                run_id="run-a2",
                agent_id="agent-alpha",
                error="model timeout",
                attempt=1,
                max_attempts=2,
                terminal=False,
                created_at=datetime(2026, 4, 2, 12, 0, 2, tzinfo=timezone.utc),
            ),
            StepBackApplied(
                sequence=10,
                session_id="session-a",
                run_id="run-a2",
                agent_id="agent-alpha",
                affected_count=2,
                checkpoint_seq=7,
                experience="switch plan",
                created_at=datetime(2026, 4, 2, 12, 0, 3, tzinfo=timezone.utc),
            ),
        ]
    )

    response = await client.get("/api/sessions/session-a")

    assert response.status_code == 200
    payload = response.json()
    assert [event["kind"] for event in payload["observability"]["decision_events"]] == [
        "step_back",
        "compaction_failed",
        "termination",
    ]
    assert payload["observability"]["decision_events"][0]["details"]["experience"] == (
        "switch plan"
    )
    assert (
        payload["observability"]["decision_events"][1]["details"]["error"]
        == "model timeout"
    )


@pytest.mark.asyncio
async def test_list_runs_returns_page_envelope(client) -> None:
    await _seed_runs_and_steps(client)

    response = await client.get("/api/runs?session_id=session-a&limit=2&offset=0")

    assert response.status_code == 200
    payload = response.json()
    assert payload["limit"] == 2
    assert payload["offset"] == 0
    assert payload["has_more"] is False
    assert payload["total"] is None
    assert len(payload["items"]) == 2
    assert [item["id"] for item in payload["items"]] == ["run-a2", "run-a1"]


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
    assert [item["sequence"] for item in payload["items"]] == [7, 6]


@pytest.mark.asyncio
async def test_get_session_steps_reports_total_for_unfiltered_session(client) -> None:
    await _seed_runs_and_steps(client)

    response = await client.get("/api/sessions/session-a/steps?limit=10&order=asc")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 3
    assert [item["sequence"] for item in payload["items"]] == [6, 7, 8]
