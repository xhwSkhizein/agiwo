"""Integration tests for traces API envelopes."""

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace

from server.app import create_app
from server.config import ConsoleConfig
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
    get_console_runtime_from_app,
)
from server.services.agent_registry import AgentRegistry
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


@pytest.mark.asyncio
async def test_get_trace_returns_runtime_decisions_and_timeline_events(client) -> None:
    runtime = _runtime(client)
    created_at = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
    root_span = Span(
        trace_id="trace-detail",
        kind=SpanKind.AGENT,
        name="agent-alpha",
        depth=0,
        run_id="run-1",
        start_time=created_at,
        end_time=created_at.replace(second=2),
        duration_ms=2000.0,
        status=SpanStatus.OK,
        attributes={
            "agent_id": "agent-alpha",
            "session_id": "sess-1",
            "nested": False,
            "parent_run_id": None,
            "start_sequence": 1,
            "end_sequence": 9,
        },
        output_preview="done",
    )
    await runtime.trace_storage.save_trace(
        Trace(
            trace_id="trace-detail",
            agent_id="agent-alpha",
            session_id="sess-1",
            status=SpanStatus.OK,
            start_time=created_at,
            end_time=created_at.replace(second=2),
            duration_ms=2000.0,
            root_span_id=root_span.span_id,
            spans=[
                root_span,
                Span(
                    trace_id="trace-detail",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.LLM_CALL,
                    name="gpt-5.4",
                    depth=1,
                    run_id="run-1",
                    start_time=created_at,
                    end_time=created_at.replace(second=1),
                    duration_ms=900.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 4,
                        "agent_id": "agent-alpha",
                        "provider": "openai-response",
                    },
                    llm_details={
                        "messages": [{"role": "user", "content": "fix auth flow"}],
                        "tools": [
                            {"name": "web_search"},
                            {"name": "review_trajectory"},
                        ],
                        "response_content": "Looking at JWT references",
                        "response_tool_calls": [{"id": "tc-1"}],
                        "finish_reason": "tool_calls",
                        "metrics": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                            "first_token_ms": 321.0,
                        },
                    },
                    output_preview="Looking at JWT references",
                ),
                Span(
                    trace_id="trace-detail",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.TOOL_CALL,
                    name="web_search",
                    depth=1,
                    run_id="run-1",
                    step_id="step-1",
                    start_time=created_at.replace(second=1),
                    end_time=created_at.replace(second=1),
                    duration_ms=42.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 5,
                        "agent_id": "agent-alpha",
                        "tool_name": "web_search",
                        "tool_call_id": "tc-1",
                    },
                    tool_details={
                        "tool_name": "web_search",
                        "tool_call_id": "tc-1",
                        "input_args": {"q": "jwt"},
                        "output": (
                            "Found 15 JWT references\n\n"
                            "<system-review>\n"
                            'Active milestone: "Fix auth"\n\n'
                            "Trigger: step_interval\n"
                            "Steps since last review: 8\n"
                            "Hook advice: narrow the search\n"
                            "</system-review>"
                        ),
                        "status": "completed",
                    },
                ),
                Span(
                    trace_id="trace-detail",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.TOOL_CALL,
                    name="review_trajectory",
                    depth=1,
                    run_id="run-1",
                    step_id="step-2",
                    start_time=created_at.replace(second=1),
                    end_time=created_at.replace(second=1),
                    duration_ms=18.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 6,
                        "agent_id": "agent-alpha",
                        "tool_name": "review_trajectory",
                        "tool_call_id": "tc-review",
                    },
                    tool_details={
                        "tool_name": "review_trajectory",
                        "tool_call_id": "tc-review",
                        "input_args": {"aligned": False},
                        "output": "Trajectory review: aligned=False. JWT was a dead end",
                        "status": "completed",
                    },
                ),
                Span(
                    trace_id="trace-detail",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.RUNTIME,
                    name="review_trigger",
                    depth=1,
                    run_id="run-1",
                    start_time=created_at.replace(second=1),
                    end_time=created_at.replace(second=1),
                    duration_ms=0.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 6,
                        "agent_id": "agent-alpha",
                        "trigger_reason": "step_interval",
                        "active_milestone_id": "fix",
                        "review_count_since_checkpoint": 8,
                        "trigger_tool_call_id": "tc-1",
                        "trigger_tool_step_id": "step-1",
                        "notice_step_id": "step-1",
                    },
                ),
                Span(
                    trace_id="trace-detail",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.RUNTIME,
                    name="review_outcome",
                    depth=1,
                    run_id="run-1",
                    start_time=created_at.replace(second=1),
                    end_time=created_at.replace(second=1),
                    duration_ms=0.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 7,
                        "agent_id": "agent-alpha",
                        "aligned": False,
                        "mode": "step_back",
                        "experience": "switch plan",
                        "active_milestone_id": "fix",
                        "review_tool_call_id": "tc-review",
                        "review_step_id": "step-2",
                        "condensed_step_ids": ["step-a", "step-b"],
                    },
                ),
                Span(
                    trace_id="trace-detail",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.RUNTIME,
                    name="step_back",
                    depth=1,
                    run_id="run-1",
                    start_time=created_at.replace(second=1),
                    end_time=created_at.replace(second=1),
                    duration_ms=0.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 8,
                        "agent_id": "agent-alpha",
                        "affected_count": 2,
                        "checkpoint_seq": 4,
                        "experience": "switch plan",
                    },
                ),
            ],
            total_tokens=10,
            total_input_tokens=4,
            total_output_tokens=6,
            total_tool_calls=2,
            total_llm_calls=0,
            input_query="fix auth flow",
            final_output="done",
        )
    )

    response = await client.get("/api/traces/trace-detail")

    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime_decisions"][0]["kind"] == "step_back"
    assert payload["runtime_decisions"][0]["details"]["experience"] == "switch plan"
    assert [event["kind"] for event in payload["timeline_events"]] == [
        "run_started",
        "llm_call",
        "tool_call",
        "tool_call",
        "review_checkpoint",
        "review_result",
        "runtime_decision",
        "run_finished",
    ]
    assert payload["timeline_events"][4]["details"]["trigger_reason"] == "step_interval"
    assert [event["kind"] for event in payload["mainline_events"]] == [
        "run_started",
        "review_checkpoint",
        "review_result",
        "runtime_decision",
        "run_finished",
    ]
    assert payload["review_cycles"][0]["trigger_reason"] == "step_interval"
    assert payload["review_cycles"][0]["step_back_applied"] is True
    assert payload["llm_calls"][0]["model"] == "gpt-5.4"
    assert payload["llm_calls"][0]["response_tool_call_count"] == 1


@pytest.mark.asyncio
async def test_get_trace_handles_null_milestone_list(client) -> None:
    runtime = _runtime(client)
    created_at = datetime(2026, 4, 25, 12, 5, tzinfo=timezone.utc)
    root_span = Span(
        trace_id="trace-milestones-null",
        kind=SpanKind.AGENT,
        name="agent-alpha",
        depth=0,
        run_id="run-2",
        start_time=created_at,
        end_time=created_at.replace(second=1),
        duration_ms=1000.0,
        status=SpanStatus.OK,
        attributes={
            "agent_id": "agent-alpha",
            "session_id": "sess-2",
            "nested": False,
            "parent_run_id": None,
            "start_sequence": 1,
            "end_sequence": 2,
        },
    )
    await runtime.trace_storage.save_trace(
        Trace(
            trace_id="trace-milestones-null",
            agent_id="agent-alpha",
            session_id="sess-2",
            status=SpanStatus.OK,
            start_time=created_at,
            end_time=created_at.replace(second=1),
            duration_ms=1000.0,
            root_span_id=root_span.span_id,
            spans=[
                root_span,
                Span(
                    trace_id="trace-milestones-null",
                    parent_span_id=root_span.span_id,
                    kind=SpanKind.RUNTIME,
                    name="review_milestones",
                    depth=1,
                    run_id="run-2",
                    start_time=created_at,
                    end_time=created_at,
                    duration_ms=0.0,
                    status=SpanStatus.OK,
                    attributes={
                        "sequence": 2,
                        "agent_id": "agent-alpha",
                        "active_milestone_id": None,
                        "source_tool_call_id": "tc-milestones",
                        "source_step_id": "step-milestones",
                        "milestones": None,
                    },
                ),
            ],
        )
    )

    response = await client.get("/api/traces/trace-milestones-null")

    assert response.status_code == 200
    payload = response.json()
    milestone_event = next(
        event
        for event in payload["timeline_events"]
        if event["kind"] == "milestone_update"
    )
    assert milestone_event["summary"] == "0 milestones declared/updated"
    assert milestone_event["details"]["milestones"] == []
