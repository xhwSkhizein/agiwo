from datetime import datetime, timezone

import pytest

from agiwo.agent.schema import (
    EventType,
    LLMCallContext,
    MessageRole,
    StepMetrics,
    StepRecord,
    StreamEvent,
)
from agiwo.observability.collector import TraceCollector
from agiwo.observability.trace import SpanKind


@pytest.mark.asyncio
async def test_collect_routes_assistant_and_tool_steps() -> None:
    collector = TraceCollector()
    collector.start(
        trace_id="trace-1",
        agent_id="agent-1",
        session_id="session-1",
        input_query="search for updates",
    )

    await collector.collect(
        StreamEvent(
            type=EventType.RUN_STARTED,
            run_id="run-1",
            agent_id="agent-1",
            data={"session_id": "session-1"},
        )
    )

    assistant_step = StepRecord(
        session_id="session-1",
        run_id="run-1",
        sequence=1,
        role=MessageRole.ASSISTANT,
        content="Let me search that.",
        tool_calls=[
            {
                "id": "call-1",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "latest updates"}',
                },
            }
        ],
        metrics=StepMetrics(
            start_at=datetime(2026, 3, 10, 10, 0, 0, tzinfo=timezone.utc),
            end_at=datetime(2026, 3, 10, 10, 0, 1, tzinfo=timezone.utc),
            duration_ms=1000.0,
            input_tokens=120,
            output_tokens=30,
            total_tokens=150,
            usage_source="provider",
            model_name="test-model",
            provider="test-provider",
        ),
    )
    await collector.collect(
        StreamEvent(
            type=EventType.STEP_COMPLETED,
            run_id="run-1",
            step=assistant_step,
            llm=LLMCallContext(
                messages=[{"role": "user", "content": "search"}],
                tools=[{"name": "web_search"}],
                request_params={"temperature": 0.1},
                finish_reason="tool_calls",
            ),
        )
    )

    tool_step = StepRecord(
        session_id="session-1",
        run_id="run-1",
        sequence=2,
        role=MessageRole.TOOL,
        tool_call_id="call-1",
        name="web_search",
        content="Found matching documents",
        content_for_user="Found matching documents",
        metrics=StepMetrics(duration_ms=75.0),
    )
    await collector.collect(
        StreamEvent(
            type=EventType.STEP_COMPLETED,
            run_id="run-1",
            step=tool_step,
        )
    )

    trace = collector._trace
    assert trace is not None

    llm_span = next(span for span in trace.spans if span.kind == SpanKind.LLM_CALL)
    assert llm_span.llm_details == {
        "request": {"temperature": 0.1},
        "messages": [{"role": "user", "content": "search"}],
        "tools": [{"name": "web_search"}],
        "response_content": "Let me search that.",
        "response_tool_calls": assistant_step.tool_calls,
        "finish_reason": "tool_calls",
        "status": "completed",
        "metrics": {
            "duration_ms": 1000.0,
            "first_token_ms": None,
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
            "cache_read_tokens": None,
            "cache_creation_tokens": None,
            "usage_source": "provider",
        },
    }
    assert llm_span.metrics["usage_source"] == "provider"

    tool_span = next(span for span in trace.spans if span.kind == SpanKind.TOOL_CALL)
    assert tool_span.tool_details is not None
    assert tool_span.tool_details["tool_name"] == "web_search"
    assert tool_span.tool_details["input_args"] == {"query": "latest updates"}
    assert tool_span.tool_details["status"] == "completed"
    assert tool_span.tool_details["content_for_user"] == "Found matching documents"
