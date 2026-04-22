from datetime import datetime, timezone

import pytest

from agiwo.agent import StepMetrics, TerminationReason
from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CompactionApplied,
    LLMCallCompleted,
    LLMCallStarted,
    RetrospectApplied,
    RunFinished,
    RunStarted,
    TerminationDecided,
    ToolStepCommitted,
)
from agiwo.agent.models.step import MessageRole
from agiwo.agent.trace_writer import AgentTraceCollector
from agiwo.observability.base import BaseTraceStorage, TraceQuery
from agiwo.observability.trace import SpanKind, Trace


class _RecordingTraceStorage(BaseTraceStorage):
    def __init__(self) -> None:
        self.saved: list[Trace] = []

    async def save_trace(self, trace: Trace) -> None:
        self.saved.append(trace.model_copy(deep=True))

    async def get_trace(self, trace_id: str) -> Trace | None:
        for trace in reversed(self.saved):
            if trace.trace_id == trace_id:
                return trace
        return None

    async def query_traces(self, query: TraceQuery | dict) -> list[Trace]:
        del query
        return list(self.saved)

    async def close(self) -> None:
        self.saved.clear()


@pytest.mark.asyncio
async def test_collects_trace_from_committed_run_log_entries_only() -> None:
    storage = _RecordingTraceStorage()
    collector = AgentTraceCollector(store=storage)
    collector.start(
        trace_id="trace-1",
        agent_id="agent-1",
        session_id="session-1",
        input_query="search for updates",
    )

    tool_call = {
        "id": "call-1",
        "function": {
            "name": "web_search",
            "arguments": '{"query": "latest updates"}',
        },
    }
    metrics = StepMetrics(
        start_at=datetime(2026, 3, 10, 10, 0, 0, tzinfo=timezone.utc),
        end_at=datetime(2026, 3, 10, 10, 0, 1, tzinfo=timezone.utc),
        duration_ms=1000.0,
        input_tokens=120,
        output_tokens=30,
        total_tokens=150,
        usage_source="provider",
        model_name="test-model",
        provider="test-provider",
    )

    await collector.on_run_log_entries(
        [
            RunStarted(
                sequence=1,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                user_input="search for updates",
            ),
            LLMCallStarted(
                sequence=2,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                messages=[{"role": "user", "content": "search"}],
                tools=[{"name": "web_search"}],
            ),
            AssistantStepCommitted(
                sequence=3,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                step_id="step-1",
                role=MessageRole.ASSISTANT,
                content="Let me search that.",
                tool_calls=[tool_call],
                metrics=metrics,
            ),
            LLMCallCompleted(
                sequence=4,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                content="Let me search that.",
                tool_calls=[tool_call],
                finish_reason="tool_calls",
                metrics=metrics,
            ),
            ToolStepCommitted(
                sequence=5,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                step_id="step-2",
                role=MessageRole.TOOL,
                tool_call_id="call-1",
                name="web_search",
                content="Found matching documents",
                content_for_user="Found matching documents",
                metrics=StepMetrics(duration_ms=75.0),
            ),
            RunFinished(
                sequence=6,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                response="Found matching documents",
                termination_reason=TerminationReason.COMPLETED,
            ),
        ]
    )

    trace = collector._trace
    assert trace is not None
    assert len(storage.saved) == 1

    llm_span = next(span for span in trace.spans if span.kind == SpanKind.LLM_CALL)
    assert llm_span.llm_details == {
        "request": {},
        "messages": [{"role": "user", "content": "search"}],
        "tools": [{"name": "web_search"}],
        "response_content": "Let me search that.",
        "response_tool_calls": [tool_call],
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


@pytest.mark.asyncio
async def test_collector_records_runtime_run_log_entries() -> None:
    storage = _RecordingTraceStorage()
    collector = AgentTraceCollector(store=storage)
    collector.start(
        trace_id="trace-2",
        agent_id="agent-1",
        session_id="session-1",
        input_query="hello",
    )

    await collector.on_run_log_entries(
        [
            RunStarted(
                sequence=1,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                user_input="hello",
            ),
            CompactionApplied(
                sequence=3,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                start_sequence=1,
                end_sequence=2,
                before_token_estimate=1200,
                after_token_estimate=320,
                message_count=4,
                transcript_path="/tmp/transcript.jsonl",
                analysis={"summary": "short summary"},
                summary="short summary",
                compact_model="compact-model",
                compact_tokens=128,
            ),
            RetrospectApplied(
                sequence=4,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                affected_sequences=[2],
                affected_step_ids=["step-2"],
                feedback="switch plan",
                replacement="[ToolResult offloaded to /tmp/x.txt]",
            ),
            TerminationDecided(
                sequence=5,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                termination_reason=TerminationReason.MAX_STEPS,
                phase="pre_llm",
                source="non_recoverable_limit",
            ),
        ]
    )

    trace = collector._trace
    assert trace is not None
    assert len(storage.saved) == 1
    runtime_spans = [span for span in trace.spans if span.kind == SpanKind.RUNTIME]
    assert [span.name for span in runtime_spans] == [
        "compaction",
        "retrospect",
        "termination",
    ]
    assert runtime_spans[0].attributes["summary"] == "short summary"
    assert runtime_spans[1].attributes["feedback"] == "switch plan"
    assert runtime_spans[2].attributes["termination_reason"] == "max_steps"


@pytest.mark.asyncio
async def test_build_trace_from_run_log_entries() -> None:
    collector = AgentTraceCollector()

    trace = collector.build_from_entries(
        [
            RunStarted(
                sequence=1,
                session_id="session-2",
                run_id="run-2",
                agent_id="agent-2",
                user_input="hello",
            ),
            TerminationDecided(
                sequence=2,
                session_id="session-2",
                run_id="run-2",
                agent_id="agent-2",
                termination_reason=TerminationReason.COMPLETED,
                phase="post_llm",
                source="assistant_completed_without_tools",
            ),
        ]
    )

    assert trace.agent_id == "agent-2"
    runtime_spans = [span for span in trace.spans if span.kind == SpanKind.RUNTIME]
    assert len(runtime_spans) == 1
    assert runtime_spans[0].name == "termination"
