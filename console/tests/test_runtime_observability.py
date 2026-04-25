from datetime import datetime, timezone

from agiwo.agent import MessageRole, StepView
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace

from server.services.runtime.runtime_observability import (
    build_conversation_events,
    build_session_milestone_board,
    build_trace_llm_call_records,
    build_trace_mainline_events,
    build_trace_review_cycles,
)


def _trace_with_review_and_llm() -> Trace:
    trace = Trace(
        trace_id="trace-1",
        agent_id="agent-1",
        session_id="sess-1",
        start_time=datetime(2026, 4, 25, tzinfo=timezone.utc),
    )
    trace.spans = [
        Span(
            trace_id="trace-1",
            span_id="run-1-root",
            parent_span_id=None,
            kind=SpanKind.AGENT,
            name="agent-1",
            depth=0,
            run_id="run-1",
            start_time=datetime(2026, 4, 25, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, 0, 0, 4, tzinfo=timezone.utc),
            duration_ms=4000.0,
            status=SpanStatus.OK,
            attributes={
                "agent_id": "agent-1",
                "session_id": "sess-1",
                "nested": False,
                "parent_run_id": None,
                "start_sequence": 1,
                "end_sequence": 9,
            },
            output_preview="done",
        ),
        Span(
            trace_id="trace-1",
            span_id="llm-1",
            parent_span_id="run-1-root",
            kind=SpanKind.LLM_CALL,
            name="gpt-5.4",
            depth=1,
            run_id="run-1",
            start_time=datetime(2026, 4, 25, 0, 0, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, 0, 0, 2, tzinfo=timezone.utc),
            duration_ms=1234.0,
            status=SpanStatus.OK,
            attributes={
                "agent_id": "agent-1",
                "sequence": 5,
                "provider": "openai-response",
            },
            llm_details={
                "messages": [{"role": "user", "content": "Inspect auth"}],
                "tools": [{"name": "bash"}, {"name": "review_trajectory"}],
                "response_content": "Aligned.",
                "response_tool_calls": [{"id": "tc-1"}],
                "finish_reason": "stop",
                "metrics": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "first_token_ms": 345.0,
                },
            },
            output_preview="Aligned.",
        ),
        Span(
            trace_id="trace-1",
            span_id="runtime-milestones",
            parent_span_id="run-1-root",
            kind=SpanKind.RUNTIME,
            name="review_milestones",
            depth=1,
            run_id="run-1",
            start_time=datetime(2026, 4, 25, 0, 0, 0, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, 0, 0, 0, tzinfo=timezone.utc),
            duration_ms=0.0,
            status=SpanStatus.OK,
            attributes={
                "agent_id": "agent-1",
                "sequence": 2,
                "active_milestone_id": "inspect",
                "source_tool_call_id": "tc-milestones",
                "source_step_id": "step-milestones",
                "reason": "declared",
                "milestones": [
                    {
                        "id": "inspect",
                        "description": "Inspect auth",
                        "status": "active",
                        "declared_at_seq": 2,
                        "completed_at_seq": None,
                    },
                    {
                        "id": "fix",
                        "description": "Apply fix",
                        "status": "pending",
                        "declared_at_seq": 2,
                        "completed_at_seq": None,
                    },
                ],
            },
        ),
        Span(
            trace_id="trace-1",
            span_id="tool-1",
            parent_span_id="run-1-root",
            kind=SpanKind.TOOL_CALL,
            name="bash",
            depth=1,
            run_id="run-1",
            step_id="step-1",
            start_time=datetime(2026, 4, 25, 0, 0, 2, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, 0, 0, 2, tzinfo=timezone.utc),
            duration_ms=10.0,
            status=SpanStatus.OK,
            attributes={"agent_id": "agent-1", "sequence": 6},
            tool_details={
                "tool_name": "bash",
                "tool_call_id": "tc-1",
                "output": (
                    "<system-review>\n"
                    'Active milestone: "Inspect auth"\n\n'
                    "Trigger: step_interval\n"
                    "Steps since last review: 8\n"
                    "Hook advice: narrow the search\n"
                    "</system-review>"
                ),
                "status": "completed",
            },
        ),
        Span(
            trace_id="trace-1",
            span_id="runtime-review-trigger",
            parent_span_id="run-1-root",
            kind=SpanKind.RUNTIME,
            name="review_trigger",
            depth=1,
            run_id="run-1",
            start_time=datetime(2026, 4, 25, 0, 0, 2, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, 0, 0, 2, tzinfo=timezone.utc),
            duration_ms=0.0,
            status=SpanStatus.OK,
            attributes={
                "agent_id": "agent-1",
                "sequence": 6,
                "trigger_reason": "step_interval",
                "active_milestone_id": "inspect",
                "review_count_since_checkpoint": 8,
                "trigger_tool_call_id": "tc-1",
                "trigger_tool_step_id": "step-1",
                "notice_step_id": "step-1",
            },
        ),
        Span(
            trace_id="trace-1",
            span_id="tool-2",
            parent_span_id="run-1-root",
            kind=SpanKind.TOOL_CALL,
            name="review_trajectory",
            depth=1,
            run_id="run-1",
            step_id="step-2",
            start_time=datetime(2026, 4, 25, 0, 0, 3, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, 0, 0, 3, tzinfo=timezone.utc),
            duration_ms=10.0,
            status=SpanStatus.OK,
            attributes={"agent_id": "agent-1", "sequence": 7},
            tool_details={
                "tool_name": "review_trajectory",
                "tool_call_id": "tc-review",
                "input_args": {"aligned": False, "experience": "Auth grep was noisy."},
                "output": "Trajectory review: aligned=false. Auth grep was noisy.",
                "status": "completed",
            },
        ),
        Span(
            trace_id="trace-1",
            span_id="runtime-review-outcome",
            parent_span_id="run-1-root",
            kind=SpanKind.RUNTIME,
            name="review_outcome",
            depth=1,
            run_id="run-1",
            start_time=datetime(2026, 4, 25, 0, 0, 3, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, 0, 0, 3, tzinfo=timezone.utc),
            duration_ms=0.0,
            status=SpanStatus.OK,
            attributes={
                "agent_id": "agent-1",
                "sequence": 7,
                "aligned": False,
                "mode": "step_back",
                "experience": "switch plan",
                "active_milestone_id": "inspect",
                "review_tool_call_id": "tc-review",
                "review_step_id": "step-2",
                "hidden_step_ids": ["assistant-review", "step-2"],
                "notice_cleaned_step_ids": [],
                "condensed_step_ids": ["step-a", "step-b"],
            },
        ),
        Span(
            trace_id="trace-1",
            span_id="runtime-1",
            parent_span_id="run-1-root",
            kind=SpanKind.RUNTIME,
            name="step_back",
            depth=1,
            run_id="run-1",
            start_time=datetime(2026, 4, 25, 0, 0, 4, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, 0, 0, 4, tzinfo=timezone.utc),
            duration_ms=0.0,
            status=SpanStatus.OK,
            attributes={
                "agent_id": "agent-1",
                "sequence": 8,
                "affected_count": 2,
                "checkpoint_seq": 4,
                "experience": "switch plan",
            },
        ),
    ]
    return trace


def test_build_trace_llm_call_records_extracts_summary_fields() -> None:
    trace = _trace_with_review_and_llm()

    records = build_trace_llm_call_records(trace)

    assert len(records) == 1
    record = records[0]
    assert record.span_id == "llm-1"
    assert record.model == "gpt-5.4"
    assert record.provider == "openai-response"
    assert record.finish_reason == "stop"
    assert record.input_tokens == 100
    assert record.response_tool_call_count == 1
    assert record.output_preview == "Aligned."


def test_build_trace_review_cycles_groups_checkpoint_result_and_step_back() -> None:
    trace = _trace_with_review_and_llm()

    cycles = build_trace_review_cycles(trace)

    assert len(cycles) == 1
    cycle = cycles[0]
    assert cycle.cycle_id == "run-1:6:runtime-review-trigger"
    assert cycle.trigger_reason == "step_interval"
    assert cycle.steps_since_last_review == 8
    assert cycle.active_milestone == "Inspect auth"
    assert cycle.aligned is False
    assert cycle.experience == "switch plan"
    assert cycle.step_back_applied is True
    assert cycle.affected_count == 2


def test_build_trace_review_cycles_disambiguates_missing_sequences() -> None:
    trace = _trace_with_review_and_llm()
    trace.spans = [span for span in trace.spans if span.kind != SpanKind.RUNTIME]
    for index, span in enumerate(trace.spans):
        if span.kind == SpanKind.TOOL_CALL:
            span.attributes.pop("sequence", None)
            span.span_id = f"review-notice-{index}"
    trace.spans.append(
        Span(
            trace_id="trace-1",
            span_id="review-notice-extra",
            parent_span_id="run-1-root",
            kind=SpanKind.TOOL_CALL,
            name="bash",
            depth=1,
            run_id="run-1",
            step_id="step-extra",
            start_time=datetime(2026, 4, 25, 0, 0, 5, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, 0, 0, 5, tzinfo=timezone.utc),
            duration_ms=10.0,
            status=SpanStatus.OK,
            attributes={"agent_id": "agent-1"},
            tool_details={
                "tool_name": "bash",
                "tool_call_id": "tc-extra",
                "output": (
                    "<system-review>\n"
                    'Active milestone: "Inspect auth"\n\n'
                    "Trigger: token_pressure\n"
                    "Steps since last review: 2\n"
                    "</system-review>"
                ),
                "status": "completed",
            },
        )
    )

    cycles = build_trace_review_cycles(trace)

    assert cycles == []


def test_build_trace_mainline_events_returns_readable_narrative_sequence() -> None:
    trace = _trace_with_review_and_llm()

    events = build_trace_mainline_events(trace)

    assert [event.kind for event in events] == [
        "run_started",
        "milestone_update",
        "review_checkpoint",
        "review_result",
        "runtime_decision",
        "run_finished",
    ]
    assert events[2].summary == "triggered by step_interval after 8 steps"
    assert events[3].summary == "trajectory misaligned"
    assert events[4].details["kind"] == "step_back"


def test_build_session_milestone_board_and_conversation_events() -> None:
    trace = _trace_with_review_and_llm()
    review_cycles = build_trace_review_cycles(trace)

    board = build_session_milestone_board(
        session_id="sess-1",
        trace=trace,
        review_cycles=review_cycles,
    )

    assert board is not None
    assert review_cycles[0].active_milestone_id == "inspect"
    assert board.active_milestone_id == "inspect"
    assert board.milestones[0].description == "Inspect auth"
    assert board.latest_review_outcome is not None
    assert board.latest_review_outcome.step_back_applied is True

    steps = [
        StepView(
            id="step-user",
            session_id="sess-1",
            run_id="run-1",
            sequence=1,
            role=MessageRole.USER,
            content="please inspect auth",
            user_input="please inspect auth",
        ),
        StepView(
            id="step-assistant",
            session_id="sess-1",
            run_id="run-1",
            sequence=2,
            role=MessageRole.ASSISTANT,
            content="I will inspect auth",
        ),
        StepView(
            id="step-tool",
            session_id="sess-1",
            run_id="run-1",
            sequence=3,
            role=MessageRole.TOOL,
            name="declare_milestones",
            content="Milestones declared: inspect, fix",
        ),
    ]

    events = build_conversation_events(
        session_id="sess-1",
        steps=steps,
        review_cycles=review_cycles,
    )

    assert [event.kind for event in events] == [
        "user_message",
        "assistant_message",
        "milestone_event",
        "review_event",
    ]
    assert events[-1].summary == "Review misaligned; 2 steps condensed"
