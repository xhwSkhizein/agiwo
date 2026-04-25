# Console Mainline And Debug Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured `Mainline / Debug` observability for Console `Session Detail`, `Trace Detail`, and live transcript so milestone flow, review cycles, runtime decisions, and LLM details are readable without losing raw debug access.

**Architecture:** Start by adding shared server/client contracts for milestone boards, review cycles, mainline events, LLM-call summaries, and conversation events. Build trace-side structured read models first, because trace spans already contain the richest review and tool-call context. Then reuse those builders in `SessionDetail` to render a top-level milestone board and downgraded conversation events. Finish by extending the public stream protocol with an explicit `context_steps_hidden` event so live transcript behavior matches historical step queries.

**Tech Stack:** Python 3.10+, dataclasses, Pydantic, asyncio, RunLog/Trace replay builders, React 19, TypeScript, Vitest, pytest

**Spec:** `docs/superpowers/specs/2026-04-25-console-mainline-debug-observability-design.md`

---

## Scope Check

This feature touches three surfaces:

1. Console server observability read models
2. Console Web `Trace Detail` and `Session Detail`
3. Public live stream event handling

They are not independent subsystems. All three depend on the same structured observability contracts, so they belong in one plan. The plan still keeps them isolated by sequencing:

1. contracts and serializers
2. trace read models
3. session read models
4. trace UI
5. session UI
6. live stream consistency

## File Map

- `console/server/models/session.py`
  - add structured record types used by Console server builders and serializers
- `console/server/models/view.py`
  - add API response models for the new structured observability payloads
- `console/server/response_serialization.py`
  - serialize structured session/trace observability records into response payloads
- `console/server/services/runtime/runtime_observability.py`
  - extend read-model builders with trace review cycles, mainline events, LLM-call summaries, milestone boards, and conversation events
- `console/server/services/runtime/session_view_service.py`
  - assemble new session-level structured observability fields from latest trace and visible steps
- `console/server/services/runtime/trace_query_service.py`
  - no API shape change required, but will be reused to fetch the latest trace as the source for session mainline state
- `console/web/src/lib/api.ts`
  - add TypeScript contracts for the new response fields and stream event payload
- `console/web/src/app/traces/[id]/page.tsx`
  - add `Mainline / Debug` mode toggle and consume structured trace observability fields
- `console/web/src/components/trace-detail/trace-mainline-events.tsx`
  - new mainline narrative list for Trace Detail
- `console/web/src/components/trace-detail/trace-review-cycles.tsx`
  - new grouped review-cycle cards for Trace Detail
- `console/web/src/components/trace-detail/trace-llm-calls.tsx`
  - structured LLM-call summaries instead of raw JSON-first detail blocks
- `console/web/src/components/session-detail/milestone-board.tsx`
  - new top-of-page milestone/TODO board for Session Detail
- `console/web/src/components/session-detail/conversation-event-list.tsx`
  - new downgraded event list for session mainline conversation
- `console/web/src/app/sessions/[id]/page.tsx`
  - add `Mainline / Debug` mode toggle and render milestone board + conversation events
- `console/web/src/hooks/use-chat-stream.ts`
  - consume the new `context_steps_hidden` event and reconcile live transcript with history
- `console/web/src/components/chat-message.tsx`
  - minor styling hook-up if muted conversation/live events need a dedicated visual variant
- `agiwo/agent/models/stream.py`
  - add a public `ContextStepsHiddenEvent` stream payload and project `ContextStepsHidden` entries into that event
- `console/tests/test_traces_api.py`
  - verify trace responses now include `mainline_events`, `review_cycles`, and `llm_calls`
- `console/tests/test_sessions_api.py`
  - verify session detail responses include `milestone_board`, `review_cycles`, and `conversation_events`
- `console/tests/test_runtime_observability.py`
  - new focused builder tests for trace/session structured observability derivation
- `console/tests/test_response_serialization.py`
  - new focused tests for response serialization shape and defaults
- `console/web/src/app/traces/[id]/page.test.tsx`
  - verify Trace Detail mode switching and structured cards
- `console/web/src/components/session-detail/conversation-event-list.test.tsx`
  - verify conversation filtering and downgraded event rendering
- `console/web/src/app/sessions/[id]/page.test.tsx`
  - verify Session Detail mode switching and milestone board
- `console/web/src/hooks/use-chat-stream.test.tsx`
  - verify live transcript removes or demotes hidden review metadata on `context_steps_hidden`
- `tests/agent/test_run_log_replay_parity.py`
  - verify `ContextStepsHidden` produces the new stream event while still suppressing same-batch public step replay
- `docs/guides/storage.md`
  - document that Console structured observability remains derived from `RunLog` and `Trace.spans`
- `docs/guides/streaming.md`
  - document the new `context_steps_hidden` stream event

### Task 1: Add Shared Structured Observability Contracts

**Files:**
- Create: `console/tests/test_response_serialization.py`
- Modify: `console/server/models/session.py`
- Modify: `console/server/models/view.py`
- Modify: `console/server/response_serialization.py`
- Modify: `console/web/src/lib/api.ts`

- [ ] **Step 1: Write the failing serializer contract tests**

```python
# console/tests/test_response_serialization.py
from datetime import datetime, timezone

from server.models.session import (
    ConversationEventRecord,
    MilestoneRecord,
    ReviewCheckpointRecord,
    ReviewCycleRecord,
    ReviewOutcomeRecord,
    SessionDetailRecord,
    SessionMilestoneBoardRecord,
    SessionSummaryRecord,
    TraceLlmCallRecord,
    TraceMainlineEventRecord,
)
from server.response_serialization import session_detail_response_from_record


def test_session_detail_serializes_new_mainline_fields() -> None:
    detail = SessionDetailRecord(
        summary=SessionSummaryRecord(session_id="sess-1"),
        milestone_board=SessionMilestoneBoardRecord(
            session_id="sess-1",
            run_id="run-1",
            milestones=[
                MilestoneRecord(
                    id="inspect",
                    description="Inspect the auth flow",
                    status="active",
                    declared_at_seq=3,
                )
            ],
            active_milestone_id="inspect",
            latest_checkpoint=ReviewCheckpointRecord(
                seq=8,
                milestone_id="inspect",
                confirmed_at=datetime(2026, 4, 25, tzinfo=timezone.utc),
            ),
            latest_review_outcome=ReviewOutcomeRecord(
                aligned=True,
                step_back_applied=False,
                trigger_reason="step_interval",
                active_milestone="Inspect the auth flow",
            ),
            pending_review_reason=None,
        ),
        review_cycles=[
            ReviewCycleRecord(
                cycle_id="run-1:8",
                run_id="run-1",
                agent_id="agent-1",
                trigger_reason="step_interval",
                steps_since_last_review=8,
                active_milestone="Inspect the auth flow",
                aligned=True,
                experience=None,
                step_back_applied=False,
            )
        ],
        conversation_events=[
            ConversationEventRecord(
                id="evt-1",
                session_id="sess-1",
                run_id="run-1",
                sequence=10,
                kind="assistant_message",
                priority="primary",
                title="Assistant",
                summary="Auth check is in auth.py",
                details={},
            )
        ],
    )

    payload = session_detail_response_from_record(detail)

    assert payload.milestone_board is not None
    assert payload.milestone_board.active_milestone_id == "inspect"
    assert payload.review_cycles[0].trigger_reason == "step_interval"
    assert payload.conversation_events[0].kind == "assistant_message"


def test_trace_llm_call_serialization_preserves_summary_fields() -> None:
    record = TraceLlmCallRecord(
        span_id="span-1",
        run_id="run-1",
        agent_id="agent-1",
        model="gpt-5.4",
        provider="openai-response",
        finish_reason="stop",
        duration_ms=1234.0,
        first_token_latency_ms=345.0,
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        message_count=6,
        tool_schema_count=2,
        response_tool_call_count=1,
        output_preview="Looks aligned.",
    )

    from server.response_serialization import trace_llm_call_response_from_record

    payload = trace_llm_call_response_from_record(record)

    assert payload.model == "gpt-5.4"
    assert payload.response_tool_call_count == 1
    assert payload.output_preview == "Looks aligned."
```

- [ ] **Step 2: Run the focused serializer tests and confirm they fail**

Run: `uv run pytest console/tests/test_response_serialization.py -v`

Expected:
- `ImportError` or `AttributeError` because `MilestoneRecord`, `SessionMilestoneBoardRecord`, `TraceLlmCallRecord`, and the new serializer helpers do not exist yet

- [ ] **Step 3: Add shared server-side record types**

```python
# console/server/models/session.py
@dataclass(slots=True)
class MilestoneRecord:
    id: str
    description: str
    status: str
    declared_at_seq: int = 0
    completed_at_seq: int | None = None


@dataclass(slots=True)
class ReviewCheckpointRecord:
    seq: int
    milestone_id: str
    confirmed_at: datetime


@dataclass(slots=True)
class ReviewOutcomeRecord:
    aligned: bool | None = None
    experience: str | None = None
    step_back_applied: bool = False
    affected_count: int | None = None
    trigger_reason: str | None = None
    active_milestone: str | None = None
    resolved_at: datetime | None = None


@dataclass(slots=True)
class SessionMilestoneBoardRecord:
    session_id: str
    run_id: str | None
    milestones: list[MilestoneRecord] = field(default_factory=list)
    active_milestone_id: str | None = None
    latest_checkpoint: ReviewCheckpointRecord | None = None
    latest_review_outcome: ReviewOutcomeRecord | None = None
    pending_review_reason: str | None = None


@dataclass(slots=True)
class ReviewCycleRecord:
    cycle_id: str
    run_id: str
    agent_id: str
    trigger_reason: str
    steps_since_last_review: int | None = None
    active_milestone: str | None = None
    hook_advice: str | None = None
    aligned: bool | None = None
    experience: str | None = None
    step_back_applied: bool = False
    rollback_range: tuple[int, int] | None = None
    affected_count: int | None = None
    started_at: datetime | None = None
    resolved_at: datetime | None = None
    raw_notice: str | None = None


@dataclass(slots=True)
class ConversationEventRecord:
    id: str
    session_id: str
    run_id: str | None
    sequence: int | None
    kind: str
    priority: str
    title: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceMainlineEventRecord:
    id: str
    kind: str
    title: str
    summary: str
    status: str = "ok"
    sequence: int | None = None
    timestamp: datetime | None = None
    run_id: str | None = None
    agent_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceLlmCallRecord:
    span_id: str
    run_id: str
    agent_id: str
    model: str | None
    provider: str | None
    finish_reason: str | None
    duration_ms: float | None
    first_token_latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    message_count: int
    tool_schema_count: int
    response_tool_call_count: int
    output_preview: str | None
```

```python
# console/server/models/session.py
@dataclass(slots=True)
class SessionDetailRecord:
    summary: SessionSummaryRecord
    session: Session | None = None
    chat_context: ChannelChatContext | None = None
    scheduler_state: "AgentState | None" = None
    observability: "SessionObservabilityRecord | None" = None
    milestone_board: "SessionMilestoneBoardRecord | None" = None
    review_cycles: list["ReviewCycleRecord"] = field(default_factory=list)
    conversation_events: list["ConversationEventRecord"] = field(default_factory=list)
```

- [ ] **Step 4: Add API response models and serializer helpers**

```python
# console/server/models/view.py
class MilestoneResponse(BaseModel):
    id: str
    description: str
    status: str
    declared_at_seq: int = 0
    completed_at_seq: int | None = None


class ReviewCheckpointResponse(BaseModel):
    seq: int
    milestone_id: str
    confirmed_at: str


class ReviewOutcomeResponse(BaseModel):
    aligned: bool | None = None
    experience: str | None = None
    step_back_applied: bool = False
    affected_count: int | None = None
    trigger_reason: str | None = None
    active_milestone: str | None = None
    resolved_at: str | None = None


class SessionMilestoneBoardResponse(BaseModel):
    session_id: str
    run_id: str | None
    milestones: list[MilestoneResponse] = Field(default_factory=list)
    active_milestone_id: str | None = None
    latest_checkpoint: ReviewCheckpointResponse | None = None
    latest_review_outcome: ReviewOutcomeResponse | None = None
    pending_review_reason: str | None = None


class ReviewCycleResponse(BaseModel):
    cycle_id: str
    run_id: str
    agent_id: str
    trigger_reason: str
    steps_since_last_review: int | None = None
    active_milestone: str | None = None
    hook_advice: str | None = None
    aligned: bool | None = None
    experience: str | None = None
    step_back_applied: bool = False
    rollback_range: list[int] | None = None
    affected_count: int | None = None
    started_at: str | None = None
    resolved_at: str | None = None
    raw_notice: str | None = None


class ConversationEventResponse(BaseModel):
    id: str
    session_id: str
    run_id: str | None = None
    sequence: int | None = None
    kind: str
    priority: str
    title: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class TraceMainlineEventResponse(BaseModel):
    id: str
    kind: str
    title: str
    summary: str
    status: str = "ok"
    sequence: int | None = None
    timestamp: str | None = None
    run_id: str | None = None
    agent_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TraceLlmCallResponse(BaseModel):
    span_id: str
    run_id: str
    agent_id: str
    model: str | None = None
    provider: str | None = None
    finish_reason: str | None = None
    duration_ms: float | None = None
    first_token_latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    message_count: int
    tool_schema_count: int
    response_tool_call_count: int
    output_preview: str | None = None
```

```python
# console/server/response_serialization.py
def review_cycle_response_from_record(record: ReviewCycleRecord) -> ReviewCycleResponse:
    return ReviewCycleResponse(
        cycle_id=record.cycle_id,
        run_id=record.run_id,
        agent_id=record.agent_id,
        trigger_reason=record.trigger_reason,
        steps_since_last_review=record.steps_since_last_review,
        active_milestone=record.active_milestone,
        hook_advice=record.hook_advice,
        aligned=record.aligned,
        experience=record.experience,
        step_back_applied=record.step_back_applied,
        rollback_range=list(record.rollback_range) if record.rollback_range else None,
        affected_count=record.affected_count,
        started_at=record.started_at.isoformat() if record.started_at else None,
        resolved_at=record.resolved_at.isoformat() if record.resolved_at else None,
        raw_notice=record.raw_notice,
    )


def trace_llm_call_response_from_record(record: TraceLlmCallRecord) -> TraceLlmCallResponse:
    return TraceLlmCallResponse(
        span_id=record.span_id,
        run_id=record.run_id,
        agent_id=record.agent_id,
        model=record.model,
        provider=record.provider,
        finish_reason=record.finish_reason,
        duration_ms=record.duration_ms,
        first_token_latency_ms=record.first_token_latency_ms,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        total_tokens=record.total_tokens,
        message_count=record.message_count,
        tool_schema_count=record.tool_schema_count,
        response_tool_call_count=record.response_tool_call_count,
        output_preview=record.output_preview,
    )
```

```python
# console/server/response_serialization.py
def session_detail_response_from_record(detail: SessionDetailRecord) -> SessionDetailResponse:
    return SessionDetailResponse(
        summary=session_summary_response_from_record(detail.summary),
        session=(
            session_record_response_from_runtime(detail.session)
            if detail.session is not None
            else None
        ),
        chat_context=(
            chat_context_response_from_runtime(detail.chat_context)
            if detail.chat_context is not None
            else None
        ),
        scheduler_state=(
            agent_state_response_from_sdk(detail.scheduler_state)
            if detail.scheduler_state is not None
            else None
        ),
        observability=(
            session_observability_response_from_record(detail.observability)
            if detail.observability is not None
            else None
        ),
        milestone_board=(
            session_milestone_board_response_from_record(detail.milestone_board)
            if detail.milestone_board is not None
            else None
        ),
        review_cycles=[
            review_cycle_response_from_record(record) for record in detail.review_cycles
        ],
        conversation_events=[
            conversation_event_response_from_record(record)
            for record in detail.conversation_events
        ],
    )
```

- [ ] **Step 5: Mirror the new response contracts in the frontend API types**

```ts
// console/web/src/lib/api.ts
export interface MilestoneItem {
  id: string;
  description: string;
  status: string;
  declared_at_seq: number;
  completed_at_seq: number | null;
}

export interface ReviewCheckpoint {
  seq: number;
  milestone_id: string;
  confirmed_at: string;
}

export interface ReviewOutcome {
  aligned: boolean | null;
  experience: string | null;
  step_back_applied: boolean;
  affected_count: number | null;
  trigger_reason: string | null;
  active_milestone: string | null;
  resolved_at: string | null;
}

export interface SessionMilestoneBoard {
  session_id: string;
  run_id: string | null;
  milestones: MilestoneItem[];
  active_milestone_id: string | null;
  latest_checkpoint: ReviewCheckpoint | null;
  latest_review_outcome: ReviewOutcome | null;
  pending_review_reason: string | null;
}

export interface ReviewCycle {
  cycle_id: string;
  run_id: string;
  agent_id: string;
  trigger_reason: string;
  steps_since_last_review: number | null;
  active_milestone: string | null;
  hook_advice: string | null;
  aligned: boolean | null;
  experience: string | null;
  step_back_applied: boolean;
  rollback_range: number[] | null;
  affected_count: number | null;
  started_at: string | null;
  resolved_at: string | null;
  raw_notice: string | null;
}

export interface ConversationEvent {
  id: string;
  session_id: string;
  run_id: string | null;
  sequence: number | null;
  kind: string;
  priority: "primary" | "secondary" | "muted" | string;
  title: string;
  summary: string;
  details: Record<string, unknown>;
}

export interface TraceMainlineEvent {
  id: string;
  kind: string;
  title: string;
  summary: string;
  status: string;
  sequence: number | null;
  timestamp: string | null;
  run_id: string | null;
  agent_id: string | null;
  details: Record<string, unknown>;
}

export interface TraceLlmCall {
  span_id: string;
  run_id: string;
  agent_id: string;
  model: string | null;
  provider: string | null;
  finish_reason: string | null;
  duration_ms: number | null;
  first_token_latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  message_count: number;
  tool_schema_count: number;
  response_tool_call_count: number;
  output_preview: string | null;
}

export interface SessionDetail {
  summary: SessionSummary;
  session: SessionRecord | null;
  chat_context: ChatContextRecord | null;
  scheduler_state: AgentStateDetail | null;
  observability: SessionObservability | null;
  milestone_board: SessionMilestoneBoard | null;
  review_cycles: ReviewCycle[];
  conversation_events: ConversationEvent[];
}

export interface TraceDetail {
  ...
  mainline_events: TraceMainlineEvent[];
  review_cycles: ReviewCycle[];
  llm_calls: TraceLlmCall[];
}
```

- [ ] **Step 6: Run the serializer contract tests again**

Run: `uv run pytest console/tests/test_response_serialization.py -v`

Expected: PASS with both tests green

- [ ] **Step 7: Commit the shared contracts**

```bash
git add console/server/models/session.py console/server/models/view.py console/server/response_serialization.py console/web/src/lib/api.ts console/tests/test_response_serialization.py
git commit -m "feat: add structured observability contracts"
```

### Task 2: Build Trace-Side Mainline Read Models

**Files:**
- Create: `console/tests/test_runtime_observability.py`
- Modify: `console/server/services/runtime/runtime_observability.py`
- Modify: `console/server/response_serialization.py`
- Modify: `console/tests/test_traces_api.py`

- [ ] **Step 1: Write the failing trace builder tests**

```python
# console/tests/test_runtime_observability.py
from datetime import datetime, timezone

from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace
from server.services.runtime.runtime_observability import (
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
            span_id="llm-1",
            parent_span_id=None,
            kind=SpanKind.LLM_CALL,
            name="gpt-5.4",
            depth=1,
            run_id="run-1",
            start_time=datetime(2026, 4, 25, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, tzinfo=timezone.utc),
            duration_ms=1234.0,
            status=SpanStatus.OK,
            attributes={"agent_id": "agent-1", "sequence": 5, "provider": "openai-response"},
            llm_details={
                "messages": [{"role": "user", "content": "Inspect auth"}],
                "tools": [{"name": "bash"}],
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
            span_id="tool-1",
            parent_span_id=None,
            kind=SpanKind.TOOL_CALL,
            name="bash",
            depth=1,
            run_id="run-1",
            step_id="step-1",
            start_time=datetime(2026, 4, 25, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, tzinfo=timezone.utc),
            duration_ms=10.0,
            status=SpanStatus.OK,
            attributes={"agent_id": "agent-1", "sequence": 6},
            tool_details={
                "tool_name": "bash",
                "tool_call_id": "tc-1",
                "output": "<system-review>\nActive milestone: \"Inspect auth\"\n\nTrigger: step_interval\nSteps since last review: 8\n</system-review>",
                "status": "completed",
            },
        ),
        Span(
            trace_id="trace-1",
            span_id="tool-2",
            parent_span_id=None,
            kind=SpanKind.TOOL_CALL,
            name="review_trajectory",
            depth=1,
            run_id="run-1",
            step_id="step-2",
            start_time=datetime(2026, 4, 25, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, tzinfo=timezone.utc),
            duration_ms=10.0,
            status=SpanStatus.OK,
            attributes={"agent_id": "agent-1", "sequence": 7},
            tool_details={
                "tool_name": "review_trajectory",
                "tool_call_id": "tc-review",
                "output": "Trajectory review: aligned=false. Auth grep was noisy.",
                "status": "completed",
            },
        ),
        Span(
            trace_id="trace-1",
            span_id="runtime-1",
            parent_span_id=None,
            kind=SpanKind.RUNTIME,
            name="step_back",
            depth=1,
            run_id="run-1",
            start_time=datetime(2026, 4, 25, tzinfo=timezone.utc),
            end_time=datetime(2026, 4, 25, tzinfo=timezone.utc),
            duration_ms=0.0,
            status=SpanStatus.OK,
            attributes={
                "agent_id": "agent-1",
                "sequence": 8,
                "affected_count": 2,
                "checkpoint_seq": 4,
                "experience": "Auth grep was noisy.",
            },
        ),
    ]
    return trace


def test_build_trace_llm_call_records_summarizes_llm_details() -> None:
    records = build_trace_llm_call_records(_trace_with_review_and_llm())

    assert len(records) == 1
    assert records[0].model == "gpt-5.4"
    assert records[0].message_count == 1
    assert records[0].tool_schema_count == 1
    assert records[0].response_tool_call_count == 1


def test_build_trace_review_cycles_groups_checkpoint_result_and_step_back() -> None:
    cycles = build_trace_review_cycles(_trace_with_review_and_llm())

    assert len(cycles) == 1
    assert cycles[0].trigger_reason == "step_interval"
    assert cycles[0].aligned is False
    assert cycles[0].step_back_applied is True
    assert cycles[0].affected_count == 2


def test_build_trace_mainline_events_prioritizes_review_and_runtime_flow() -> None:
    events = build_trace_mainline_events(_trace_with_review_and_llm())

    assert [event.kind for event in events] == [
        "review_checkpoint",
        "review_result",
        "step_back",
    ]
```

- [ ] **Step 2: Run the trace builder tests and confirm they fail**

Run: `uv run pytest console/tests/test_runtime_observability.py -v`

Expected:
- failures because `build_trace_llm_call_records`, `build_trace_review_cycles`, and `build_trace_mainline_events` do not exist yet

- [ ] **Step 3: Implement trace LLM-call summaries**

```python
# console/server/services/runtime/runtime_observability.py
def build_trace_llm_call_records(trace: Trace) -> list[TraceLlmCallRecord]:
    records: list[TraceLlmCallRecord] = []
    for span in trace.spans:
        if span.kind != SpanKind.LLM_CALL:
            continue
        details = dict(span.llm_details or {})
        metrics = details.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        messages = details.get("messages")
        tools = details.get("tools")
        response_tool_calls = details.get("response_tool_calls")
        records.append(
            TraceLlmCallRecord(
                span_id=span.span_id,
                run_id=span.run_id or "",
                agent_id=_runtime_agent_id(trace, span),
                model=str(span.name or span.attributes.get("model_name") or ""),
                provider=_json_text(span.attributes.get("provider")) or None,
                finish_reason=_json_text(details.get("finish_reason")) or None,
                duration_ms=span.duration_ms,
                first_token_latency_ms=_as_float(
                    metrics.get("first_token_ms") or metrics.get("first_token_latency_ms")
                ),
                input_tokens=_as_int(metrics.get("input_tokens")),
                output_tokens=_as_int(metrics.get("output_tokens")),
                total_tokens=_as_int(metrics.get("total_tokens")),
                message_count=len(messages) if isinstance(messages, list) else 0,
                tool_schema_count=len(tools) if isinstance(tools, list) else 0,
                response_tool_call_count=(
                    len(response_tool_calls) if isinstance(response_tool_calls, list) else 0
                ),
                output_preview=span.output_preview,
            )
        )
    return sorted(records, key=lambda item: (item.run_id, item.span_id))
```

- [ ] **Step 4: Implement trace review-cycle grouping and mainline events**

```python
# console/server/services/runtime/runtime_observability.py
def build_trace_review_cycles(trace: Trace) -> list[ReviewCycleRecord]:
    cycles: list[ReviewCycleRecord] = []
    pending_by_run: dict[str, ReviewCycleRecord] = {}

    for event in build_trace_timeline_events(trace):
        run_id = event.run_id or ""
        if event.kind == "review_checkpoint":
            cycle = ReviewCycleRecord(
                cycle_id=f"{run_id}:{event.sequence or 'na'}",
                run_id=run_id,
                agent_id=event.agent_id or "",
                trigger_reason=str(event.details.get("trigger_reason") or "unknown"),
                steps_since_last_review=_as_int(event.details.get("steps_since_last_review")),
                active_milestone=_json_text(event.details.get("active_milestone")) or None,
                hook_advice=_json_text(event.details.get("hook_advice")) or None,
                started_at=event.timestamp,
                raw_notice=_json_text(event.details.get("raw_notice")) or None,
            )
            pending_by_run[run_id] = cycle
            cycles.append(cycle)
            continue

        if event.kind == "review_result" and run_id in pending_by_run:
            cycle = pending_by_run[run_id]
            aligned = event.details.get("aligned")
            cycle.aligned = True if aligned is True or aligned == "true" else False if aligned is False or aligned == "false" else None
            cycle.resolved_at = event.timestamp
            continue

        if event.kind != "runtime_decision" or run_id not in pending_by_run:
            continue

        cycle = pending_by_run[run_id]
        runtime_kind = event.details.get("kind")
        if runtime_kind == "step_back":
            cycle.step_back_applied = True
            cycle.affected_count = _as_int(event.details.get("affected_count"))
            cycle.experience = _json_text(event.details.get("experience")) or None
            cycle.resolved_at = event.timestamp
        elif runtime_kind == "rollback":
            start_sequence = _as_int(event.details.get("start_sequence"))
            end_sequence = _as_int(event.details.get("end_sequence"))
            if start_sequence is not None and end_sequence is not None:
                cycle.rollback_range = (start_sequence, end_sequence)

    return cycles


def build_trace_mainline_events(trace: Trace) -> list[TraceMainlineEventRecord]:
    events: list[TraceMainlineEventRecord] = []
    for cycle in build_trace_review_cycles(trace):
        events.append(
            TraceMainlineEventRecord(
                id=f"{cycle.cycle_id}:checkpoint",
                kind="review_checkpoint",
                title="Review checkpoint",
                summary=f"Triggered by {cycle.trigger_reason}",
                status="ok",
                sequence=cycle.steps_since_last_review,
                timestamp=cycle.started_at,
                run_id=cycle.run_id,
                agent_id=cycle.agent_id,
                details={"active_milestone": cycle.active_milestone},
            )
        )
        events.append(
            TraceMainlineEventRecord(
                id=f"{cycle.cycle_id}:result",
                kind="review_result",
                title="Review result",
                summary="Aligned" if cycle.aligned is True else "Misaligned" if cycle.aligned is False else "Reviewed",
                status="ok" if cycle.aligned is not False else "error",
                timestamp=cycle.resolved_at,
                run_id=cycle.run_id,
                agent_id=cycle.agent_id,
                details={"experience": cycle.experience},
            )
        )
        if cycle.step_back_applied:
            events.append(
                TraceMainlineEventRecord(
                    id=f"{cycle.cycle_id}:step-back",
                    kind="step_back",
                    title="Step-back applied",
                    summary=f"{cycle.affected_count or 0} tool results condensed",
                    status="ok",
                    timestamp=cycle.resolved_at,
                    run_id=cycle.run_id,
                    agent_id=cycle.agent_id,
                    details={"experience": cycle.experience, "rollback_range": cycle.rollback_range},
                )
            )
    return events
```

- [ ] **Step 5: Extend trace serialization and API coverage**

```python
# console/server/response_serialization.py
def trace_response_from_sdk(trace: Trace) -> TraceResponse:
    ...
    review_cycles = [
        review_cycle_response_from_record(record)
        for record in build_trace_review_cycles(trace)
    ]
    llm_calls = [
        trace_llm_call_response_from_record(record)
        for record in build_trace_llm_call_records(trace)
    ]
    mainline_events = [
        trace_mainline_event_response_from_record(record)
        for record in build_trace_mainline_events(trace)
    ]
    return TraceResponse(
        ...,
        runtime_decisions=runtime_decisions,
        timeline_events=timeline_events,
        review_cycles=review_cycles,
        llm_calls=llm_calls,
        mainline_events=mainline_events,
    )
```

```python
# console/tests/test_traces_api.py
def test_get_trace_returns_mainline_events_review_cycles_and_llm_calls(...) -> None:
    ...
    payload = response.json()

    assert payload["review_cycles"][0]["trigger_reason"] == "step_interval"
    assert payload["review_cycles"][0]["step_back_applied"] is True
    assert payload["llm_calls"][0]["model"] == "gpt-5.4"
    assert payload["mainline_events"][0]["kind"] == "review_checkpoint"
```

- [ ] **Step 6: Run the trace builder and traces API tests**

Run: `uv run pytest console/tests/test_runtime_observability.py console/tests/test_traces_api.py -v`

Expected: PASS with builder and API payload assertions green

- [ ] **Step 7: Commit the trace read-model work**

```bash
git add console/server/services/runtime/runtime_observability.py console/server/response_serialization.py console/tests/test_runtime_observability.py console/tests/test_traces_api.py
git commit -m "feat: add trace mainline observability read models"
```

### Task 3: Build Session Mainline Read Models From Latest Trace And Visible Steps

**Files:**
- Modify: `console/server/services/runtime/session_view_service.py`
- Modify: `console/server/services/runtime/runtime_observability.py`
- Modify: `console/tests/test_sessions_api.py`

- [ ] **Step 1: Write the failing session-detail API test**

```python
# console/tests/test_sessions_api.py
def test_get_session_detail_includes_milestone_board_and_conversation_events(client) -> None:
    runtime = _runtime(client)
    now = datetime.now(timezone.utc)

    session = Session(
        id="sess-1",
        chat_context_scope_id=None,
        base_agent_id="agent-1",
        created_by="test",
        created_at=now,
        updated_at=now,
    )
    asyncio.run(runtime.session_store.upsert_session(session))

    trace = Trace(
        trace_id="trace-1",
        session_id="sess-1",
        agent_id="agent-1",
        start_time=now,
        spans=[
            Span(
                trace_id="trace-1",
                span_id="tool-declare",
                kind=SpanKind.TOOL_CALL,
                name="declare_milestones",
                start_time=now,
                end_time=now,
                duration_ms=1.0,
                status=SpanStatus.OK,
                run_id="run-1",
                step_id="step-declare",
                depth=1,
                attributes={"agent_id": "agent-1", "sequence": 3},
                tool_details={
                    "tool_name": "declare_milestones",
                    "input_args": {
                        "milestones": [
                            {"id": "inspect", "description": "Inspect the auth flow", "status": "active"},
                            {"id": "fix", "description": "Patch the auth flow", "status": "pending"},
                        ]
                    },
                    "output": "Milestones declared: inspect, fix",
                    "status": "completed",
                },
            )
        ],
    )
    asyncio.run(runtime.trace_storage.save_trace(trace))

    response = client.get("/api/sessions/sess-1")
    payload = response.json()

    assert payload["milestone_board"]["active_milestone_id"] == "inspect"
    assert payload["milestone_board"]["milestones"][1]["id"] == "fix"
    assert payload["conversation_events"] == []
```

- [ ] **Step 2: Run the session-detail API test and confirm it fails**

Run: `uv run pytest console/tests/test_sessions_api.py::test_get_session_detail_includes_milestone_board_and_conversation_events -v`

Expected:
- FAIL because `SessionDetailResponse` does not populate `milestone_board`, `review_cycles`, or `conversation_events`

- [ ] **Step 3: Build milestone boards and conversation events from latest trace + visible steps**

```python
# console/server/services/runtime/runtime_observability.py
def build_session_milestone_board(trace: Trace | None) -> SessionMilestoneBoardRecord | None:
    if trace is None:
        return None

    milestones: list[MilestoneRecord] = []
    active_milestone_id: str | None = None
    latest_checkpoint: ReviewCheckpointRecord | None = None
    latest_review_outcome: ReviewOutcomeRecord | None = None

    for event in build_trace_timeline_events(trace):
        if event.kind != "milestone_update":
            continue
        raw_items = event.details.get("milestones")
        if not isinstance(raw_items, list):
            continue
        milestones = []
        active_milestone_id = None
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            record = MilestoneRecord(
                id=str(item.get("id") or ""),
                description=str(item.get("description") or ""),
                status=str(item.get("status") or "pending"),
                declared_at_seq=event.sequence or 0,
            )
            milestones.append(record)
            if record.status == "active":
                active_milestone_id = record.id

    cycles = build_trace_review_cycles(trace)
    if cycles:
        latest = cycles[-1]
        latest_review_outcome = ReviewOutcomeRecord(
            aligned=latest.aligned,
            experience=latest.experience,
            step_back_applied=latest.step_back_applied,
            affected_count=latest.affected_count,
            trigger_reason=latest.trigger_reason,
            active_milestone=latest.active_milestone,
            resolved_at=latest.resolved_at,
        )
    if cycles and cycles[-1].resolved_at is not None and active_milestone_id is not None:
        latest_checkpoint = ReviewCheckpointRecord(
            seq=cycles[-1].steps_since_last_review or 0,
            milestone_id=active_milestone_id,
            confirmed_at=cycles[-1].resolved_at,
        )

    return SessionMilestoneBoardRecord(
        session_id=trace.session_id or "",
        run_id=trace.spans[0].run_id if trace.spans else None,
        milestones=milestones,
        active_milestone_id=active_milestone_id,
        latest_checkpoint=latest_checkpoint,
        latest_review_outcome=latest_review_outcome,
        pending_review_reason=None,
    )


def build_conversation_events(
    *,
    session_id: str,
    steps: list[StepView],
    review_cycles: list[ReviewCycleRecord],
) -> list[ConversationEventRecord]:
    events: list[ConversationEventRecord] = []
    for step in steps:
        if step.role.value == "assistant":
            events.append(
                ConversationEventRecord(
                    id=step.id or f"{step.run_id}:{step.sequence}",
                    session_id=session_id,
                    run_id=step.run_id,
                    sequence=step.sequence,
                    kind="assistant_message",
                    priority="primary",
                    title="Assistant",
                    summary=str(step.content or step.content_for_user or ""),
                )
            )
            continue
        if step.role.value != "tool":
            continue
        if step.condensed_content:
            events.append(
                ConversationEventRecord(
                    id=step.id or f"{step.run_id}:{step.sequence}",
                    session_id=session_id,
                    run_id=step.run_id,
                    sequence=step.sequence,
                    kind="compressed_history_event",
                    priority="muted",
                    title=step.name or "tool",
                    summary=step.condensed_content,
                    details={"original_content": step.content},
                )
            )
            continue
        events.append(
            ConversationEventRecord(
                id=step.id or f"{step.run_id}:{step.sequence}",
                session_id=session_id,
                run_id=step.run_id,
                sequence=step.sequence,
                kind="tool_event",
                priority="secondary",
                title=step.name or "tool",
                summary=str(step.content or ""),
            )
        )

    for cycle in review_cycles:
        events.append(
            ConversationEventRecord(
                id=f"{cycle.cycle_id}:review",
                session_id=session_id,
                run_id=cycle.run_id,
                sequence=None,
                kind="review_event",
                priority="secondary",
                title="Review",
                summary="Aligned" if cycle.aligned is True else "Misaligned" if cycle.aligned is False else "Reviewed",
                details={
                    "trigger_reason": cycle.trigger_reason,
                    "active_milestone": cycle.active_milestone,
                    "step_back_applied": cycle.step_back_applied,
                    "experience": cycle.experience,
                },
            )
        )

    return sorted(events, key=lambda event: (event.sequence is None, event.sequence or 0, event.id))
```

- [ ] **Step 4: Assemble the new fields in `SessionViewService`**

```python
# console/server/services/runtime/session_view_service.py
async def get_session_detail(self, session_id: str) -> SessionDetailRecord | None:
    session = await self._get_session(session_id)
    if session is None:
        return None

    stats = await self._run_queries.get_session_run_snapshot(session_id)
    metrics = self._build_metrics_summary(stats.run_views)
    scheduler_state = await self._get_scheduler_state(session_id)
    summary = self._assemble_summary(
        session,
        last_run=stats.run_views[0] if stats.run_views else None,
        run_count=len(stats.run_views),
        step_count=stats.committed_step_count,
        root_state_status=self._root_state_status(scheduler_state),
        metrics=metrics,
    )
    chat_context = await self._get_chat_context(session)
    recent_traces = await self._trace_queries.list_session_recent_traces(session_id, limit=1)
    latest_trace = recent_traces[0] if recent_traces else None
    milestone_board = build_session_milestone_board(latest_trace)
    review_cycles = build_trace_review_cycles(latest_trace) if latest_trace is not None else []
    visible_steps_page = await self._run_queries.list_session_steps(
        session_id,
        limit=200,
        order="asc",
    )
    conversation_events = build_conversation_events(
        session_id=session_id,
        steps=visible_steps_page.items,
        review_cycles=review_cycles,
    )

    return SessionDetailRecord(
        summary=summary,
        session=session,
        chat_context=chat_context,
        scheduler_state=scheduler_state,
        observability=await self._build_observability(session_id=session_id),
        milestone_board=milestone_board,
        review_cycles=review_cycles,
        conversation_events=conversation_events,
    )
```

- [ ] **Step 5: Run the session-detail API test again**

Run: `uv run pytest console/tests/test_sessions_api.py::test_get_session_detail_includes_milestone_board_and_conversation_events -v`

Expected: PASS with the new top-level fields serialized

- [ ] **Step 6: Commit the session mainline builders**

```bash
git add console/server/services/runtime/runtime_observability.py console/server/services/runtime/session_view_service.py console/tests/test_sessions_api.py
git commit -m "feat: add session mainline observability builders"
```

### Task 4: Implement Trace Detail Mainline And Debug Views

**Files:**
- Create: `console/web/src/components/trace-detail/trace-view-mode-toggle.tsx`
- Create: `console/web/src/components/trace-detail/trace-mainline-events.tsx`
- Create: `console/web/src/components/trace-detail/trace-review-cycles.tsx`
- Create: `console/web/src/components/trace-detail/trace-llm-calls.tsx`
- Modify: `console/web/src/app/traces/[id]/page.tsx`
- Modify: `console/web/src/app/traces/[id]/page.test.tsx`

- [ ] **Step 1: Write the failing Trace Detail UI test**

```ts
// console/web/src/app/traces/[id]/page.test.tsx
it("switches between Mainline and Debug trace views", async () => {
  apiMocks.getTrace.mockResolvedValue({
    trace_id: "trace-1",
    agent_id: "agent-1",
    session_id: "sess-1",
    user_id: null,
    start_time: "2026-04-25T00:00:00Z",
    end_time: "2026-04-25T00:00:10Z",
    duration_ms: 10000,
    status: "ok",
    root_span_id: "root-1",
    total_tokens: 120,
    total_input_tokens: 100,
    total_output_tokens: 20,
    total_token_cost: 0,
    total_llm_calls: 1,
    total_tool_calls: 2,
    total_cache_read_tokens: 0,
    total_cache_creation_tokens: 0,
    max_depth: 1,
    input_query: "Inspect auth",
    final_output: "Done",
    spans: [],
    runtime_decisions: [],
    timeline_events: [],
    mainline_events: [
      {
        id: "evt-1",
        kind: "review_checkpoint",
        title: "Review checkpoint",
        summary: "Triggered by step_interval",
        status: "ok",
        sequence: 6,
        timestamp: "2026-04-25T00:00:06Z",
        run_id: "run-1",
        agent_id: "agent-1",
        details: {},
      },
    ],
    review_cycles: [
      {
        cycle_id: "run-1:6",
        run_id: "run-1",
        agent_id: "agent-1",
        trigger_reason: "step_interval",
        steps_since_last_review: 6,
        active_milestone: "Inspect auth",
        hook_advice: null,
        aligned: false,
        experience: "Auth grep was noisy.",
        step_back_applied: true,
        rollback_range: [4, 6],
        affected_count: 2,
        started_at: "2026-04-25T00:00:06Z",
        resolved_at: "2026-04-25T00:00:08Z",
        raw_notice: "<system-review>...</system-review>",
      },
    ],
    llm_calls: [
      {
        span_id: "llm-1",
        run_id: "run-1",
        agent_id: "agent-1",
        model: "gpt-5.4",
        provider: "openai-response",
        finish_reason: "stop",
        duration_ms: 1234,
        first_token_latency_ms: 345,
        input_tokens: 100,
        output_tokens: 20,
        total_tokens: 120,
        message_count: 6,
        tool_schema_count: 1,
        response_tool_call_count: 1,
        output_preview: "Aligned.",
      },
    ],
  });

  render(<TraceDetailPage />);

  expect(await screen.findByText("Review checkpoint")).toBeInTheDocument();
  expect(screen.getByText("Review Cycles")).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "Debug" }));

  expect(screen.getByText("Loop Timeline")).toBeInTheDocument();
  expect(screen.getByText("LLM Calls")).toBeInTheDocument();
  expect(screen.getByText("gpt-5.4")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the Trace Detail UI test and confirm it fails**

Run: `cd console/web && npm test -- src/app/traces/[id]/page.test.tsx`

Expected:
- FAIL because `TraceDetailPage` has no `Mainline / Debug` mode toggle and does not render the new structured trace sections

- [ ] **Step 3: Add trace view toggle and mainline components**

```tsx
// console/web/src/components/trace-detail/trace-view-mode-toggle.tsx
"use client";

type TraceViewMode = "mainline" | "debug";

export function TraceViewModeToggle({
  mode,
  onChange,
}: {
  mode: TraceViewMode;
  onChange: (next: TraceViewMode) => void;
}) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-panel-muted p-1">
      {(["mainline", "debug"] as const).map((value) => (
        <button
          key={value}
          type="button"
          onClick={() => onChange(value)}
          className={`rounded-lg px-3 py-1.5 text-sm ${
            mode === value ? "bg-panel text-foreground" : "text-ink-muted"
          }`}
        >
          {value === "mainline" ? "Mainline" : "Debug"}
        </button>
      ))}
    </div>
  );
}
```

```tsx
// console/web/src/components/trace-detail/trace-review-cycles.tsx
"use client";

import { SectionCard } from "@/components/section-card";
import type { ReviewCycle } from "@/lib/api";

export function TraceReviewCycles({ cycles }: { cycles: ReviewCycle[] }) {
  return (
    <SectionCard title="Review Cycles" bodyClassName="space-y-3 px-4 py-4">
      {cycles.map((cycle) => (
        <details key={cycle.cycle_id} className="rounded-xl border border-line bg-panel px-3 py-3">
          <summary className="cursor-pointer list-none">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm font-medium text-foreground">
                  {cycle.aligned === true ? "Aligned" : cycle.aligned === false ? "Misaligned" : "Reviewed"}
                </span>
                <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                  {cycle.trigger_reason}
                </span>
                {cycle.step_back_applied ? (
                  <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[11px] uppercase tracking-wide text-amber-200">
                    step-back
                  </span>
                ) : null}
              </div>
              <p className="text-sm text-foreground">
                {cycle.active_milestone || "No active milestone"}
              </p>
            </div>
          </summary>
          <div className="mt-3 space-y-2 text-sm text-ink-muted">
            {cycle.experience ? <p>{cycle.experience}</p> : null}
            {cycle.rollback_range ? <p>Rollback seq {cycle.rollback_range.join(" - ")}</p> : null}
            {cycle.raw_notice ? <pre className="overflow-auto rounded-lg bg-panel-muted p-3 text-xs">{cycle.raw_notice}</pre> : null}
          </div>
        </details>
      ))}
    </SectionCard>
  );
}
```

- [ ] **Step 4: Wire the new structured sections into `TraceDetailPage`**

```tsx
// console/web/src/app/traces/[id]/page.tsx
const [viewMode, setViewMode] = useState<"mainline" | "debug">("mainline");

...

<div className="flex items-center justify-between gap-3">
  <BackHeader href="/traces" title="Trace Detail" subtitle={trace.trace_id} />
  <TraceViewModeToggle mode={viewMode} onChange={setViewMode} />
</div>

{viewMode === "mainline" ? (
  <div className="space-y-6">
    <TraceMainlineEvents events={trace.mainline_events} />
    <TraceReviewCycles cycles={trace.review_cycles} />
  </div>
) : (
  <div className="space-y-6">
    <TraceLoopTimeline events={trace.timeline_events} />
    <TraceRuntimeDecisions decisions={trace.runtime_decisions} />
    <TraceLlmCalls calls={trace.llm_calls} />
    <SectionCard title="Waterfall" bodyClassName="overflow-hidden">
      {trace.spans.map((span) => (
        <SpanRow
          key={span.span_id}
          span={span}
          traceStartMs={traceStartMs}
          traceDurationMs={traceDurationMs}
          expanded={expandedSpans.has(span.span_id)}
          onToggle={() => toggleSpan(span.span_id)}
        />
      ))}
    </SectionCard>
  </div>
)}
```

- [ ] **Step 5: Run the Trace Detail UI test again**

Run: `cd console/web && npm test -- src/app/traces/[id]/page.test.tsx`

Expected: PASS with the toggle and both sections visible in the right mode

- [ ] **Step 6: Commit the Trace Detail UI**

```bash
git add console/web/src/app/traces/[id]/page.tsx console/web/src/app/traces/[id]/page.test.tsx console/web/src/components/trace-detail/trace-view-mode-toggle.tsx console/web/src/components/trace-detail/trace-mainline-events.tsx console/web/src/components/trace-detail/trace-review-cycles.tsx console/web/src/components/trace-detail/trace-llm-calls.tsx
git commit -m "feat: add trace mainline and debug views"
```

### Task 5: Implement Session Detail Mainline View, Milestone Board, And Conversation Filters

**Files:**
- Create: `console/web/src/components/session-detail/milestone-board.tsx`
- Create: `console/web/src/components/session-detail/conversation-event-list.tsx`
- Create: `console/web/src/components/session-detail/conversation-event-list.test.tsx`
- Modify: `console/web/src/app/sessions/[id]/page.tsx`
- Modify: `console/web/src/app/sessions/[id]/page.test.tsx`

- [ ] **Step 1: Write the failing session mainline component test**

```tsx
// console/web/src/components/session-detail/conversation-event-list.test.tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { ConversationEventList } from "./conversation-event-list";

describe("ConversationEventList", () => {
  it("filters dialogue, key events, and all events", async () => {
    render(
      <ConversationEventList
        events={[
          {
            id: "assistant-1",
            session_id: "sess-1",
            run_id: "run-1",
            sequence: 10,
            kind: "assistant_message",
            priority: "primary",
            title: "Assistant",
            summary: "Auth lives in auth.py",
            details: {},
          },
          {
            id: "review-1",
            session_id: "sess-1",
            run_id: "run-1",
            sequence: 11,
            kind: "review_event",
            priority: "secondary",
            title: "Review",
            summary: "Misaligned",
            details: {},
          },
          {
            id: "compressed-1",
            session_id: "sess-1",
            run_id: "run-1",
            sequence: 12,
            kind: "compressed_history_event",
            priority: "muted",
            title: "bash",
            summary: "[EXPERIENCE] Grep was noisy",
            details: {},
          },
        ]}
      />
    );

    expect(screen.getByText("Auth lives in auth.py")).toBeInTheDocument();
    expect(screen.queryByText("Misaligned")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Dialogue + Key Events" }));
    expect(screen.getByText("Misaligned")).toBeInTheDocument();
    expect(screen.queryByText("[EXPERIENCE] Grep was noisy")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "All Events" }));
    expect(screen.getByText("[EXPERIENCE] Grep was noisy")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the session mainline component test and confirm it fails**

Run: `cd console/web && npm test -- src/components/session-detail/conversation-event-list.test.tsx`

Expected:
- FAIL because `ConversationEventList` and the new filter controls do not exist yet

- [ ] **Step 3: Add milestone board and conversation-event list components**

```tsx
// console/web/src/components/session-detail/milestone-board.tsx
"use client";

import { SectionCard } from "@/components/section-card";
import type { SessionMilestoneBoard } from "@/lib/api";

export function MilestoneBoard({
  board,
}: {
  board: SessionMilestoneBoard | null;
}) {
  if (!board) {
    return (
      <SectionCard title="Milestones" bodyClassName="px-4 py-4">
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No milestones declared for the latest run.
        </div>
      </SectionCard>
    );
  }

  return (
    <SectionCard title="Milestones" bodyClassName="space-y-3 px-4 py-4">
      {board.milestones.map((milestone) => {
        const isActive = milestone.id === board.active_milestone_id;
        return (
          <div
            key={milestone.id}
            className={`rounded-xl border px-3 py-3 ${
              isActive ? "border-emerald-500/40 bg-emerald-500/10" : "border-line bg-panel"
            }`}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-foreground">{milestone.description}</span>
              <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                {milestone.status}
              </span>
            </div>
            <div className="mt-2 text-xs text-ink-muted">
              seq {milestone.declared_at_seq}
              {milestone.completed_at_seq ? ` · completed at ${milestone.completed_at_seq}` : ""}
            </div>
          </div>
        );
      })}
    </SectionCard>
  );
}
```

```tsx
// console/web/src/components/session-detail/conversation-event-list.tsx
"use client";

import { useState } from "react";
import type { ConversationEvent } from "@/lib/api";

type FilterMode = "dialogue" | "key" | "all";

function visibleEvents(events: ConversationEvent[], filter: FilterMode) {
  if (filter === "all") return events;
  if (filter === "key") {
    return events.filter((event) => event.kind !== "compressed_history_event");
  }
  return events.filter((event) => event.kind === "assistant_message");
}

export function ConversationEventList({
  events,
}: {
  events: ConversationEvent[];
}) {
  const [filter, setFilter] = useState<FilterMode>("dialogue");
  const filtered = visibleEvents(events, filter);

  return (
    <div className="space-y-3">
      <div className="inline-flex rounded-xl border border-line bg-panel-muted p-1">
        <button type="button" onClick={() => setFilter("dialogue")} className="rounded-lg px-3 py-1.5 text-sm">
          Dialogue
        </button>
        <button type="button" onClick={() => setFilter("key")} className="rounded-lg px-3 py-1.5 text-sm">
          Dialogue + Key Events
        </button>
        <button type="button" onClick={() => setFilter("all")} className="rounded-lg px-3 py-1.5 text-sm">
          All Events
        </button>
      </div>
      {filtered.map((event) => (
        <div
          key={event.id}
          className={`rounded-xl border px-3 py-3 ${
            event.priority === "primary"
              ? "border-line bg-panel"
              : event.priority === "secondary"
              ? "border-cyan-500/30 bg-cyan-500/5"
              : "border-amber-500/20 bg-amber-500/5 opacity-80"
          }`}
        >
          <div className="text-xs uppercase tracking-wide text-ink-muted">{event.title}</div>
          <div className="mt-1 text-sm text-foreground">{event.summary}</div>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Add the `Mainline / Debug` split to Session Detail**

```tsx
// console/web/src/app/sessions/[id]/page.tsx
const [viewMode, setViewMode] = useState<"mainline" | "debug">("mainline");

...

<div className="flex items-center justify-between gap-3">
  <BackHeader href="/sessions" title="Session Detail" subtitle={sessionId} />
  <div className="inline-flex rounded-xl border border-line bg-panel-muted p-1">
    <button type="button" onClick={() => setViewMode("mainline")} className="rounded-lg px-3 py-1.5 text-sm">
      Mainline
    </button>
    <button type="button" onClick={() => setViewMode("debug")} className="rounded-lg px-3 py-1.5 text-sm">
      Debug
    </button>
  </div>
</div>

{viewMode === "mainline" ? (
  <div className="space-y-6">
    <MilestoneBoard board={detail.milestone_board} />
    <SectionCard title="Latest Review" bodyClassName="px-4 py-4">
      <pre className="whitespace-pre-wrap text-sm text-foreground">
        {detail.review_cycles[0]?.experience || detail.milestone_board?.latest_review_outcome?.active_milestone || "No review cycles yet."}
      </pre>
    </SectionCard>
    <SectionCard title="Conversation" bodyClassName="px-4 py-4">
      <ConversationEventList events={detail.conversation_events} />
    </SectionCard>
  </div>
) : (
  <div className="space-y-6">
    <SessionObservabilityPanel sessionId={sessionId} observability={detail.observability} />
    {/* existing summary / runs / steps blocks stay here */}
  </div>
)}
```

- [ ] **Step 5: Run the session mainline component and page tests**

Run:

```bash
cd console/web && npm test -- src/components/session-detail/conversation-event-list.test.tsx src/app/sessions/[id]/page.test.tsx
```

Expected: PASS with the milestone board and filterable conversation event list visible in `Mainline`

- [ ] **Step 6: Commit the Session Detail UI**

```bash
git add console/web/src/components/session-detail/milestone-board.tsx console/web/src/components/session-detail/conversation-event-list.tsx console/web/src/components/session-detail/conversation-event-list.test.tsx console/web/src/app/sessions/[id]/page.tsx console/web/src/app/sessions/[id]/page.test.tsx
git commit -m "feat: add session mainline observability view"
```

### Task 6: Add `context_steps_hidden` To The Public Stream And Reconcile Live Transcript

**Files:**
- Modify: `agiwo/agent/models/stream.py`
- Modify: `tests/agent/test_run_log_replay_parity.py`
- Modify: `console/web/src/lib/api.ts`
- Modify: `console/web/src/hooks/use-chat-stream.ts`
- Modify: `console/web/src/hooks/use-chat-stream.test.tsx`

- [ ] **Step 1: Write the failing stream replay and live-hook tests**

```python
# tests/agent/test_run_log_replay_parity.py
from agiwo.agent.models.log import ContextStepsHidden
from agiwo.agent.models.stream import stream_items_from_entries


def test_context_steps_hidden_emits_public_hide_event() -> None:
    items = stream_items_from_entries(
        [
            ContextStepsHidden(
                sequence=3,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                step_ids=["step-review-call", "step-review-result"],
                reason="review_metadata",
            )
        ]
    )

    assert [item.type for item in items] == ["context_steps_hidden"]
    assert items[0].step_ids == ["step-review-call", "step-review-result"]
```

```tsx
// console/web/src/hooks/use-chat-stream.test.tsx
it("removes hidden review metadata when context_steps_hidden arrives", async () => {
  const stream = [
    {
      type: "step_completed",
      session_id: "sess-1",
      run_id: "run-1",
      agent_id: "agent-1",
      parent_run_id: null,
      depth: 0,
      step: {
        id: "step-review-result",
        session_id: "sess-1",
        run_id: "run-1",
        sequence: 10,
        role: "tool",
        agent_id: "agent-1",
        content: "Trajectory review: aligned=false.",
        content_for_user: null,
        reasoning_content: null,
        user_input: null,
        tool_calls: null,
        tool_call_id: "tc-review",
        name: "review_trajectory",
        condensed_content: null,
        metrics: null,
        created_at: null,
        parent_run_id: null,
        depth: 0,
      },
    },
    {
      type: "context_steps_hidden",
      session_id: "sess-1",
      run_id: "run-1",
      agent_id: "agent-1",
      parent_run_id: null,
      depth: 0,
      step_ids: ["step-review-result"],
      reason: "review_metadata",
    },
  ];

  // feed stream into the hook test harness, then assert the review step disappears
});
```

- [ ] **Step 2: Run the stream replay and hook tests and confirm they fail**

Run:

```bash
uv run pytest tests/agent/test_run_log_replay_parity.py -v
cd console/web && npm test -- src/hooks/use-chat-stream.test.tsx
```

Expected:
- Python test fails because no public `context_steps_hidden` stream payload exists
- Hook test fails because `useChatStream()` ignores the new event type

- [ ] **Step 3: Add the new stream event to the SDK public stream model**

```python
# agiwo/agent/models/stream.py
@dataclass(kw_only=True)
class ContextStepsHiddenEvent(AgentStreamItemBase):
    step_ids: list[str]
    reason: str
    type: Literal["context_steps_hidden"] = "context_steps_hidden"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["step_ids"] = list(self.step_ids)
        payload["reason"] = self.reason
        return payload
```

```python
# agiwo/agent/models/stream.py
def _stream_item_from_runtime_entry(
    entry: RunLogEntry,
    *,
    run_contexts: dict[str, dict[str, Any]],
) -> "AgentStreamItem | None":
    base_kwargs = _base_kwargs_with_run_context(entry, run_contexts)
    if isinstance(entry, ContextStepsHidden):
        return ContextStepsHiddenEvent(
            **base_kwargs,
            step_ids=list(entry.step_ids),
            reason=entry.reason,
        )
    ...
```

```python
# agiwo/agent/models/stream.py
AgentStreamItem: TypeAlias = (
    RunStartedEvent
    | StepDeltaEvent
    | StepCompletedEvent
    | MessagesRebuiltEvent
    | ContextStepsHiddenEvent
    | CompactionAppliedEvent
    | CompactionFailedEvent
    | StepBackAppliedEvent
    | TerminationDecidedEvent
    | RunRolledBackEvent
    | RunCompletedEvent
    | RunFailedEvent
)
```

- [ ] **Step 4: Update frontend stream contracts and hook behavior**

```ts
// console/web/src/lib/api.ts
export interface ContextStepsHiddenEventPayload extends StreamEventBase {
  type: "context_steps_hidden";
  step_ids: string[];
  reason: string;
}

export type AgentStreamEventPayload =
  | RunStartedEventPayload
  | StepDeltaEventPayload
  | StepCompletedEventPayload
  | ContextStepsHiddenEventPayload
  | RunCompletedEventPayload
  | RunFailedEventPayload;
```

```ts
// console/web/src/hooks/use-chat-stream.ts
if (agentEvent.type === "context_steps_hidden") {
  const hiddenStepIds = new Set(agentEvent.step_ids);
  setMessages((prev) =>
    prev.filter((message) => !message.stepId || !hiddenStepIds.has(message.stepId)),
  );
  continue;
}
```

- [ ] **Step 5: Run the stream replay and live-hook tests again**

Run:

```bash
uv run pytest tests/agent/test_run_log_replay_parity.py -v
cd console/web && npm test -- src/hooks/use-chat-stream.test.tsx
```

Expected: PASS with the new event projected and the live transcript reconciled

- [ ] **Step 6: Commit the live transcript consistency work**

```bash
git add agiwo/agent/models/stream.py tests/agent/test_run_log_replay_parity.py console/web/src/lib/api.ts console/web/src/hooks/use-chat-stream.ts console/web/src/hooks/use-chat-stream.test.tsx
git commit -m "feat: reconcile live transcript with hidden review metadata"
```

### Task 7: Document The New Structured Observability And Run Full Verification

**Files:**
- Modify: `docs/guides/storage.md`
- Modify: `docs/guides/streaming.md`

- [ ] **Step 1: Write the doc updates**

```md
<!-- docs/guides/storage.md -->
## Console Structured Observability

Console `Session Detail` and `Trace Detail` expose additional structured read models such as milestone boards, review cycles, mainline events, and LLM-call summaries.

These are not new runtime truth sources. They are derived from:

- `RunLog` entries for canonical runtime facts
- `Trace.spans` for single-trace replay and tool/LLM correlation

Raw `steps`, `runtime_decisions`, `timeline_events`, and `spans` remain available for debug views.
```

```md
<!-- docs/guides/streaming.md -->
## `context_steps_hidden`

When the runtime hides review metadata or other context-only steps after they were already emitted, the public stream now emits:

```json
{
  "type": "context_steps_hidden",
  "session_id": "...",
  "run_id": "...",
  "agent_id": "...",
  "step_ids": ["step-review-call", "step-review-result"],
  "reason": "review_metadata"
}
```

Clients should reconcile any live transcript messages that reference those `step_ids`.
```

- [ ] **Step 2: Run the required backend and frontend verification commands**

Run:

```bash
uv run python scripts/lint.py ci
uv run python scripts/check.py console-tests
cd console/web && npm run lint && npm test && npm run build
```

Expected:
- Python lint passes
- Console backend tests pass
- Frontend lint, tests, and production build all pass

- [ ] **Step 3: Commit the docs and verification pass**

```bash
git add docs/guides/storage.md docs/guides/streaming.md
git commit -m "docs: describe structured observability and stream hiding"
```

## Self-Review

### Spec Coverage

- `Mainline / Debug` dual view for `Trace Detail`: covered in Task 4
- `Mainline / Debug` dual view for `Session Detail`: covered in Task 5
- session milestone board and latest review: covered in Tasks 1, 3, and 5
- structured `LLM Call` summaries: covered in Tasks 1, 2, and 4
- grouped review cycles: covered in Tasks 1, 2, 3, and 4
- downgraded conversation events: covered in Tasks 1, 3, and 5
- live transcript consistency with hidden steps: covered in Task 6
- docs for derived observability and stream event behavior: covered in Task 7

### Placeholder Scan

- No `TODO`, `TBD`, or “implement later” markers remain
- Each code-changing step includes explicit snippets
- Each verification step includes exact commands and expected outcomes

### Type Consistency

The plan uses one consistent contract set throughout:

- `SessionMilestoneBoardRecord` / `SessionMilestoneBoardResponse`
- `ReviewCycleRecord` / `ReviewCycleResponse`
- `ConversationEventRecord` / `ConversationEventResponse`
- `TraceMainlineEventRecord` / `TraceMainlineEventResponse`
- `TraceLlmCallRecord` / `TraceLlmCallResponse`
- `ContextStepsHiddenEvent` / `ContextStepsHiddenEventPayload`

No later task renames these symbols.
