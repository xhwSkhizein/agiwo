# Console Trace Runtime Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Console Web observability good enough to replay one trace end-to-end, including runtime decisions and goal-directed review details, while keeping `RunLog` and `Trace.spans` as the only runtime truth sources.

**Architecture:** First fix the `step_back` condensation contract so prompt-visible tool results, stored condensed content, and UI rendering all agree. Then extend the trace collector so runtime spans cover all important runtime facts. On top of those facts, add one server-side observability read-model builder that turns run-log entries and trace spans into stable `runtime_decisions` and `timeline_events` payloads. Finally, update Session Detail and Trace Detail to render those read models as readable review surfaces instead of raw span dumps.

**Tech Stack:** Python 3.10+, dataclasses, asyncio, Pydantic, SQLite/in-memory run-log replay, React 19, TypeScript, Vitest, pytest

**Spec:** `docs/superpowers/specs/2026-04-25-console-trace-runtime-observability-design.md`

---

## File Map

- `agiwo/agent/review/step_back_executor.py`: Builds the prompt-visible step-back content and persists `StepCondensedContentUpdated`.
- `agiwo/agent/trace_writer.py`: Converts committed `RunLog` facts into trace/runtime spans.
- `console/server/services/runtime/run_query_service.py`: Console-facing run-log query facade; should stop exposing hidden review metadata by default and list recent runtime decisions.
- `console/server/services/runtime/session_view_service.py`: Builds `SessionDetailRecord` and session observability payloads.
- `console/server/services/runtime/runtime_observability.py`: New read-model builder for runtime decisions, trace timeline events, and review parsing.
- `console/server/models/view.py`: API response models for trace/session observability.
- `console/server/response_serialization.py`: Serializes expanded trace/session observability payloads.
- `console/web/src/lib/api.ts`: TypeScript contracts for trace/session observability.
- `console/web/src/components/session-detail/session-observability-panel.tsx`: Session runtime-decision cards with expand/collapse details.
- `console/web/src/components/trace-detail/trace-runtime-decisions.tsx`: New trace-level runtime decision list.
- `console/web/src/components/trace-detail/trace-loop-timeline.tsx`: New trace-level loop replay timeline.
- `console/web/src/app/traces/[id]/page.tsx`: Trace detail layout; should render summary, runtime decisions, timeline, then waterfall.
- `tests/agent/test_step_back_executor.py`: Guards the corrected step-back persistence semantics.
- `tests/observability/test_collector.py`: Guards runtime span coverage in the trace collector.
- `console/tests/test_run_query_service.py`: Guards recent runtime-decision listing and prompt-visible step replay semantics.
- `console/tests/test_sessions_api.py`: Guards expanded session observability payloads.
- `console/tests/test_traces_api.py`: Guards expanded trace response payloads.
- `console/web/src/components/session-detail/session-observability-panel.test.tsx`: Guards the new session decision-card rendering.
- `console/web/src/app/traces/[id]/page.test.tsx`: New Trace Detail UI regression coverage.
- `console/web/src/lib/chat-types.test.ts`: Guards condensed/original tool-result rendering semantics.
- `docs/guides/storage.md`: Documents richer trace/runtime observability shape.

### Task 1: Fix Step-Back Display Semantics And Hide Review Metadata From Console Step Replay

**Files:**
- Modify: `agiwo/agent/review/step_back_executor.py`
- Modify: `console/server/services/runtime/run_query_service.py`
- Test: `tests/agent/test_step_back_executor.py`
- Test: `console/tests/test_runtime_replay_consistency.py`
- Test: `console/web/src/lib/chat-types.test.ts`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agent/test_step_back_executor.py
import pytest
from unittest.mock import AsyncMock

from agiwo.agent.review.step_back_executor import execute_step_back


@pytest.mark.asyncio
async def test_persists_experience_summary_as_condensed_content() -> None:
    storage = AsyncMock()
    storage.append_step_condensed_content = AsyncMock(return_value=True)

    await execute_step_back(
        messages=[
            {
                "role": "tool",
                "tool_call_id": "tc_2",
                "content": "Found 15 JWT references",
                "_sequence": 5,
            }
        ],
        checkpoint_seq=2,
        experience="JWT search was off-track. Token validation lives in auth.py.",
        review_tool_call_id="tc_review",
        step_lookup={"tc_2": {"id": "step_2", "sequence": 5}},
        storage=storage,
        session_id="s1",
        run_id="r1",
        agent_id="a1",
    )

    storage.append_step_condensed_content.assert_awaited_once_with(
        "s1",
        "r1",
        "a1",
        "step_2",
        "[EXPERIENCE] JWT search was off-track. Token validation lives in auth.py.",
    )
```

```python
# console/tests/test_runtime_replay_consistency.py
import pytest

from agiwo.agent import MessageRole
from agiwo.agent.models.log import ContextStepsHidden, ToolStepCommitted
from agiwo.agent.storage.base import InMemoryRunLogStorage
from server.services.runtime.run_query_service import RunQueryService


@pytest.mark.asyncio
async def test_console_step_queries_hide_review_metadata_by_default() -> None:
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            ToolStepCommitted(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                step_id="step-review-result",
                role=MessageRole.TOOL,
                content="Trajectory review: aligned=True.",
                tool_call_id="tc_review",
                name="review_trajectory",
            ),
            ContextStepsHidden(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                step_ids=["step-review-result"],
                reason="review_metadata",
            ),
        ]
    )
    service = RunQueryService(run_storage=storage)

    page = await service.list_session_steps("sess-1", limit=20, order="asc")

    assert page.items == []
    assert page.total == 0
```

```ts
// console/web/src/lib/chat-types.test.ts
import { describe, expect, test } from "vitest";

import { messageFromStep } from "./chat-types";

describe("messageFromStep", () => {
  test("prefers condensed tool content and keeps the original result available", () => {
    const message = messageFromStep({
      id: "step-1",
      session_id: "sess-1",
      run_id: "run-1",
      sequence: 5,
      role: "tool",
      agent_id: "agent-1",
      content: "Found 15 JWT references",
      content_for_user: null,
      reasoning_content: null,
      user_input: null,
      tool_calls: null,
      tool_call_id: "tc_2",
      name: "search",
      condensed_content: "[EXPERIENCE] JWT search was off-track.",
      metrics: null,
      created_at: null,
      parent_run_id: null,
      depth: 0,
    });

    expect(message?.text).toBe("[EXPERIENCE] JWT search was off-track.");
    expect(message?.originalContent).toBe("Found 15 JWT references");
  });
});
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

```bash
uv run pytest tests/agent/test_step_back_executor.py console/tests/test_runtime_replay_consistency.py -v
cd console/web && npm test -- src/lib/chat-types.test.ts
```

Expected:
- Python tests fail because `execute_step_back()` still stores the original content as `condensed_content`
- Console replay test fails because `RunQueryService.list_session_steps()` still includes hidden review metadata
- Frontend test fails until the corrected condensed/original contract is reflected end-to-end

- [ ] **Step 3: Fix the step-back persistence contract**

```python
# agiwo/agent/review/step_back_executor.py
async def execute_step_back(
    *,
    messages: list[dict[str, Any]],
    checkpoint_seq: int,
    experience: str,
    review_tool_call_id: str | None = None,
    step_lookup: dict[str, dict[str, Any]],
    storage: RunLogStorage,
    session_id: str,
    run_id: str,
    agent_id: str,
) -> StepBackOutcome:
    """Condense tool results after *checkpoint_seq* into targeted updates."""

    content_updates: list[ContentUpdate] = []

    for message in messages:
        if message.get("role") != "tool":
            continue
        sequence = message.get("_sequence", 0)
        if sequence <= checkpoint_seq:
            continue
        tool_call_id = message.get("tool_call_id", "")
        if not tool_call_id or tool_call_id == review_tool_call_id:
            continue
        original_content = message.get("content", "")
        if not original_content:
            continue

        step_info = step_lookup.get(tool_call_id)
        step_id = step_info.get("id", "") if step_info is not None else ""
        if not step_id:
            step = await storage.get_step_by_tool_call_id(session_id, tool_call_id)
            step_id = step.id if step is not None else ""
        if not step_id:
            continue

        condensed_content = f"[EXPERIENCE] {experience}"
        await storage.append_step_condensed_content(
            session_id,
            run_id,
            agent_id,
            step_id,
            condensed_content,
        )
        content_updates.append(
            ContentUpdate(
                step_id=step_id,
                tool_call_id=tool_call_id,
                content=condensed_content,
            )
        )

    return StepBackOutcome(
        applied=True,
        review_tool_call_id=review_tool_call_id,
        content_updates=content_updates,
        step_back_applied=True,
        affected_count=len(content_updates),
        checkpoint_seq=checkpoint_seq,
        experience=experience,
    )
```

- [ ] **Step 4: Make Console step replay prompt-visible by default**

```python
# console/server/services/runtime/run_query_service.py
async def list_session_steps(
    self,
    session_id: str,
    *,
    start_seq: int | None = None,
    end_seq: int | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    limit: int,
    order: str,
) -> PageSlice[StepView]:
    raw_steps = await self.run_storage.list_step_views(
        session_id=session_id,
        start_seq=start_seq,
        end_seq=end_seq,
        run_id=run_id,
        agent_id=agent_id,
        include_hidden_from_context=False,
        limit=limit + 1,
        order=order,
    )
    has_more = len(raw_steps) > limit
    total = None
    if (
        start_seq is None
        and end_seq is None
        and run_id is None
        and agent_id is None
    ):
        total = len(
            await self.run_storage.list_step_views(
                session_id=session_id,
                include_hidden_from_context=False,
                limit=100_000,
                order="asc",
            )
        )
    return PageSlice(
        items=raw_steps[:limit],
        limit=limit,
        offset=0,
        has_more=has_more,
        total=total,
    )
```

- [ ] **Step 5: Re-run the focused tests and confirm they pass**

```bash
uv run pytest tests/agent/test_step_back_executor.py console/tests/test_runtime_replay_consistency.py -v
cd console/web && npm test -- src/lib/chat-types.test.ts
```

Expected: PASS. The persisted condensed content is `[EXPERIENCE] ...`, hidden review steps stay out of default session step pages, and frontend chat rendering shows condensed content by default with original content available as secondary detail.

- [ ] **Step 6: Commit**

```bash
git add agiwo/agent/review/step_back_executor.py console/server/services/runtime/run_query_service.py tests/agent/test_step_back_executor.py console/tests/test_runtime_replay_consistency.py console/web/src/lib/chat-types.test.ts
git commit -m "fix: align step-back observability semantics"
```

### Task 2: Extend Trace Collector Runtime Coverage

**Files:**
- Modify: `agiwo/agent/trace_writer.py`
- Test: `tests/observability/test_collector.py`

- [ ] **Step 1: Write the failing trace-collector tests**

```python
# tests/observability/test_collector.py
from agiwo.agent.models.log import CompactionFailed, HookFailed, RunRolledBack
from agiwo.observability.trace import SpanKind, SpanStatus


@pytest.mark.asyncio
async def test_collector_records_runtime_failure_and_rollback_spans() -> None:
    storage = _RecordingTraceStorage()
    collector = AgentTraceCollector(store=storage)
    collector.start(trace_id="trace-3", agent_id="agent-1", session_id="session-1")

    await collector.on_run_log_entries(
        [
            RunStarted(
                sequence=1,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                user_input="hello",
            ),
            CompactionFailed(
                sequence=2,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                error="model timeout",
                attempt=1,
                max_attempts=3,
                terminal=False,
            ),
            RunRolledBack(
                sequence=3,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                start_sequence=5,
                end_sequence=8,
                reason="scheduler_no_progress_periodic",
            ),
            HookFailed(
                sequence=4,
                session_id="session-1",
                run_id="run-1",
                agent_id="agent-1",
                phase="before_review",
                handler_name="observe_step_back",
                critical=False,
                error="hook boom",
            ),
        ]
    )

    runtime_spans = [span for span in collector._trace.spans if span.kind == SpanKind.RUNTIME]
    assert [span.name for span in runtime_spans] == [
        "compaction_failed",
        "rollback",
        "hook_failed",
    ]
    assert runtime_spans[0].status is SpanStatus.ERROR
    assert runtime_spans[0].attributes["sequence"] == 2
    assert runtime_spans[1].attributes["start_sequence"] == 5
    assert runtime_spans[2].attributes["phase"] == "before_review"
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

```bash
uv run pytest tests/observability/test_collector.py -v
```

Expected: FAIL because `trace_writer.py` does not currently create runtime spans for `CompactionFailed`, `RunRolledBack`, or `HookFailed`, and existing runtime spans do not carry `sequence`.

- [ ] **Step 3: Extend runtime span generation in the trace collector**

```python
# agiwo/agent/trace_writer.py
from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CompactionApplied,
    CompactionFailed,
    HookFailed,
    LLMCallCompleted,
    LLMCallStarted,
    RunFailed as RunFailedEntry,
    RunFinished,
    RunLogEntry,
    RunRolledBack,
    RunStarted,
    StepBackApplied,
    TerminationDecided,
    ToolStepCommitted,
)


def _build_runtime_span_from_entry(
    trace_id: str,
    entry: CompactionApplied
    | CompactionFailed
    | StepBackApplied
    | TerminationDecided
    | RunRolledBack
    | HookFailed,
    run_span: Span | None,
) -> Span:
    parent_id = run_span.span_id if run_span else None
    parent_depth = run_span.depth if run_span else 0
    attributes: dict[str, Any] = {"sequence": entry.sequence}
    name = "runtime"
    status = SpanStatus.OK
    error_message: str | None = None

    if isinstance(entry, CompactionApplied):
        name = "compaction"
        attributes.update(
            {
                "start_sequence": entry.start_sequence,
                "end_sequence": entry.end_sequence,
                "before_token_estimate": entry.before_token_estimate,
                "after_token_estimate": entry.after_token_estimate,
                "message_count": entry.message_count,
                "transcript_path": entry.transcript_path,
                "summary": entry.summary,
            }
        )
    elif isinstance(entry, CompactionFailed):
        name = "compaction_failed"
        status = SpanStatus.ERROR
        error_message = entry.error
        attributes.update(
            {
                "error": entry.error,
                "attempt": entry.attempt,
                "max_attempts": entry.max_attempts,
                "terminal": entry.terminal,
            }
        )
    elif isinstance(entry, StepBackApplied):
        name = "step_back"
        attributes.update(
            {
                "affected_count": entry.affected_count,
                "checkpoint_seq": entry.checkpoint_seq,
                "experience": entry.experience,
            }
        )
    elif isinstance(entry, RunRolledBack):
        name = "rollback"
        attributes.update(
            {
                "start_sequence": entry.start_sequence,
                "end_sequence": entry.end_sequence,
                "reason": entry.reason,
            }
        )
    elif isinstance(entry, HookFailed):
        name = "hook_failed"
        status = SpanStatus.ERROR
        error_message = entry.error
        attributes.update(
            {
                "phase": entry.phase,
                "handler_name": entry.handler_name,
                "critical": entry.critical,
                "error": entry.error,
            }
        )
    else:
        name = "termination"
        attributes.update(
            {
                "termination_reason": entry.termination_reason.value,
                "phase": entry.phase,
                "source": entry.source,
            }
        )

    return Span(
        trace_id=trace_id,
        parent_span_id=parent_id,
        kind=SpanKind.RUNTIME,
        name=name,
        depth=parent_depth + 1,
        attributes=attributes,
        run_id=entry.run_id,
        start_time=entry.created_at,
        end_time=entry.created_at,
        duration_ms=0.0,
        status=status,
        error_message=error_message,
    )
```

```python
# agiwo/agent/trace_writer.py
def _apply_runtime_entry_to_trace(
    trace: Trace,
    entry: RunLogEntry,
    *,
    run_spans: dict[str, Span],
) -> bool:
    if not isinstance(
        entry,
        (
            CompactionApplied,
            CompactionFailed,
            StepBackApplied,
            TerminationDecided,
            RunRolledBack,
            HookFailed,
        ),
    ):
        return False
    _append_runtime_entry_to_trace(
        trace,
        entry,
        run_spans=run_spans,
    )
    return True
```

- [ ] **Step 4: Re-run the collector tests and confirm they pass**

```bash
uv run pytest tests/observability/test_collector.py -v
```

Expected: PASS. Trace spans now cover runtime failure, rollback, and hook failure facts, and every runtime span exposes `sequence` in `attributes`.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/trace_writer.py tests/observability/test_collector.py
git commit -m "feat: extend trace runtime observability coverage"
```

### Task 3: Add Server-Side Observability Read Models For Session And Trace Views

**Files:**
- Create: `console/server/services/runtime/runtime_observability.py`
- Modify: `console/server/services/runtime/run_query_service.py`
- Modify: `console/server/services/runtime/session_view_service.py`
- Modify: `console/server/models/view.py`
- Modify: `console/server/response_serialization.py`
- Test: `console/tests/test_run_query_service.py`
- Test: `console/tests/test_sessions_api.py`
- Test: `console/tests/test_traces_api.py`

- [ ] **Step 1: Write the failing backend/API tests**

```python
# console/tests/test_run_query_service.py
@pytest.mark.asyncio
async def test_run_query_service_lists_recent_runtime_decision_entries() -> None:
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            CompactionApplied(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                start_sequence=1,
                end_sequence=2,
                before_token_estimate=500,
                after_token_estimate=120,
                message_count=2,
                transcript_path="/tmp/compact.json",
                summary="compact",
            ),
            CompactionFailed(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                error="model timeout",
                attempt=1,
                max_attempts=3,
                terminal=False,
            ),
            StepBackApplied(
                sequence=3,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                affected_count=1,
                checkpoint_seq=2,
                experience="switch plan",
            ),
        ]
    )
    service = RunQueryService(run_storage=storage)

    decisions = await service.list_runtime_decision_events("sess-1", limit=10)

    assert [decision.kind for decision in decisions] == [
        "step_back",
        "compaction_failed",
        "compaction",
    ]
    assert decisions[1].details["attempt"] == 1
```

```python
# console/tests/test_sessions_api.py
@pytest.mark.asyncio
async def test_session_detail_returns_recent_runtime_decision_list(client) -> None:
    await _seed_session_context(client)
    runtime = _runtime(client)
    await runtime.run_log_storage.append_entries(
        [
            CompactionApplied(
                sequence=1,
                session_id="session-a",
                run_id="run-a1",
                agent_id="agent-alpha",
                start_sequence=1,
                end_sequence=2,
                before_token_estimate=300,
                after_token_estimate=90,
                message_count=2,
                transcript_path="/tmp/a.json",
                summary="compact a",
            ),
            StepBackApplied(
                sequence=2,
                session_id="session-a",
                run_id="run-a1",
                agent_id="agent-alpha",
                affected_count=1,
                checkpoint_seq=2,
                experience="switch plan",
            ),
            CompactionFailed(
                sequence=3,
                session_id="session-a",
                run_id="run-a2",
                agent_id="agent-alpha",
                error="model timeout",
                attempt=2,
                max_attempts=3,
                terminal=False,
            ),
        ]
    )

    response = await client.get("/api/sessions/session-a")
    payload = response.json()

    assert [event["kind"] for event in payload["observability"]["decision_events"]] == [
        "compaction_failed",
        "step_back",
        "compaction",
    ]
    assert payload["observability"]["decision_events"][0]["details"]["error"] == "model timeout"
```

```python
# console/tests/test_traces_api.py
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace


@pytest.mark.asyncio
async def test_trace_detail_exposes_runtime_decisions_and_timeline_events(client) -> None:
    runtime = get_console_runtime_from_app(client._transport.app)  # type: ignore[attr-defined]
    trace = Trace(
        trace_id="trace-review",
        agent_id="agent-alpha",
        session_id="session-1",
        status=SpanStatus.OK,
        spans=[
            Span(
                trace_id="trace-review",
                span_id="agent-span",
                kind=SpanKind.AGENT,
                name="agent-alpha",
                run_id="run-1",
                status=SpanStatus.OK,
                depth=0,
            ),
            Span(
                trace_id="trace-review",
                parent_span_id="agent-span",
                kind=SpanKind.TOOL_CALL,
                name="search_code",
                run_id="run-1",
                status=SpanStatus.OK,
                depth=1,
                tool_details={
                    "tool_name": "search_code",
                    "tool_call_id": "tc-1",
                    "input_args": {"q": "jwt"},
                    "output": "search result\n\n<system-review>\nTrigger: step_interval\nSteps since last review: 8\n</system-review>",
                    "status": "completed",
                },
            ),
            Span(
                trace_id="trace-review",
                parent_span_id="agent-span",
                kind=SpanKind.RUNTIME,
                name="step_back",
                run_id="run-1",
                status=SpanStatus.OK,
                depth=1,
                attributes={
                    "sequence": 9,
                    "affected_count": 2,
                    "checkpoint_seq": 4,
                    "experience": "switch plan",
                },
            ),
        ],
    )
    await runtime.trace_storage.save_trace(trace)

    response = await client.get("/api/traces/trace-review")
    payload = response.json()

    assert payload["runtime_decisions"][0]["kind"] == "step_back"
    assert payload["timeline_events"][0]["kind"] == "review_checkpoint"
    assert payload["timeline_events"][1]["kind"] == "runtime_decision"
```

- [ ] **Step 2: Run the focused backend/API tests and confirm they fail**

```bash
uv run pytest console/tests/test_run_query_service.py console/tests/test_sessions_api.py console/tests/test_traces_api.py -v
```

Expected: FAIL because there is no `list_runtime_decision_events(...)`, `TraceResponse` has no `runtime_decisions` or `timeline_events`, and no server-side builder parses review/runtime observability from run-log entries or trace spans.

- [ ] **Step 3: Add one shared observability read-model builder**

```python
# console/server/services/runtime/runtime_observability.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from agiwo.agent.models.log import (
    CompactionApplied,
    CompactionFailed,
    HookFailed,
    RunLogEntry,
    RunRolledBack,
    StepBackApplied,
    TerminationDecided,
)
from agiwo.observability.trace import Span, Trace
from server.models.session import RuntimeDecisionRecord

_SYSTEM_REVIEW_BLOCK_RE = re.compile(
    r"<system-review>\s*(?P<body>.*?)\s*</system-review>",
    re.DOTALL,
)
_TRIGGER_RE = re.compile(r"Trigger:\s*(?P<value>[^\n]+)")
_STEPS_RE = re.compile(r"Steps since last review:\s*(?P<value>\d+)")
_MILESTONE_RE = re.compile(r'Active milestone:\s*"(?P<value>[^"]+)"')
_ALIGNED_RE = re.compile(r"aligned\s*=\s*(?P<value>true|false)", re.IGNORECASE)


def build_runtime_decision_record_from_entry(entry: RunLogEntry) -> RuntimeDecisionRecord:
    if isinstance(entry, CompactionApplied):
        return RuntimeDecisionRecord(
            kind="compaction",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            summary=(
                f"seq {entry.start_sequence}-{entry.end_sequence}, "
                f"{entry.before_token_estimate} -> {entry.after_token_estimate} tokens"
            ),
            details={
                "start_sequence": entry.start_sequence,
                "end_sequence": entry.end_sequence,
                "before_token_estimate": entry.before_token_estimate,
                "after_token_estimate": entry.after_token_estimate,
                "message_count": entry.message_count,
                "summary": entry.summary,
                "transcript_path": entry.transcript_path,
            },
        )
    if isinstance(entry, CompactionFailed):
        return RuntimeDecisionRecord(
            kind="compaction_failed",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            summary=f"attempt {entry.attempt}/{entry.max_attempts}: {entry.error}",
            details={
                "error": entry.error,
                "attempt": entry.attempt,
                "max_attempts": entry.max_attempts,
                "terminal": entry.terminal,
            },
        )
    if isinstance(entry, StepBackApplied):
        return RuntimeDecisionRecord(
            kind="step_back",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            summary=(
                f"{entry.affected_count} results condensed after checkpoint seq "
                f"{entry.checkpoint_seq}"
            ),
            details={
                "affected_count": entry.affected_count,
                "checkpoint_seq": entry.checkpoint_seq,
                "experience": entry.experience,
            },
        )
    if isinstance(entry, RunRolledBack):
        return RuntimeDecisionRecord(
            kind="rollback",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            summary=f"seq {entry.start_sequence}-{entry.end_sequence} hidden",
            details={
                "start_sequence": entry.start_sequence,
                "end_sequence": entry.end_sequence,
                "reason": entry.reason,
            },
        )
    if isinstance(entry, TerminationDecided):
        return RuntimeDecisionRecord(
            kind="termination",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            summary=f"{entry.termination_reason.value} via {entry.source}",
            details={
                "reason": entry.termination_reason.value,
                "phase": entry.phase,
                "source": entry.source,
            },
        )
    if isinstance(entry, HookFailed):
        return RuntimeDecisionRecord(
            kind="hook_failed",
            sequence=entry.sequence,
            run_id=entry.run_id,
            agent_id=entry.agent_id,
            created_at=entry.created_at,
            summary=f"{entry.phase}: {entry.handler_name} failed",
            details={
                "phase": entry.phase,
                "handler_name": entry.handler_name,
                "critical": entry.critical,
                "error": entry.error,
            },
        )
    raise TypeError(f"Unsupported runtime decision entry: {type(entry).__name__}")
```

```python
# console/server/services/runtime/runtime_observability.py
def parse_system_review_notice(text: str) -> dict[str, Any] | None:
    match = _SYSTEM_REVIEW_BLOCK_RE.search(text)
    if match is None:
        return None
    body = match.group("body")
    trigger_match = _TRIGGER_RE.search(body)
    step_match = _STEPS_RE.search(body)
    milestone_match = _MILESTONE_RE.search(body)
    return {
        "raw_notice": body,
        "trigger_reason": trigger_match.group("value").strip()
        if trigger_match is not None
        else "unknown",
        "steps_since_last_review": int(step_match.group("value"))
        if step_match is not None
        else None,
        "active_milestone": milestone_match.group("value").strip()
        if milestone_match is not None
        else None,
    }


def build_trace_timeline_events(trace: Trace) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for span in trace.spans:
        timestamp = span.start_time.isoformat() if span.start_time is not None else None
        sequence = int(span.attributes.get("sequence")) if "sequence" in span.attributes else None

        if span.kind == "tool_call" and span.tool_details:
            tool_name = str(span.tool_details.get("tool_name") or span.name)
            raw_output = str(span.tool_details.get("output") or "")
            if tool_name == "review_trajectory":
                aligned_match = _ALIGNED_RE.search(raw_output)
                aligned = (
                    aligned_match.group("value").lower()
                    if aligned_match is not None
                    else "unknown"
                )
                events.append(
                    {
                        "kind": "review_result",
                        "timestamp": timestamp,
                        "sequence": sequence,
                        "run_id": span.run_id,
                        "agent_id": trace.agent_id,
                        "span_id": span.span_id,
                        "step_id": span.step_id,
                        "title": "Review Result",
                        "summary": (
                            "trajectory aligned"
                            if aligned == "true"
                            else "trajectory misaligned"
                            if aligned == "false"
                            else "trajectory reviewed"
                        ),
                        "status": span.status,
                        "details": {
                            "tool_name": tool_name,
                            "tool_call_id": span.tool_details.get("tool_call_id"),
                            "aligned": aligned,
                            "raw_output": raw_output,
                        },
                    }
                )
                continue
            if tool_name == "declare_milestones":
                milestones = span.tool_details.get("input_args", {}).get("milestones", [])
                events.append(
                    {
                        "kind": "milestone_update",
                        "timestamp": timestamp,
                        "sequence": sequence,
                        "run_id": span.run_id,
                        "agent_id": trace.agent_id,
                        "span_id": span.span_id,
                        "step_id": span.step_id,
                        "title": "Milestone Update",
                        "summary": f"{len(milestones)} milestones declared/updated",
                        "status": span.status,
                        "details": {"milestones": milestones},
                    }
                )
                continue
            review_notice = parse_system_review_notice(raw_output)
            if review_notice is not None:
                events.append(
                    {
                        "kind": "review_checkpoint",
                        "timestamp": timestamp,
                        "sequence": sequence,
                        "run_id": span.run_id,
                        "agent_id": trace.agent_id,
                        "span_id": span.span_id,
                        "step_id": span.step_id,
                        "title": "Review Checkpoint",
                        "summary": (
                            f"triggered by {review_notice['trigger_reason']} after "
                            f"{review_notice['steps_since_last_review']} steps"
                        ),
                        "status": "ok",
                        "details": review_notice,
                    }
                )
            events.append(
                {
                    "kind": "tool_call",
                    "timestamp": timestamp,
                    "sequence": sequence,
                    "run_id": span.run_id,
                    "agent_id": trace.agent_id,
                    "span_id": span.span_id,
                    "step_id": span.step_id,
                    "title": f"Tool Call: {tool_name}",
                    "summary": span.tool_details.get("status", "completed"),
                    "status": span.status,
                    "details": dict(span.tool_details),
                }
            )
            continue

        if span.kind == "runtime":
            events.append(
                {
                    "kind": "hook_failed" if span.name == "hook_failed" else "runtime_decision",
                    "timestamp": timestamp,
                    "sequence": sequence,
                    "run_id": span.run_id,
                    "agent_id": trace.agent_id,
                    "span_id": span.span_id,
                    "step_id": span.step_id,
                    "title": span.name.replace("_", " ").title(),
                    "summary": dict(span.attributes),
                    "status": span.status,
                    "details": dict(span.attributes),
                }
            )

    return sorted(
        events,
        key=lambda event: (
            event["timestamp"] or "",
            event["sequence"] or 0,
            event["kind"],
        ),
    )
```

- [ ] **Step 4: Expose recent session decisions and expanded trace payloads**

```python
# console/server/services/runtime/run_query_service.py
from agiwo.agent.models.log import (
    CompactionApplied,
    CompactionFailed,
    HookFailed,
    RunRolledBack,
    StepBackApplied,
    TerminationDecided,
)
from server.models.session import RuntimeDecisionRecord
from server.services.runtime.runtime_observability import (
    build_runtime_decision_record_from_entry,
)


async def list_runtime_decision_events(
    self,
    session_id: str,
    *,
    run_id: str | None = None,
    agent_id: str | None = None,
    limit: int = 20,
) -> list[RuntimeDecisionRecord]:
    entries = await self.run_storage.list_entries(
        session_id=session_id,
        run_id=run_id,
        agent_id=agent_id,
        limit=100_000,
    )
    decision_entries = [
        entry
        for entry in entries
        if isinstance(
            entry,
            (
                CompactionApplied,
                CompactionFailed,
                StepBackApplied,
                RunRolledBack,
                TerminationDecided,
            ),
        )
    ]
    decision_entries.sort(key=lambda entry: entry.sequence, reverse=True)
    return [
        build_runtime_decision_record_from_entry(entry)
        for entry in decision_entries[:limit]
    ]
```

```python
# console/server/services/runtime/session_view_service.py
async def _build_observability(
    self,
    *,
    session_id: str,
    runtime_decisions: RuntimeDecisionState,
) -> SessionObservabilityRecord:
    del runtime_decisions
    return SessionObservabilityRecord(
        recent_traces=await self._trace_queries.list_session_recent_traces(session_id),
        decision_events=await self._run_queries.list_runtime_decision_events(
            session_id,
            limit=12,
        ),
    )
```

```python
# console/server/models/view.py
class TraceTimelineEventResponse(BaseModel):
    kind: str
    timestamp: str | None = None
    sequence: int | None = None
    run_id: str | None = None
    agent_id: str | None = None
    span_id: str | None = None
    step_id: str | None = None
    title: str
    summary: str
    status: str = "ok"
    details: dict[str, Any] = Field(default_factory=dict)


class TraceResponse(TraceBase):
    end_time: str | None = None
    root_span_id: str | None = None
    max_depth: int = 0
    spans: list[SpanResponse] = Field(default_factory=list)
    runtime_decisions: list[RuntimeDecisionResponse] = Field(default_factory=list)
    timeline_events: list[TraceTimelineEventResponse] = Field(default_factory=list)
```

```python
# console/server/response_serialization.py
from server.services.runtime.runtime_observability import (
    build_trace_runtime_decisions,
    build_trace_timeline_events,
)


def trace_response_from_sdk(trace: Trace) -> TraceResponse:
    spans = [...]
    runtime_decisions = [
        runtime_decision_response_from_record(decision)
        for decision in build_trace_runtime_decisions(trace)
    ]
    timeline_events = [
        TraceTimelineEventResponse(**event)
        for event in build_trace_timeline_events(trace)
    ]
    return TraceResponse(
        **_trace_base_kwargs(trace),
        end_time=trace.end_time.isoformat()
        if getattr(trace, "end_time", None)
        else None,
        root_span_id=getattr(trace, "root_span_id", None),
        max_depth=getattr(trace, "max_depth", 0),
        spans=spans,
        runtime_decisions=runtime_decisions,
        timeline_events=timeline_events,
    )
```

- [ ] **Step 5: Re-run the backend/API tests and confirm they pass**

```bash
uv run pytest console/tests/test_run_query_service.py console/tests/test_sessions_api.py console/tests/test_traces_api.py -v
```

Expected: PASS. Session API exposes a recent decision list, and Trace API now returns both `runtime_decisions` and `timeline_events`.

- [ ] **Step 6: Commit**

```bash
git add console/server/services/runtime/runtime_observability.py console/server/services/runtime/run_query_service.py console/server/services/runtime/session_view_service.py console/server/models/view.py console/server/response_serialization.py console/tests/test_run_query_service.py console/tests/test_sessions_api.py console/tests/test_traces_api.py
git commit -m "feat: add console runtime observability read models"
```

### Task 4: Render The New Observability Surfaces In Console Web

**Files:**
- Modify: `console/web/src/lib/api.ts`
- Modify: `console/web/src/components/session-detail/session-observability-panel.tsx`
- Create: `console/web/src/components/trace-detail/trace-runtime-decisions.tsx`
- Create: `console/web/src/components/trace-detail/trace-loop-timeline.tsx`
- Modify: `console/web/src/app/traces/[id]/page.tsx`
- Test: `console/web/src/components/session-detail/session-observability-panel.test.tsx`
- Create: `console/web/src/app/traces/[id]/page.test.tsx`

- [ ] **Step 1: Write the failing frontend tests**

```ts
// console/web/src/components/session-detail/session-observability-panel.test.tsx
import { screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";

import { renderWithProviders } from "@/test/render";
import { SessionObservabilityPanel } from "./session-observability-panel";

describe("SessionObservabilityPanel", () => {
  test("renders detailed runtime decision cards", () => {
    renderWithProviders(
      <SessionObservabilityPanel
        sessionId="sess-1"
        observability={{
          recent_traces: [],
          decision_events: [
            {
              kind: "step_back",
              sequence: 8,
              run_id: "run-1",
              agent_id: "agent-1",
              created_at: "2026-04-25T12:00:00Z",
              summary: "2 results condensed after checkpoint seq 4",
              details: {
                affected_count: 2,
                checkpoint_seq: 4,
                experience: "switch plan",
              },
            },
          ],
        }}
      />
    );

    expect(screen.getByText("2 results condensed after checkpoint seq 4")).toBeInTheDocument();
    expect(screen.getByText("checkpoint_seq")).toBeInTheDocument();
    expect(screen.getByText("switch plan")).toBeInTheDocument();
  });
});
```

```ts
// console/web/src/app/traces/[id]/page.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getTrace: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "trace-1" }),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return {
    ...actual,
    getTrace: apiMocks.getTrace,
  };
});

import TraceDetailPage from "./page";

describe("TraceDetailPage", () => {
  test("renders runtime decisions and loop timeline above the waterfall", async () => {
    apiMocks.getTrace.mockResolvedValue({
      trace_id: "trace-1",
      agent_id: "agent-1",
      session_id: "sess-1",
      user_id: null,
      start_time: "2026-04-25T12:00:00Z",
      end_time: "2026-04-25T12:00:02Z",
      duration_ms: 2000,
      status: "ok",
      root_span_id: "root-1",
      max_depth: 1,
      total_tokens: 10,
      total_input_tokens: 4,
      total_output_tokens: 6,
      total_cache_read_tokens: 0,
      total_cache_creation_tokens: 0,
      total_token_cost: 0.01,
      total_llm_calls: 1,
      total_tool_calls: 2,
      input_query: "fix the bug",
      final_output: "done",
      spans: [],
      runtime_decisions: [
        {
          kind: "step_back",
          sequence: 9,
          run_id: "run-1",
          agent_id: "agent-1",
          created_at: "2026-04-25T12:00:01Z",
          summary: "2 results condensed after checkpoint seq 4",
          details: {
            affected_count: 2,
            checkpoint_seq: 4,
            experience: "switch plan",
          },
        },
      ],
      timeline_events: [
        {
          kind: "review_checkpoint",
          timestamp: "2026-04-25T12:00:00Z",
          sequence: 8,
          run_id: "run-1",
          agent_id: "agent-1",
          span_id: "span-1",
          step_id: "step-1",
          title: "Review Checkpoint",
          summary: "triggered by step_interval after 8 steps",
          status: "ok",
          details: {
            trigger_reason: "step_interval",
            steps_since_last_review: 8,
          },
        },
      ],
    });

    render(<TraceDetailPage />);

    await waitFor(() => {
      expect(screen.getByText("Runtime Decisions")).toBeInTheDocument();
    });

    expect(screen.getByText("Loop Timeline")).toBeInTheDocument();
    expect(screen.getByText("Review Checkpoint")).toBeInTheDocument();
    expect(screen.getByText("Span Waterfall (0 spans)")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the frontend tests and confirm they fail**

```bash
cd console/web && npm test -- src/components/session-detail/session-observability-panel.test.tsx src/app/traces/[id]/page.test.tsx
```

Expected: FAIL because the frontend types do not include `runtime_decisions` or `timeline_events`, the session panel does not render structured details, and Trace Detail still only knows about the span waterfall.

- [ ] **Step 3: Add typed payloads and reusable trace-detail components**

```ts
// console/web/src/lib/api.ts
export interface TraceTimelineEvent {
  kind: string;
  timestamp: string | null;
  sequence: number | null;
  run_id: string | null;
  agent_id: string | null;
  span_id: string | null;
  step_id: string | null;
  title: string;
  summary: string;
  status: string;
  details: Record<string, unknown>;
}

export interface TraceDetail {
  trace_id: string;
  agent_id: string | null;
  session_id: string | null;
  user_id: string | null;
  start_time: string | null;
  end_time: string | null;
  duration_ms: number | null;
  status: string;
  root_span_id: string | null;
  total_tokens: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cache_read_tokens: number;
  total_cache_creation_tokens: number;
  total_token_cost: number;
  total_llm_calls: number;
  total_tool_calls: number;
  max_depth: number;
  input_query: string | null;
  final_output: string | null;
  spans: SpanResponse[];
  runtime_decisions: RuntimeDecisionEvent[];
  timeline_events: TraceTimelineEvent[];
}
```

```tsx
// console/web/src/components/trace-detail/trace-runtime-decisions.tsx
"use client";

import { JsonDisclosure } from "@/components/json-disclosure";
import { SectionCard } from "@/components/section-card";
import type { RuntimeDecisionEvent } from "@/lib/api";

export function TraceRuntimeDecisions({
  decisions,
}: {
  decisions: RuntimeDecisionEvent[];
}) {
  return (
    <SectionCard
      title="Runtime Decisions"
      bodyClassName="space-y-3 px-4 py-4"
    >
      {decisions.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No runtime decisions recorded for this trace.
        </div>
      ) : (
        decisions.map((decision) => (
          <details
            key={`${decision.kind}-${decision.sequence}-${decision.run_id}`}
            className="rounded-xl border border-line bg-panel px-3 py-3"
          >
            <summary className="cursor-pointer list-none">
              <div className="space-y-1">
                <div className="text-sm font-medium text-foreground">{decision.summary}</div>
                <div className="text-xs text-ink-muted">
                  {decision.kind} · seq {decision.sequence} · {decision.run_id} · {decision.agent_id}
                </div>
              </div>
            </summary>
            <div className="mt-3">
              <JsonDisclosure label="Details" value={decision.details} />
            </div>
          </details>
        ))
      )}
    </SectionCard>
  );
}
```

```tsx
// console/web/src/components/trace-detail/trace-loop-timeline.tsx
"use client";

import { JsonDisclosure } from "@/components/json-disclosure";
import { SectionCard } from "@/components/section-card";
import type { TraceTimelineEvent } from "@/lib/api";

export function TraceLoopTimeline({
  events,
}: {
  events: TraceTimelineEvent[];
}) {
  return (
    <SectionCard title="Loop Timeline" bodyClassName="space-y-3 px-4 py-4">
      {events.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-6 text-sm text-ink-muted">
          No loop events replayed for this trace.
        </div>
      ) : (
        events.map((event) => (
          <details
            key={`${event.kind}-${event.sequence ?? "na"}-${event.span_id ?? "span"}`}
            className="rounded-xl border border-line bg-panel px-3 py-3"
          >
            <summary className="cursor-pointer list-none">
              <div className="space-y-1">
                <div className="text-sm font-medium text-foreground">{event.title}</div>
                <div className="text-sm text-foreground">{event.summary}</div>
                <div className="text-xs text-ink-muted">
                  {event.run_id} · {event.agent_id} · seq {event.sequence ?? "-"}
                </div>
              </div>
            </summary>
            <div className="mt-3">
              <JsonDisclosure label="Details" value={event.details} />
            </div>
          </details>
        ))
      )}
    </SectionCard>
  );
}
```

- [ ] **Step 4: Update Session Detail and Trace Detail to use the new read models**

```tsx
// console/web/src/components/session-detail/session-observability-panel.tsx
function DecisionTitle({ event }: { event: RuntimeDecisionEvent }) {
  return (
    <details className="rounded-xl border border-line bg-panel px-3 py-3">
      <summary className="list-none cursor-pointer">
        <div className="flex items-start gap-3">
          <DecisionIcon kind={event.kind} />
          <div className="min-w-0 flex-1 space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium capitalize text-foreground">
                {event.kind}
              </span>
              <span className="rounded-full border border-line px-2 py-0.5 text-[11px] uppercase tracking-wide text-ink-muted">
                seq {event.sequence}
              </span>
            </div>
            <p className="text-sm text-foreground">{event.summary}</p>
          </div>
        </div>
      </summary>
      <div className="mt-3">
        <JsonDisclosure label="Details" value={event.details} />
      </div>
    </details>
  );
}
```

```tsx
// console/web/src/app/traces/[id]/page.tsx
import { TraceLoopTimeline } from "@/components/trace-detail/trace-loop-timeline";
import { TraceRuntimeDecisions } from "@/components/trace-detail/trace-runtime-decisions";

export default function TraceDetailPage() {
  ...
  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <BackHeader href="/traces" title="Trace Detail" subtitle={trace.trace_id} />
      ...
      <TokenSummaryCards
        ...
        extraCards={
          <>
            <MetricCard label="Status" value={<TraceStatusBadge status={trace.status} />} />
            <MetricCard label="Duration" value={formatDurationMs(trace.duration_ms || 0)} />
            <MetricCard label="LLM / Tool" value={`${trace.total_llm_calls} / ${trace.total_tool_calls}`} />
            <MetricCard label="Runtime Decisions" value={String(trace.runtime_decisions.length)} />
            <MetricCard
              label="Review Events"
              value={String(
                trace.timeline_events.filter((event) =>
                  ["review_checkpoint", "review_result", "milestone_update"].includes(event.kind),
                ).length,
              )}
            />
          </>
        }
      />

      <TraceRuntimeDecisions decisions={trace.runtime_decisions} />
      <TraceLoopTimeline events={trace.timeline_events} />

      <SectionCard
        className="overflow-hidden"
        title={`Span Waterfall (${trace.spans.length} spans)`}
        headerClassName="border-b border-line px-4 py-3"
      >
        ...
      </SectionCard>
    </div>
  );
}
```

- [ ] **Step 5: Re-run the frontend tests and confirm they pass**

```bash
cd console/web && npm test -- src/components/session-detail/session-observability-panel.test.tsx src/app/traces/[id]/page.test.tsx
```

Expected: PASS. Session Detail renders expandable runtime-decision cards, and Trace Detail renders both `Runtime Decisions` and `Loop Timeline` before the waterfall.

- [ ] **Step 6: Commit**

```bash
git add console/web/src/lib/api.ts console/web/src/components/session-detail/session-observability-panel.tsx console/web/src/components/trace-detail/trace-runtime-decisions.tsx console/web/src/components/trace-detail/trace-loop-timeline.tsx console/web/src/app/traces/[id]/page.tsx console/web/src/components/session-detail/session-observability-panel.test.tsx console/web/src/app/traces/[id]/page.test.tsx
git commit -m "feat: add console trace runtime observability views"
```

### Task 5: Update Docs And Run The Full Validation Sweep

**Files:**
- Modify: `docs/guides/storage.md`
- Test/Run: `uv run python scripts/lint.py ci`
- Test/Run: `uv run pytest tests/agent/test_step_back_executor.py tests/observability/test_collector.py console/tests/test_run_query_service.py console/tests/test_sessions_api.py console/tests/test_traces_api.py -v`
- Test/Run: `uv run python scripts/check.py console-tests`
- Test/Run: `cd console/web && npm run lint && npm test && npm run build`

- [ ] **Step 1: Update the storage/observability guide**

```md
<!-- docs/guides/storage.md -->
### Trace Structure

```
Trace (one per trace id)
├── Span: Agent Execution
│   ├── Span: LLM Call
│   ├── Span: Tool Call
│   ├── Span: Runtime Decision (compaction / step_back / rollback / termination)
│   └── Span: Runtime Failure (compaction_failed / hook_failed)
└── Span: Child Agent (if spawned)
```

Console Web derives two higher-level observability views from these spans:

- **Runtime Decisions**: structured cards for compaction, compaction_failed, step_back, rollback, and termination
- **Loop Timeline**: a replay-oriented timeline that combines tool calls, review checkpoints, review results, milestone updates, and runtime decisions
```

- [ ] **Step 2: Run the targeted Python regression suite**

```bash
uv run pytest tests/agent/test_step_back_executor.py tests/observability/test_collector.py console/tests/test_run_query_service.py console/tests/test_sessions_api.py console/tests/test_traces_api.py -v
```

Expected: PASS.

- [ ] **Step 3: Run the required repo and console checks**

```bash
uv run python scripts/lint.py ci
uv run python scripts/check.py console-tests
(cd console/web && npm run lint)
(cd console/web && npm test)
(cd console/web && npm run build)
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add docs/guides/storage.md
git commit -m "docs: describe console trace runtime observability"
```

## Self-Review Checklist

- Confirm every spec requirement maps to one task:
  - Trace runtime decision cards: Task 3 + Task 4
  - Loop timeline with review detail: Task 3 + Task 4
  - Session recent decision list: Task 3 + Task 4
  - `step_back` semantic correction: Task 1
  - runtime span coverage for `compaction_failed` / `rollback` / `hook_failed`: Task 2
- Confirm no placeholder language remains:
  - no `TBD`
  - no “handle appropriately”
  - no “similar to”
- Confirm naming consistency:
  - `runtime_decisions`
  - `timeline_events`
  - `review_checkpoint`
  - `review_result`
  - `milestone_update`
  - `compaction_failed`
  - `hook_failed`
