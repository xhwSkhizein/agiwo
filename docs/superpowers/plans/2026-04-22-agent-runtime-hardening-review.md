# Agent Runtime Hardening Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining gaps between the runtime refactor spec and the shipped Phase 1-3 implementation, then prove end-to-end replay, query, and runtime-experience consistency across agent, scheduler, and console.

**Architecture:** This plan treats the main refactor as shipped and focuses on hardening. The first pass is a spec-to-implementation audit and terminology sync, the second pass fills missing runtime-fact read surfaces, the third pass adds live-vs-replay parity tests and query guardrails, and the last pass trims the remaining runtime hot spots that still make future iteration harder than necessary.

**Tech Stack:** Python 3.10+, `RunLogStorage`, `RunView`, `StepView`, `AgentStreamItem`, scheduler runtime facts, console runtime/query services, pytest, ruff

---

## Scope Check

This plan is a **post-refactor hardening plan**, not a new architecture project.

It covers:

1. checking every remaining runtime-refactor spec claim against current code
2. reconciling outdated spec wording where the implementation intentionally settled on a different but simpler surface
3. adding the missing runtime-fact read/query surfaces that are still absent
4. adding behavior-level parity tests for live execution vs replayed views
5. tightening query correctness and cleaning up the largest remaining runtime hotspot

It does **not**:

1. reopen the fundamental `RunLog` architecture
2. reintroduce `Run`, `StepRecord`, or legacy `AgentHooks`
3. redesign scheduler public APIs
4. introduce a new generic event bus or untyped timeline blob model

## Recommended Direction

Before execution starts, lock these decisions:

1. **Do not add a generic `TimelineView` just to satisfy old wording in the spec.** The shipped public read surface should stay `RunView`, `StepView`, replayed `AgentStreamItem`, and explicit runtime-decision views.
2. **Add a dedicated runtime-decision read model** for `termination`, `compaction`, `step-back`, and rollback, rather than making callers replay raw `RunLog` families manually.
3. **Treat live/replay parity as a first-class acceptance criterion.** The refactor is not “done” until live stream/trace/query results can be cross-checked against replayed `RunLog`.
4. **Use targeted extractions, not broad rewrites, for oversized modules.** The main hotspot to reduce is `agiwo/scheduler/runner.py`.

## File Structure

### Create

- `docs/superpowers/reviews/2026-04-22-agent-runtime-audit.md`
- `agiwo/agent/models/runtime_decision.py`
- `tests/agent/test_runtime_decision_views.py`
- `tests/agent/test_run_log_replay_parity.py`
- `tests/scheduler/test_scheduler_runtime_parity.py`
- `console/tests/test_runtime_replay_consistency.py`
- `agiwo/scheduler/runner_output.py`

### Modify

- `docs/superpowers/specs/2026-04-21-agent-runtime-refactor-design.md`
- `docs/architecture/scheduler-console-runtime-refactor.md`
- `agiwo/agent/__init__.py`
- `agiwo/agent/models/__init__.py`
- `agiwo/agent/storage/base.py`
- `agiwo/agent/storage/sqlite.py`
- `agiwo/agent/storage/serialization.py`
- `agiwo/agent/trace_writer.py`
- `agiwo/scheduler/runtime_facts.py`
- `agiwo/scheduler/runner.py`
- `console/server/services/runtime/run_query_service.py`
- `console/server/services/runtime/session_view_service.py`
- `console/server/response_serialization.py`
- `tests/agent/test_storage_serialization.py`
- `tests/scheduler/test_context_rollback.py`
- `tests/scheduler/test_scheduler.py`
- `console/tests/test_session_summary.py`
- `console/tests/test_sessions_api.py`

### Responsibilities

- `docs/superpowers/reviews/2026-04-22-agent-runtime-audit.md`
  - Record a concrete spec-to-code audit matrix and classify each item as shipped, spec-update-needed, or code-gap.
- `agiwo/agent/models/runtime_decision.py`
  - Define the stable read models for latest `termination`, `compaction`, `step-back`, and rollback state.
- `agiwo/agent/storage/base.py` / `agiwo/agent/storage/sqlite.py`
  - Expose a stable query facade for runtime-decision state without forcing callers to replay raw entries.
- `agiwo/scheduler/runtime_facts.py`
  - Extend scheduler runtime facts beyond latest run summary into runtime-decision views.
- `console/server/services/runtime/run_query_service.py`
  - Extend console query surfaces to expose the same runtime-decision state when needed for detail/timeline views.
- `agiwo/scheduler/runner_output.py`
  - Extract output-finalization and periodic/no-progress handling out of the oversized scheduler runner.

## Task 1: Write The Audit Matrix And Sync The Spec Vocabulary

**Files:**
- Create: `docs/superpowers/reviews/2026-04-22-agent-runtime-audit.md`
- Modify: `docs/superpowers/specs/2026-04-21-agent-runtime-refactor-design.md`
- Modify: `docs/architecture/scheduler-console-runtime-refactor.md`

- [ ] **Step 1: Write the audit matrix document**

```md
# Agent Runtime Audit Matrix

## Status Legend

- `SHIPPED`
- `SPEC_UPDATE`
- `CODE_GAP`

## Core Runtime

| Spec Item | Current Code | Status | Notes |
| --- | --- | --- | --- |
| Single persisted source of truth is `RunLog` | `agiwo/agent/models/log.py`, `agiwo/agent/storage/base.py` | SHIPPED | Canonical |
| `Run`, `StepRecord`, `AgentHooks` removed | `agiwo/agent/`, `tests/` | SHIPPED | No compatibility layer |
| Stable public read surface includes `TimelineView` | `agiwo/agent/storage/base.py` | SPEC_UPDATE | Use `RunView` / `StepView` / replayed stream items instead |
| Latest runtime-decision state is queryable | `agiwo/agent/storage/base.py` | CODE_GAP | Add dedicated read model |

## Scheduler

| Spec Item | Current Code | Status | Notes |
| --- | --- | --- | --- |
| Scheduler reads replayed runtime facts | `agiwo/scheduler/runtime_facts.py` | SHIPPED | Partial surface only |
| Scheduler can query latest runtime decisions | `agiwo/scheduler/runtime_facts.py` | CODE_GAP | Extend facade |

## Console

| Spec Item | Current Code | Status | Notes |
| --- | --- | --- | --- |
| Queries use RunLog-backed facade | `console/server/services/runtime/run_query_service.py` | SHIPPED | Runs / steps / session detail |
| SSE and replay semantics are parity-tested | `console/tests/` | CODE_GAP | Add dedicated parity tests |
```

- [ ] **Step 2: Run a terminology scan and confirm the current drift**

Run: `rg -n "TimelineView|RunEngine|HookDispatcher|RunLogWriter" docs/superpowers/specs/2026-04-21-agent-runtime-refactor-design.md docs/architecture/scheduler-console-runtime-refactor.md agiwo/agent agiwo/scheduler console/server -g '*.md' -g '*.py'`
Expected: matches in the spec still use design-era names that need either explicit implementation mapping or wording updates.

- [ ] **Step 3: Update the runtime spec to match the shipped read surface**

```md
Implementation mapping:

- `RunEngine` in this document maps to the shipped `RunLoopOrchestrator`.
- `HookDispatcher` maps to `agiwo.agent.hooks.HookRegistry`.
- `RunLogWriter` maps to `SessionRuntime.append_run_log_entries(...)` plus typed entry builders.
- The stable public read surface is `RunView`, `StepView`, replayed `AgentStreamItem`, and explicit runtime-decision views.
- This design no longer requires a separate generic `TimelineView`.
```

- [ ] **Step 4: Update the architecture note to reflect the audit outcome**

```md
Hardening status:

- spec wording for legacy design names has been normalized to shipped symbols
- `TimelineView` wording was replaced by the actual shipped view surface
- remaining open items are limited to runtime-decision read models and parity tests
```

- [ ] **Step 5: Re-read the audit matrix against the spec**

Run: `sed -n '1,220p' docs/superpowers/reviews/2026-04-22-agent-runtime-audit.md`
Expected: every unresolved item is classified explicitly as `SPEC_UPDATE` or `CODE_GAP`; no vague “later” language remains.

- [ ] **Step 6: Commit the audit baseline**

```bash
git add docs/superpowers/reviews/2026-04-22-agent-runtime-audit.md docs/superpowers/specs/2026-04-21-agent-runtime-refactor-design.md docs/architecture/scheduler-console-runtime-refactor.md
git commit -m "docs: add runtime refactor audit matrix"
```

## Task 2: Add Stable Runtime-Decision Read Models And Queries

**Files:**
- Create: `agiwo/agent/models/runtime_decision.py`
- Modify: `agiwo/agent/models/__init__.py`
- Modify: `agiwo/agent/__init__.py`
- Modify: `agiwo/agent/storage/base.py`
- Modify: `agiwo/agent/storage/sqlite.py`
- Test: `tests/agent/test_runtime_decision_views.py`

- [ ] **Step 1: Write the failing decision-view tests**

```python
import pytest

from agiwo.agent import (
    CompactionApplied,
    StepBackApplied,
    TerminationDecided,
    TerminationReason,
)
from agiwo.agent.models.runtime_decision import RuntimeDecisionState
from agiwo.agent.storage.base import InMemoryRunLogStorage


@pytest.mark.asyncio
async def test_storage_builds_latest_runtime_decision_state() -> None:
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            TerminationDecided(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                termination_reason=TerminationReason.MAX_STEPS,
                phase="before_termination",
                source="limit",
            ),
            StepBackApplied(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                affected_sequences=[7, 8],
                affected_step_ids=["step-7"],
                feedback="summarized",
                replacement="summary",
                trigger="token_threshold",
            ),
        ]
    )

    state = await storage.get_runtime_decision_state(session_id="sess-1")

    assert isinstance(state, RuntimeDecisionState)
    assert state.latest_termination is not None
    assert state.latest_termination.reason is TerminationReason.MAX_STEPS
    assert state.latest_step_back is not None
    assert state.latest_step_back.trigger == "token_threshold"
```

- [ ] **Step 2: Run the targeted tests to confirm the gap**

Run: `uv run pytest tests/agent/test_runtime_decision_views.py -v`
Expected: FAIL because `RuntimeDecisionState` and `get_runtime_decision_state(...)` do not exist yet.

- [ ] **Step 3: Define the runtime-decision view models**

```python
from dataclasses import dataclass

from agiwo.agent.models.run import CompactMetadata, TerminationReason


@dataclass(frozen=True, slots=True)
class TerminationDecisionView:
    reason: TerminationReason
    phase: str
    source: str
    run_id: str
    sequence: int


@dataclass(frozen=True, slots=True)
class StepBackDecisionView:
    run_id: str
    sequence: int
    affected_sequences: tuple[int, ...]
    affected_step_ids: tuple[str, ...]
    feedback: str | None
    replacement: str | None
    trigger: str | None


@dataclass(frozen=True, slots=True)
class RollbackDecisionView:
    run_id: str
    sequence: int
    start_sequence: int
    end_sequence: int
    reason: str


@dataclass(frozen=True, slots=True)
class RuntimeDecisionState:
    latest_compaction: CompactMetadata | None = None
    latest_step_back: StepBackDecisionView | None = None
    latest_termination: TerminationDecisionView | None = None
    latest_rollback: RollbackDecisionView | None = None
```

- [ ] **Step 4: Add stable storage query methods**

```python
class RunLogStorage(ABC):
    async def get_runtime_decision_state(
        self,
        *,
        session_id: str,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> RuntimeDecisionState:
        entries = await self.list_entries(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            limit=100_000,
        )
        ...
```

```python
async def get_runtime_decision_state(... ) -> RuntimeDecisionState:
    entries = await self.list_entries(...)
    latest_compaction = ...
    latest_step_back = ...
    latest_termination = ...
    latest_rollback = ...
    return RuntimeDecisionState(...)
```

- [ ] **Step 5: Add the SQLite implementation without exposing raw storage layout**

```python
async def get_runtime_decision_state(... ) -> RuntimeDecisionState:
    entries = await self.list_entries(
        session_id=session_id,
        run_id=run_id,
        agent_id=agent_id,
        limit=100_000,
    )
    return _build_runtime_decision_state(entries)
```

- [ ] **Step 6: Re-run the targeted decision-view tests**

Run: `uv run pytest tests/agent/test_runtime_decision_views.py -v`
Expected: PASS

- [ ] **Step 7: Commit the runtime-decision query surface**

```bash
git add agiwo/agent/models/runtime_decision.py agiwo/agent/models/__init__.py agiwo/agent/__init__.py agiwo/agent/storage/base.py agiwo/agent/storage/sqlite.py tests/agent/test_runtime_decision_views.py
git commit -m "feat: add runtime decision read models"
```

## Task 3: Extend Scheduler And Console Facts To Consume Runtime Decisions

**Files:**
- Modify: `agiwo/scheduler/runtime_facts.py`
- Modify: `console/server/services/runtime/run_query_service.py`
- Modify: `console/server/services/runtime/session_view_service.py`
- Modify: `console/server/response_serialization.py`
- Test: `tests/scheduler/test_scheduler.py`
- Test: `console/tests/test_session_summary.py`

- [ ] **Step 1: Write the failing scheduler/runtime query tests**

```python
@pytest.mark.asyncio
async def test_scheduler_runtime_facts_exposes_runtime_decision_state():
    facts = SchedulerRuntimeFacts(...)

    state = await facts.get_runtime_decision_state(agent_state)

    assert state.latest_rollback is not None
    assert state.latest_rollback.reason == "scheduler_no_progress_periodic"
```

```python
@pytest.mark.asyncio
async def test_session_view_service_can_read_runtime_decision_state():
    detail = await service.get_session_detail("sess-1")

    assert detail.summary.session_id == "sess-1"
    assert detail.scheduler_state is not None
```

- [ ] **Step 2: Run the targeted tests to confirm the missing surface**

Run: `uv run pytest tests/scheduler/test_scheduler.py console/tests/test_session_summary.py -v`
Expected: FAIL because scheduler/runtime facts and console query services do not expose `RuntimeDecisionState`.

- [ ] **Step 3: Extend `SchedulerRuntimeFacts`**

```python
class SchedulerRuntimeFacts:
    async def get_runtime_decision_state(
        self,
        state: AgentState,
    ) -> RuntimeDecisionState | None:
        agent = self._rt.agents.get(state.id)
        if agent is None:
            return None
        return await agent.run_log_storage.get_runtime_decision_state(
            session_id=state.resolve_runtime_session_id(),
            agent_id=state.id,
        )
```

- [ ] **Step 4: Extend the console run query facade**

```python
@dataclass(slots=True)
class SessionRunSnapshot:
    run_views: list[RunView]
    committed_step_count: int
    runtime_decisions: RuntimeDecisionState | None = None
```

```python
async def get_session_run_snapshot(self, session_id: str) -> SessionRunSnapshot:
    stats = await self.run_storage.get_session_run_stats(session_id)
    decisions = await self.run_storage.get_runtime_decision_state(session_id=session_id)
    return SessionRunSnapshot(
        run_views=stats.run_views,
        committed_step_count=stats.committed_step_count,
        runtime_decisions=decisions,
    )
```

- [ ] **Step 5: Keep the API stable unless a real consumer needs new fields**

```python
# Do not change public REST models yet.
# Keep runtime_decisions inside service-level snapshots only until tests prove
# a specific API payload needs it.
```

- [ ] **Step 6: Re-run the targeted scheduler/console tests**

Run: `uv run pytest tests/scheduler/test_scheduler.py console/tests/test_session_summary.py -v`
Expected: PASS

- [ ] **Step 7: Commit the runtime-facts extension**

```bash
git add agiwo/scheduler/runtime_facts.py console/server/services/runtime/run_query_service.py console/server/services/runtime/session_view_service.py console/server/response_serialization.py tests/scheduler/test_scheduler.py console/tests/test_session_summary.py
git commit -m "feat: expose runtime decisions through query facades"
```

## Task 4: Add Live-Vs-Replay Parity Tests For Agent, Scheduler, And Console

**Files:**
- Create: `tests/agent/test_run_log_replay_parity.py`
- Create: `tests/scheduler/test_scheduler_runtime_parity.py`
- Create: `console/tests/test_runtime_replay_consistency.py`
- Modify: `agiwo/agent/trace_writer.py`
- Modify: `tests/agent/test_storage_serialization.py`
- Modify: `tests/scheduler/test_context_rollback.py`
- Modify: `console/tests/test_sessions_api.py`

- [ ] **Step 1: Write the failing agent replay parity test**

```python
@pytest.mark.asyncio
async def test_stream_items_replayed_from_run_log_match_live_terminal_sequence():
    handle = agent.start("hello", session_id="sess-1")
    live_types = []
    async for item in handle.stream():
        live_types.append(item.type)
    await handle.wait()

    entries = await agent.run_log_storage.list_entries(session_id="sess-1", limit=100_000)
    replayed_types = [item.type for item in stream_items_from_entries(entries)]

    assert replayed_types[-1] == live_types[-1]
    assert "run_started" in replayed_types
    assert "run_completed" in replayed_types
```

- [ ] **Step 2: Write the failing rollback / scheduler parity test**

```python
@pytest.mark.asyncio
async def test_scheduler_periodic_no_progress_replay_matches_visible_steps():
    output = await scheduler.wait_for(state_id, timeout=2.0)
    assert output.termination_reason is not None

    facts = await scheduler._runtime_facts.get_runtime_decision_state(state)
    visible_steps = await scheduler._runtime_facts.list_step_views(state)

    assert facts is not None
    assert facts.latest_rollback is not None
    assert all(
        step.sequence < facts.latest_rollback.start_sequence
        or step.sequence > facts.latest_rollback.end_sequence
        for step in visible_steps
    )
```

- [ ] **Step 3: Write the failing console replay consistency test**

```python
@pytest.mark.asyncio
async def test_session_runs_api_matches_run_query_service_ordering(client):
    response = await client.get("/api/runs?session_id=session-a")
    payload = response.json()

    runtime = _runtime(client)
    page = await get_run_query_service(runtime).list_runs(
        session_id="session-a",
        limit=20,
        offset=0,
    )

    assert [item["id"] for item in payload["items"]] == [run.run_id for run in page.items]
```

- [ ] **Step 4: Run the parity suite to confirm gaps**

Run: `uv run pytest tests/agent/test_run_log_replay_parity.py tests/scheduler/test_scheduler_runtime_parity.py console/tests/test_runtime_replay_consistency.py -v`
Expected: FAIL because the parity suite does not exist yet and at least one live-vs-replay mismatch still needs trace/stream adjustment.

- [ ] **Step 5: Fix any trace/stream rebuild mismatches revealed by the tests**

```python
# Example shape inside trace writer:
trace = await collector.build_from_entries(entries)
assert trace.root_span_id is not None
```

```python
# Example shape inside stream replay tests:
replayed = stream_items_from_entries(entries)
assert [item.type for item in replayed if item.type != "step_delta"] == expected_types
```

- [ ] **Step 6: Re-run the parity suite**

Run: `uv run pytest tests/agent/test_run_log_replay_parity.py tests/scheduler/test_scheduler_runtime_parity.py console/tests/test_runtime_replay_consistency.py -v`
Expected: PASS

- [ ] **Step 7: Commit the parity coverage**

```bash
git add tests/agent/test_run_log_replay_parity.py tests/scheduler/test_scheduler_runtime_parity.py console/tests/test_runtime_replay_consistency.py agiwo/agent/trace_writer.py tests/agent/test_storage_serialization.py tests/scheduler/test_context_rollback.py console/tests/test_sessions_api.py
git commit -m "test: add runtime replay parity coverage"
```

## Task 5: Tighten Query Guardrails And Reduce The Remaining Runtime Hotspot

**Files:**
- Create: `agiwo/scheduler/runner_output.py`
- Modify: `agiwo/scheduler/runner.py`
- Modify: `agiwo/agent/storage/sqlite.py`
- Test: `tests/scheduler/test_scheduler.py`
- Test: `tests/agent/test_storage_serialization.py`

- [ ] **Step 1: Write the failing extraction/regression tests**

```python
@pytest.mark.asyncio
async def test_wait_for_still_prefers_runtime_run_view_after_runner_extraction():
    result = await scheduler.wait_for("root", timeout=2.0)
    assert result.run_id == "run-1"
```

```python
@pytest.mark.asyncio
async def test_sqlite_step_counts_ignore_hidden_steps_after_rollback():
    count = await storage.get_committed_step_count("sess-1")
    assert count == 0
```

- [ ] **Step 2: Run the targeted tests before moving code**

Run: `uv run pytest tests/scheduler/test_scheduler.py tests/agent/test_storage_serialization.py -v`
Expected: PASS before extraction; this is the safety baseline.

- [ ] **Step 3: Extract output-finalization logic from `runner.py`**

```python
# agiwo/scheduler/runner_output.py
class RunnerOutputHandlers:
    def __init__(self, runner: "SchedulerRunner") -> None:
        self._runner = runner

    async def handle_periodic_output(...): ...
    async def handle_failed_output(...): ...
    async def complete_state(...): ...
```

```python
# agiwo/scheduler/runner.py
from agiwo.scheduler.runner_output import RunnerOutputHandlers

class SchedulerRunner:
    def __init__(...):
        self._output = RunnerOutputHandlers(self)
```

- [ ] **Step 4: Tighten SQLite correctness where replay metadata affects counts**

```python
async def get_committed_step_count(self, session_id: str) -> int:
    entries = await self.list_entries(session_id=session_id, limit=100_000)
    return len(build_step_views_from_entries(entries))
```

```python
async def batch_get_committed_step_counts(... ) -> dict[str, int]:
    return {
        session_id: await self.get_committed_step_count(session_id)
        for session_id in session_ids
    }
```

- [ ] **Step 5: Re-run the targeted hotspot/query tests**

Run: `uv run pytest tests/scheduler/test_scheduler.py tests/agent/test_storage_serialization.py -v`
Expected: PASS

- [ ] **Step 6: Re-run the repo lint gate**

Run: `uv run python scripts/lint.py ci`
Expected: PASS, with `agiwo/scheduler/runner.py` reduced below the previous warning threshold or at least materially smaller than before.

- [ ] **Step 7: Commit the hotspot cleanup**

```bash
git add agiwo/scheduler/runner_output.py agiwo/scheduler/runner.py agiwo/agent/storage/sqlite.py tests/scheduler/test_scheduler.py tests/agent/test_storage_serialization.py
git commit -m "refactor: harden runtime query paths and split runner output handlers"
```

## Task 6: Run The Final End-To-End Validation Matrix

**Files:**
- Modify: `docs/superpowers/reviews/2026-04-22-agent-runtime-audit.md`

- [ ] **Step 1: Run the focused SDK runtime suite**

Run: `uv run pytest tests/agent/ tests/scheduler/ -v`
Expected: PASS

- [ ] **Step 2: Run the console backend suite**

Run: `uv run python scripts/check.py console-tests`
Expected: PASS

- [ ] **Step 3: Run the unified lint/import/contracts gate**

Run: `uv run python scripts/lint.py ci`
Expected: PASS

- [ ] **Step 4: Mark every audit matrix row as closed or intentionally narrowed**

```md
Final status:

- `SHIPPED`: implemented in code and covered by tests
- `SPEC_UPDATE`: wording narrowed to the shipped surface intentionally
- `REJECTED`: not needed, removed from the design on purpose
```

- [ ] **Step 5: Commit the hardening close-out**

```bash
git add docs/superpowers/reviews/2026-04-22-agent-runtime-audit.md
git commit -m "docs: close runtime hardening audit"
```

## Coverage Check

This plan closes the remaining runtime-refactor hardening risks:

1. spec wording drift vs shipped symbols: Task 1
2. missing runtime-decision query surface: Task 2
3. scheduler/console consumption of those runtime facts: Task 3
4. end-to-end live vs replay parity: Task 4
5. remaining query correctness and runtime hotspot cleanup: Task 5
6. final validation against the whole shipped runtime: Task 6

There are no uncovered review/hardening categories in the current post-refactor scope.
