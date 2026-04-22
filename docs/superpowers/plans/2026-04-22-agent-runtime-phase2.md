# Agent Runtime Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate scheduler runtime semantics onto the new `RunLog` fact model so root/child execution, periodic rollback, wake/cancel/wait behavior, and scheduler-side run summaries all depend on replayable runtime facts instead of deleted-step or pre-refactor assumptions.

**Architecture:** Phase 2 keeps `Scheduler` as the orchestration owner, but moves its runtime reads onto `RunLog`-backed facts and views. The main structural change is to replace the last destructive scheduler write path (`delete_steps`) with append-only rollback facts, then centralize scheduler-side run-view lookup so `wait_for()`, child-result collection, and runtime summaries read the same replayed truth.

**Tech Stack:** Python 3.10+, dataclasses, existing `RunLogEntry` / `RunLogStorage` / `RunView` / `StepView` abstractions, scheduler facade/runner/store modules, pytest, ruff

---

## Scope Check

This plan covers only **Phase 2: Scheduler Migration** from the design spec:

1. migrate scheduler runtime integration to consume `RunLog` views and replay
2. remove direct dependence on old run/step storage semantics
3. align root/child/steer/wait/cancel semantics with the new runtime facts

It does **not** cover the full console read/query cleanup. Console can consume the new scheduler/runtime facades after this phase, but the main console materialization/query migration remains Phase 3.

## File Structure

### Create

- `agiwo/scheduler/runtime_facts.py`
- `tests/scheduler/test_runtime_facts.py`

### Modify

- `agiwo/agent/models/log.py`
- `agiwo/agent/models/stream.py`
- `agiwo/agent/storage/base.py`
- `agiwo/agent/storage/serialization.py`
- `agiwo/agent/storage/sqlite.py`
- `agiwo/scheduler/engine.py`
- `agiwo/scheduler/runner.py`
- `agiwo/scheduler/runtime_tools.py`
- `agiwo/scheduler/__init__.py`
- `tests/scheduler/test_context_rollback.py`
- `tests/scheduler/test_scheduler.py`
- `tests/agent/test_storage_serialization.py`
- `docs/guides/context-optimization.md`
- `docs/architecture/scheduler-console-runtime-refactor.md`

### Responsibilities

- `agiwo/agent/models/log.py`
  - Add a rollback fact family for append-only scheduler rollback recording.
- `agiwo/agent/storage/serialization.py`
  - Teach replay/builders to hide rolled-back step ranges by default.
- `agiwo/agent/storage/base.py`
  - Expose append-only rollback recording and step-view filtering knobs.
- `agiwo/agent/storage/sqlite.py`
  - Persist and replay rollback facts without mutating canonical committed steps.
- `agiwo/scheduler/runtime_facts.py`
  - Centralize `AgentState -> runtime session -> run views / step views / latest run facts`.
- `agiwo/scheduler/runner.py`
  - Replace destructive rollback and ad hoc child-result reads with runtime-fact lookups.
- `agiwo/scheduler/engine.py`
  - Make `wait_for()` and related reads consume the same scheduler/runtime fact surface.
- `agiwo/scheduler/runtime_tools.py`
  - Keep scheduler tool summaries aligned with runtime-fact-backed state.

## Task 1: Add Append-Only Rollback Facts To `RunLog`

**Files:**
- Modify: `agiwo/agent/models/log.py`
- Modify: `agiwo/agent/storage/base.py`
- Modify: `agiwo/agent/storage/serialization.py`
- Modify: `agiwo/agent/storage/sqlite.py`
- Test: `tests/agent/test_storage_serialization.py`
- Test: `tests/scheduler/test_context_rollback.py`

- [ ] **Step 1: Write the failing replay test for rolled-back steps**

```python
from agiwo.agent.models.log import (
    RunRolledBack,
    UserStepCommitted,
    AssistantStepCommitted,
)
from agiwo.agent.storage.serialization import build_step_views_from_entries
from agiwo.agent import MessageRole


def test_build_step_views_hides_steps_covered_by_rollback_fact() -> None:
    entries = [
        UserStepCommitted(
            sequence=10,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            step_id="step-10",
            role=MessageRole.USER,
            content="u1",
        ),
        AssistantStepCommitted(
            sequence=11,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            step_id="step-11",
            role=MessageRole.ASSISTANT,
            content="a1",
        ),
        RunRolledBack(
            sequence=12,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            start_sequence=10,
            end_sequence=11,
            reason="no_progress",
        ),
    ]

    steps = build_step_views_from_entries(entries)

    assert steps == []
```

- [ ] **Step 2: Run the targeted replay tests to confirm the gap**

Run: `uv run pytest tests/agent/test_storage_serialization.py tests/scheduler/test_context_rollback.py -v`
Expected: FAIL because no rollback fact exists and replay still returns committed steps that should have been logically removed.

- [ ] **Step 3: Add a typed rollback entry family and replay filtering**

```python
# agiwo/agent/models/log.py
class RunLogEntryKind(str, Enum):
    ...
    RUN_ROLLED_BACK = "run_rolled_back"


@dataclass(frozen=True, kw_only=True)
class RunRolledBack(RunLogEntry):
    start_sequence: int
    end_sequence: int
    reason: str
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_ROLLED_BACK)
```

```python
# agiwo/agent/storage/serialization.py
def build_step_views_from_entries(
    entries: list[RunLogEntry],
    *,
    include_rolled_back: bool = False,
) -> list[StepView]:
    hidden_sequences: set[int] = set()
    ...
    if isinstance(entry, RunRolledBack):
        if not include_rolled_back:
            hidden_sequences.update(range(entry.start_sequence, entry.end_sequence + 1))
        continue
    ...
    if step_view.sequence in hidden_sequences:
        continue
```

```python
# agiwo/agent/storage/base.py
@abstractmethod
async def append_run_rollback(
    self,
    session_id: str,
    run_id: str,
    agent_id: str,
    start_sequence: int,
    end_sequence: int,
    reason: str,
) -> None:
    ...
```

- [ ] **Step 4: Implement the in-memory and SQLite append paths**

```python
# agiwo/agent/storage/base.py
async def append_run_rollback(... ) -> None:
    sequence = await self.allocate_sequence(session_id)
    await self.append_entries(
        [
            RunRolledBack(
                sequence=sequence,
                session_id=session_id,
                run_id=run_id,
                agent_id=agent_id,
                start_sequence=start_sequence,
                end_sequence=end_sequence,
                reason=reason,
            )
        ]
    )
```

```python
# agiwo/agent/storage/sqlite.py
async def append_run_rollback(... ) -> None:
    sequence = await self.allocate_sequence(session_id)
    await self.append_entries([RunRolledBack(...)])
```

- [ ] **Step 5: Re-run the rollback/replay tests**

Run: `uv run pytest tests/agent/test_storage_serialization.py tests/scheduler/test_context_rollback.py -v`
Expected: PASS, with replay now hiding rolled-back steps without mutating canonical committed-step entries.

- [ ] **Step 6: Commit the append-only rollback boundary**

```bash
git add agiwo/agent/models/log.py agiwo/agent/storage/base.py agiwo/agent/storage/serialization.py agiwo/agent/storage/sqlite.py tests/agent/test_storage_serialization.py tests/scheduler/test_context_rollback.py
git commit -m "refactor: record scheduler rollback as run log fact"
```

## Task 2: Add A Scheduler Runtime-Facts Facade

**Files:**
- Create: `agiwo/scheduler/runtime_facts.py`
- Modify: `agiwo/scheduler/__init__.py`
- Test: `tests/scheduler/test_runtime_facts.py`

- [ ] **Step 1: Write the failing scheduler runtime-facts tests**

```python
import pytest

from agiwo.agent import RunFinished, RunStarted, TerminationReason
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.scheduler.models import AgentState, AgentStateStatus
from agiwo.scheduler.runtime_facts import SchedulerRuntimeFacts


@pytest.mark.asyncio
async def test_runtime_facts_load_latest_run_view_for_state() -> None:
    storage = InMemoryRunLogStorage()
    session_id = "sess-1"
    await storage.append_entries(
        [
            RunStarted(
                sequence=1,
                session_id=session_id,
                run_id="run-1",
                agent_id="root",
                user_input="hello",
            ),
            RunFinished(
                sequence=2,
                session_id=session_id,
                run_id="run-1",
                agent_id="root",
                response="done",
                termination_reason=TerminationReason.COMPLETED,
            ),
        ]
    )
    state = AgentState(
        id="root",
        session_id=session_id,
        status=AgentStateStatus.COMPLETED,
        task="hello",
    )

    facts = SchedulerRuntimeFacts(storage=storage)
    latest = await facts.get_latest_run_view(state)

    assert latest is not None
    assert latest.run_id == "run-1"
    assert latest.response == "done"
```

- [ ] **Step 2: Run the targeted runtime-facts tests**

Run: `uv run pytest tests/scheduler/test_runtime_facts.py -v`
Expected: FAIL because `SchedulerRuntimeFacts` does not exist yet.

- [ ] **Step 3: Add the runtime-facts helper**

```python
# agiwo/scheduler/runtime_facts.py
from dataclasses import dataclass

from agiwo.agent import RunLogStorage, RunView, StepView
from agiwo.scheduler.models import AgentState


@dataclass(frozen=True)
class SchedulerRuntimeFacts:
    storage: RunLogStorage

    async def get_latest_run_view(self, state: AgentState) -> RunView | None:
        return await self.storage.get_latest_run_view(state.resolve_runtime_session_id())

    async def list_visible_steps(
        self,
        state: AgentState,
        *,
        run_id: str | None = None,
        limit: int = 1000,
    ) -> list[StepView]:
        return await self.storage.list_step_views(
            session_id=state.resolve_runtime_session_id(),
            agent_id=state.id,
            run_id=run_id,
            limit=limit,
        )
```

- [ ] **Step 4: Export the new helper and add session/run convenience methods**

```python
# agiwo/scheduler/__init__.py
from agiwo.scheduler.runtime_facts import SchedulerRuntimeFacts

__all__ = [
    ...,
    "SchedulerRuntimeFacts",
]
```

```python
# agiwo/scheduler/runtime_facts.py
async def get_latest_run_result_text(self, state: AgentState) -> str | None:
    latest = await self.get_latest_run_view(state)
    if latest is None:
        return None
    return latest.response
```

- [ ] **Step 5: Re-run the runtime-facts tests**

Run: `uv run pytest tests/scheduler/test_runtime_facts.py -v`
Expected: PASS with a stable helper for state-to-run-log lookup.

- [ ] **Step 6: Commit the scheduler runtime-facts facade**

```bash
git add agiwo/scheduler/runtime_facts.py agiwo/scheduler/__init__.py tests/scheduler/test_runtime_facts.py
git commit -m "refactor: add scheduler runtime facts facade"
```

## Task 3: Move Scheduler Rollback And Child Result Reads Onto Runtime Facts

**Files:**
- Modify: `agiwo/scheduler/runner.py`
- Modify: `agiwo/scheduler/runtime_tools.py`
- Test: `tests/scheduler/test_scheduler.py`
- Test: `tests/scheduler/test_context_rollback.py`

- [ ] **Step 1: Write the failing scheduler rollback and child-result tests**

```python
@pytest.mark.asyncio
async def test_periodic_no_progress_records_run_rollback_fact(scheduler, agent):
    state_id = await scheduler.submit(agent, "task", persistent=True)
    output = await scheduler.wait_for(state_id, timeout=2.0)

    runtime_agent = scheduler.get_registered_agent(state_id)
    entries = await runtime_agent.run_log_storage.list_entries(
        session_id=state_id,
        limit=100_000,
    )

    assert any(entry.kind.value == "run_rolled_back" for entry in entries)


@pytest.mark.asyncio
async def test_collect_child_results_prefers_latest_run_view_summary(scheduler, store):
    ...
    succeeded, failed = await scheduler._runner._collect_child_results(parent_state)
    assert succeeded["child-1"] == "assistant reply from run log"
```

- [ ] **Step 2: Run the targeted scheduler tests**

Run: `uv run pytest tests/scheduler/test_scheduler.py tests/scheduler/test_context_rollback.py -v`
Expected: FAIL because runner still calls `delete_steps(...)` and child-result summaries still fall back to `AgentState.result_summary`.

- [ ] **Step 3: Replace destructive rollback with append-only rollback recording**

```python
# agiwo/scheduler/runner.py
async def _rollback_run_steps(
    self,
    state: AgentState,
    output: RunOutput,
) -> None:
    run_start_seq = output.metadata.get("run_start_seq")
    run_id = output.run_id
    if run_start_seq is None or run_id is None:
        return
    agent = self._ctx.rt.agents.get(state.id)
    if agent is None:
        return

    session_id = state.resolve_runtime_session_id()
    storage = agent.run_log_storage
    max_sequence = await storage.get_max_sequence(session_id)
    await storage.append_run_rollback(
        session_id=session_id,
        run_id=run_id,
        agent_id=state.id,
        start_sequence=run_start_seq,
        end_sequence=max_sequence,
        reason="no_progress",
    )
```

- [ ] **Step 4: Route child-result collection through `SchedulerRuntimeFacts`**

```python
# agiwo/scheduler/runner.py
async def _collect_child_results(...):
    ...
    runtime_facts = SchedulerRuntimeFacts(child_agent.run_log_storage)
    latest = await runtime_facts.get_latest_run_view(child)
    if child.status == AgentStateStatus.COMPLETED:
        succeeded[child_id] = (
            latest.response if latest is not None and latest.response else "Completed"
        )
```

```python
# agiwo/scheduler/runtime_tools.py
latest_run = await facts.get_latest_run_view(state)
result = latest_run.response if latest_run is not None else state.result_summary
```

- [ ] **Step 5: Re-run the scheduler rollback/result tests**

Run: `uv run pytest tests/scheduler/test_scheduler.py tests/scheduler/test_context_rollback.py -v`
Expected: PASS, with no destructive run-log mutation and child summaries sourced from replayed run facts.

- [ ] **Step 6: Commit the runner migration**

```bash
git add agiwo/scheduler/runner.py agiwo/scheduler/runtime_tools.py tests/scheduler/test_scheduler.py tests/scheduler/test_context_rollback.py
git commit -m "refactor: move scheduler rollback and child summaries onto run log facts"
```

## Task 4: Align `wait_for()` And Scheduler Terminal Reads With Runtime Facts

**Files:**
- Modify: `agiwo/scheduler/engine.py`
- Modify: `agiwo/scheduler/runner.py`
- Test: `tests/scheduler/test_scheduler.py`

- [ ] **Step 1: Write the failing `wait_for()` parity tests**

```python
@pytest.mark.asyncio
async def test_wait_for_uses_latest_run_id_and_response_from_runtime_facts(scheduler, completed_root_state):
    output = await scheduler.wait_for(completed_root_state.id, timeout=2.0)

    assert output.run_id == "run-2"
    assert output.response == "latest assistant response"


@pytest.mark.asyncio
async def test_wait_for_cancelled_without_run_log_still_uses_last_run_result(scheduler, cancelled_state):
    output = await scheduler.wait_for(cancelled_state.id, timeout=2.0)
    assert output.termination_reason == TerminationReason.CANCELLED
```

- [ ] **Step 2: Run the scheduler wait tests**

Run: `uv run pytest tests/scheduler/test_scheduler.py -v`
Expected: FAIL because `wait_for()` only returns `state.last_run_result` and cannot reconcile it with the replayed latest run view when both exist.

- [ ] **Step 3: Make `wait_for()` prefer runtime facts when a runtime session exists**

```python
# agiwo/scheduler/engine.py
async def wait_for(self, state_id: str, timeout: float | None = None) -> RunOutput:
    ...
    if state is not None and state.status in (...):
        agent = self._rt.agents.get(state.id)
        if agent is not None:
            facts = SchedulerRuntimeFacts(agent.run_log_storage)
            latest = await facts.get_latest_run_view(state)
            if latest is not None:
                return RunOutput(
                    run_id=latest.run_id,
                    response=latest.response,
                    termination_reason=latest.termination_reason,
                )
        if state.last_run_result is not None:
            ...
```

- [ ] **Step 4: Keep runner writes and `last_run_result` as the durable fallback**

```python
# agiwo/scheduler/runner.py
last_run_result = self._build_last_run_result(
    termination_reason=output.termination_reason,
    run_id=output.run_id,
    summary=text,
    error=output.error,
)
```

- [ ] **Step 5: Re-run the scheduler wait/result tests**

Run: `uv run pytest tests/scheduler/test_scheduler.py -v`
Expected: PASS, with runtime-fact-backed reads for active runtime sessions and `last_run_result` preserved as the fallback when the runtime agent is already gone.

- [ ] **Step 6: Commit the `wait_for()` alignment**

```bash
git add agiwo/scheduler/engine.py agiwo/scheduler/runner.py tests/scheduler/test_scheduler.py
git commit -m "refactor: align scheduler wait reads with runtime facts"
```

## Task 5: Refresh Scheduler Docs And Run Full Validation

**Files:**
- Modify: `docs/guides/context-optimization.md`
- Modify: `docs/architecture/scheduler-console-runtime-refactor.md`
- Test: `tests/scheduler/test_runtime_facts.py`
- Test: `tests/scheduler/test_context_rollback.py`
- Test: `tests/scheduler/test_scheduler.py`

- [ ] **Step 1: Update the context optimization guide to reflect append-only rollback**

```md
系统收到 `no_progress` 后，不再物理删除 canonical `RunLog` entries。
相反，scheduler 会追加一条 rollback fact，默认 step replay 会隐藏该范围内的 steps，
所以下次唤醒时可见上下文等价于“这一轮空转没有留下可见 steps”。
```

- [ ] **Step 2: Update the scheduler/runtime architecture doc**

```md
- root/child scheduler result summaries are sourced from replayed `RunLog` views
- periodic rollback is append-only and replay-filtered, not destructive deletion
- `last_run_result` remains the scheduler-owned fallback when the runtime agent is gone
```

- [ ] **Step 3: Run the focused scheduler validation set**

Run: `uv run pytest tests/scheduler/test_runtime_facts.py tests/scheduler/test_context_rollback.py tests/scheduler/test_scheduler.py -v`
Expected: PASS, with rollback, wait, and child-result semantics all covered by the new runtime-facts path.

- [ ] **Step 4: Run repository lint and full scheduler-adjacent tests**

Run: `uv run python scripts/lint.py ci`
Expected: PASS

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit the docs and validation sweep**

```bash
git add docs/guides/context-optimization.md docs/architecture/scheduler-console-runtime-refactor.md tests/scheduler/test_runtime_facts.py tests/scheduler/test_context_rollback.py tests/scheduler/test_scheduler.py
git commit -m "docs: sync scheduler runtime fact migration"
```

## Self-Review

Spec coverage:

1. scheduler consumes `RunLog` views and replay: Task 2, Task 3, Task 4
2. old run/step destructive semantics removed: Task 1, Task 3
3. root/child/steer/wait/cancel semantics aligned with runtime facts: Task 3, Task 4
4. scheduler-facing runtime tests strengthened: Task 2, Task 3, Task 4, Task 5

Placeholder scan:

1. no `TBD` / `TODO` placeholders remain
2. each task includes concrete files, tests, commands, and code snippets

Type consistency:

1. rollback fact is named `RunRolledBack` / `RUN_ROLLED_BACK` consistently
2. scheduler read helper is named `SchedulerRuntimeFacts` consistently
3. `append_run_rollback(...)` is the only new storage write path introduced for scheduler rollback
