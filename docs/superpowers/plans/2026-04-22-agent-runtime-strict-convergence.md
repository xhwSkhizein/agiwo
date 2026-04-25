# Agent Runtime Strict Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `RunLog` the complete runtime truth for `agiwo.agent`, tighten the public hook contract, make `RunStateWriter` the only runtime-truth write path, and move replayable trace/stream output onto committed `RunLog` projections.

**Architecture:** Keep `RunLoopOrchestrator` as the only phase decider, but route every committed runtime mutation through a new `RunStateWriter` class that updates in-memory state and appends typed `RunLog` entries together. Then make live trace updates and replayable stream events consume the same committed entry projections instead of direct runtime callbacks. Preserve `StepDeltaEvent` as the only live-only transport exception.

**Tech Stack:** Python 3.10+, dataclasses, `RunLogStorage`, `RunLoopOrchestrator`, `HookRegistry`, `SessionRuntime`, pytest, ruff

---

## File Structure

- `agiwo/agent/models/log.py`
  Add the missing `CompactionFailed` runtime fact and keep run-log families authoritative.
- `agiwo/agent/models/runtime_decision.py`
  Extend replayable latest-decision views if `compaction` failure state is exposed directly.
- `agiwo/agent/storage/serialization.py`
  Teach run-log serialization/replay builders about new entry kinds and stricter replay semantics.
- `agiwo/agent/models/stream.py`
  Keep `StepDeltaEvent` live-only, but rebuild all replayable stream items from committed `RunLog` entries.
- `agiwo/agent/hooks.py`
  Formalize hook groups, phase names, capability validation, transform allowlists, and critical-hook limits.
- `agiwo/agent/runtime/state_writer.py`
  Replace entry-builder-only helpers with a real `RunStateWriter` class that owns runtime-truth writes.
- `agiwo/agent/run_loop.py`
  Move run lifecycle, early failure handling, termination writes, and phase transitions onto the writer.
- `agiwo/agent/run_bootstrap.py`
  Stop direct message-state mutation from bypassing the writer during context assembly.
- `agiwo/agent/run_tool_batch.py`
  Route tool-step commits and step-back message rewrites through the writer.
- `agiwo/agent/compaction.py`
  Convert `compaction` success/failure into writer-mediated runtime facts.
- `agiwo/agent/runtime/step_committer.py`
  Reduce to a writer wrapper or delete after call sites move to `RunStateWriter`.
- `agiwo/agent/runtime/session.py`
  Add a projection dispatch path that turns committed entries into live trace and replayable stream updates.
- `agiwo/agent/trace_writer.py`
  Remove direct step/run callbacks from the canonical path and consume committed entries only.
- `tests/agent/test_storage_serialization.py`
  Verify new run-log entries round-trip and replay cleanly.
- `tests/agent/test_hook_dispatcher.py`
  Lock down hook ordering, validation, and rejected mutations.
- `tests/agent/test_run_engine.py`
  Cover early fatal failures and run-terminal completeness.
- `tests/agent/test_compact.py`
  Cover `CompactionFailed` facts and `compaction` projection semantics.
- `tests/agent/test_run_log_replay_parity.py`
  Keep replayable live stream parity aligned with replayed `RunLog`.
- `tests/observability/test_collector.py`
  Verify trace construction uses committed run-log entries rather than direct callbacks.
- `console/tests/test_runtime_replay_consistency.py`
  Ensure console-visible replay semantics stay aligned after the stricter projection model.
- `docs/superpowers/specs/2026-04-22-agent-runtime-strict-convergence-design.md`
  Reference implementation-facing naming if minor wording adjustments are needed during execution.
- `AGENTS.md`
  Update runtime architecture wording once the writer/projection contract is shipped.

### Task 1: Add Missing RunLog Facts And Replay Types

**Files:**
- Modify: `agiwo/agent/models/log.py`
- Modify: `agiwo/agent/storage/serialization.py`
- Modify: `agiwo/agent/models/stream.py`
- Modify: `agiwo/agent/models/runtime_decision.py`
- Test: `tests/agent/test_storage_serialization.py`
- Test: `tests/agent/test_run_log_models.py`

- [ ] **Step 1: Write the failing tests for `CompactionFailed` round-trip and replay**

```python
# tests/agent/test_storage_serialization.py
from agiwo.agent.models.log import CompactionFailed
from agiwo.agent.storage.serialization import (
    deserialize_run_log_entry_from_storage,
    serialize_run_log_entry_for_storage,
)


def test_compaction_failed_round_trips_through_storage() -> None:
    entry = CompactionFailed(
        sequence=7,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        error="compact boom",
        attempt=2,
        max_attempts=3,
        terminal=False,
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert isinstance(restored, CompactionFailed)
    assert restored.error == "compact boom"
    assert restored.attempt == 2
    assert restored.max_attempts == 3
    assert restored.terminal is False
```

```python
# tests/agent/test_run_log_models.py
from agiwo.agent.models.log import CompactionFailed, RunLogEntryKind
from agiwo.agent.models.stream import stream_items_from_entries


def test_stream_items_from_entries_replays_compaction_failed_event() -> None:
    items = stream_items_from_entries(
        [
            CompactionFailed(
                sequence=4,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                error="compact boom",
                attempt=1,
                max_attempts=3,
                terminal=False,
            )
        ]
    )

    assert items[0].type == "compaction_failed"
    assert items[0].error == "compact boom"
    assert RunLogEntryKind.COMPACTION_FAILED.value == "compaction_failed"
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest tests/agent/test_storage_serialization.py::test_compaction_failed_round_trips_through_storage tests/agent/test_run_log_models.py::test_stream_items_from_entries_replays_compaction_failed_event -v`
Expected: FAIL with `ImportError` or `AttributeError` because `CompactionFailed` and the replay event do not exist yet.

- [ ] **Step 3: Add the new run-log entry, serialization registration, and replayable stream event**

```python
# agiwo/agent/models/log.py
class RunLogEntryKind(str, Enum):
    ...
    COMPACTION_FAILED = "compaction_failed"


@dataclass(frozen=True, kw_only=True)
class CompactionFailed(RunLogEntry):
    error: str
    attempt: int
    max_attempts: int
    terminal: bool = False
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.COMPACTION_FAILED
    )
```

```python
# agiwo/agent/storage/serialization.py
from agiwo.agent.models.log import CompactionFailed

_RUN_LOG_TYPES[RunLogEntryKind.COMPACTION_FAILED] = CompactionFailed
```

```python
# agiwo/agent/models/stream.py
from agiwo.agent.models.log import CompactionFailed


@dataclass(kw_only=True)
class CompactionFailedEvent(AgentStreamItemBase):
    error: str
    attempt: int
    max_attempts: int
    terminal: bool
    type: Literal["compaction_failed"] = "compaction_failed"


elif isinstance(entry, CompactionFailed):
    item = CompactionFailedEvent(
        **base_kwargs,
        error=entry.error,
        attempt=entry.attempt,
        max_attempts=entry.max_attempts,
        terminal=entry.terminal,
    )
```

```python
# agiwo/agent/models/runtime_decision.py
@dataclass(frozen=True, slots=True)
class CompactionFailureDecisionView:
    session_id: str
    run_id: str
    agent_id: str
    sequence: int
    created_at: datetime
    error: str
    attempt: int
    max_attempts: int
    terminal: bool
```

- [ ] **Step 4: Run the focused tests and related replay tests**

Run: `uv run pytest tests/agent/test_storage_serialization.py tests/agent/test_run_log_models.py tests/agent/test_runtime_decision_views.py -v`
Expected: PASS with `CompactionFailed` serialized, deserialized, and replayed into public stream/runtime-decision views.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/models/log.py agiwo/agent/storage/serialization.py agiwo/agent/models/stream.py agiwo/agent/models/runtime_decision.py tests/agent/test_storage_serialization.py tests/agent/test_run_log_models.py tests/agent/test_runtime_decision_views.py
git commit -m "refactor: add strict compaction failure run log facts"
```

### Task 2: Tighten The Public Hook Contract

**Files:**
- Modify: `agiwo/agent/hooks.py`
- Modify: `agiwo/agent/definition.py`
- Test: `tests/agent/test_hook_dispatcher.py`
- Test: `tests/agent/test_memory_hooks.py`

- [ ] **Step 1: Write failing tests for hook groups, invalid transform keys, and critical validation**

```python
# tests/agent/test_hook_dispatcher.py
import pytest

from agiwo.agent.hooks import (
    HookCapability,
    HookGroup,
    HookPhase,
    HookRegistration,
    HookRegistry,
    observe,
    transform,
)


@pytest.mark.asyncio
async def test_hook_registry_orders_group_then_order_then_registration() -> None:
    seen: list[str] = []

    async def handler(name: str):
        async def _run(payload: dict) -> dict:
            seen.append(name)
            return payload
        return _run

    registry = HookRegistry(
        registrations=[
            HookRegistration(
                phase=HookPhase.BEFORE_LLM,
                group=HookGroup.USER,
                capability=HookCapability.TRANSFORM,
                handler_name="user_second",
                handler=await handler("user_second"),
                order=200,
            ),
            HookRegistration(
                phase=HookPhase.BEFORE_LLM,
                group=HookGroup.SYSTEM,
                capability=HookCapability.TRANSFORM,
                handler_name="system_first",
                handler=await handler("system_first"),
                order=100,
            ),
            HookRegistration(
                phase=HookPhase.BEFORE_LLM,
                group=HookGroup.RUNTIME_ADAPTER,
                capability=HookCapability.TRANSFORM,
                handler_name="adapter_middle",
                handler=await handler("adapter_middle"),
                order=100,
            ),
        ]
    )

    await registry._dispatch(
        HookPhase.BEFORE_LLM,
        {"messages": [{"role": "user", "content": "hi"}], "context": object()},
        allow_transform=True,
    )

    assert seen == ["system_first", "adapter_middle", "user_second"]


@pytest.mark.asyncio
async def test_hook_registry_rejects_transform_fields_outside_phase_allowlist() -> None:
    async def bad_transform(payload: dict) -> dict:
        updated = dict(payload)
        updated["illegal_field"] = "boom"
        return updated

    registry = HookRegistry(
        registrations=[
            transform(HookPhase.BEFORE_TOOL_CALL, "bad", bad_transform)
        ]
    )

    with pytest.raises(ValueError, match="illegal_field"):
        await registry._dispatch(
            HookPhase.BEFORE_TOOL_CALL,
            {
                "tool_call_id": "call-1",
                "tool_name": "bash",
                "parameters": {"cmd": "pwd"},
                "context": object(),
            },
            allow_transform=True,
        )


def test_hook_registry_rejects_critical_after_phase() -> None:
    async def after_llm(payload: dict) -> None:
        del payload

    with pytest.raises(ValueError, match="critical"):
        HookRegistry(
            registrations=[
                observe(
                    HookPhase.AFTER_LLM,
                    "after_llm",
                    after_llm,
                    critical=True,
                )
            ]
        )
```

- [ ] **Step 2: Run the targeted hook tests to verify they fail**

Run: `uv run pytest tests/agent/test_hook_dispatcher.py::test_hook_registry_orders_group_then_order_then_registration tests/agent/test_hook_dispatcher.py::test_hook_registry_rejects_transform_fields_outside_phase_allowlist tests/agent/test_hook_dispatcher.py::test_hook_registry_rejects_critical_after_phase -v`
Expected: FAIL because `HookGroup`, transform allowlists, and the stricter critical validation are not implemented.

- [ ] **Step 3: Add hook groups, per-phase allowlists, and the canonical public phase names**

```python
# agiwo/agent/hooks.py
class HookPhase(str, Enum):
    PREPARE = "prepare"
    ASSEMBLE_CONTEXT = "assemble_context"
    BEFORE_LLM = "before_llm"
    AFTER_LLM = "after_llm"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    BEFORE_COMPACTION = "before_compaction"
    AFTER_COMPACTION = "after_compaction"
    COMPACTION_FAILED = "compaction_failed"
    BEFORE_REVIEW = "before_review"
    AFTER_STEP_BACK = "after_step_back"
    BEFORE_TERMINATION = "before_termination"
    AFTER_TERMINATION = "after_termination"
    AFTER_STEP_COMMIT = "after_step_commit"
    RUN_FINALIZED = "run_finalized"
    MEMORY_PERSIST = "memory_persist"


class HookGroup(str, Enum):
    SYSTEM = "system"
    RUNTIME_ADAPTER = "runtime_adapter"
    USER = "user"


_TRANSFORM_ALLOWLISTS = {
    HookPhase.PREPARE: {"prelude_text"},
    HookPhase.ASSEMBLE_CONTEXT: {"memories", "context_additions"},
    HookPhase.BEFORE_LLM: {"messages", "model_settings_override"},
    HookPhase.BEFORE_TOOL_CALL: {"parameters"},
}

_DECISION_SUPPORT_ALLOWLISTS = {
    HookPhase.BEFORE_LLM: {"llm_advice"},
    HookPhase.BEFORE_TOOL_CALL: {"tool_advice"},
    HookPhase.BEFORE_COMPACTION: {"compaction_advice"},
    HookPhase.BEFORE_REVIEW: {"step-back_advice"},
    HookPhase.BEFORE_TERMINATION: {"termination_advice"},
}
```

```python
# agiwo/agent/hooks.py
@dataclass(frozen=True)
class HookRegistration:
    phase: HookPhase
    capability: HookCapability
    handler_name: str
    handler: PhaseHook
    group: HookGroup = HookGroup.USER
    order: int = 100
    critical: bool = False
```

```python
# agiwo/agent/hooks.py
def for_phase(self, phase: HookPhase) -> list[HookRegistration]:
    indexed = list(enumerate(self.registrations))
    matching = [(idx, item) for idx, item in indexed if item.phase == phase]
    matching.sort(
        key=lambda pair: (
            _GROUP_ORDER[pair[1].group],
            pair[1].order,
            pair[0],
        )
    )
    return [item for _, item in matching]
```

```python
# agiwo/agent/definition.py
resolved.add(
    transform(
        HookPhase.ASSEMBLE_CONTEXT,
        "default_memory_retrieve",
        _memory_retrieve,
        group=HookGroup.SYSTEM,
    )
)
```

- [ ] **Step 4: Run the hook and memory-hook suites**

Run: `uv run pytest tests/agent/test_hook_dispatcher.py tests/agent/test_memory_hooks.py tests/agent/test_run_loop_contracts.py -v`
Expected: PASS with deterministic group ordering, rejected illegal mutations, and memory retrieval registered as a system hook.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/hooks.py agiwo/agent/definition.py tests/agent/test_hook_dispatcher.py tests/agent/test_memory_hooks.py tests/agent/test_run_loop_contracts.py
git commit -m "refactor: tighten public hook contract"
```

### Task 3: Promote `RunStateWriter` And Guarantee Terminal Completeness

**Files:**
- Modify: `agiwo/agent/runtime/state_writer.py`
- Modify: `agiwo/agent/run_loop.py`
- Modify: `agiwo/agent/run_bootstrap.py`
- Modify: `agiwo/agent/runtime/step_committer.py`
- Test: `tests/agent/test_run_engine.py`
- Test: `tests/agent/test_run_loop_contracts.py`
- Test: `tests/agent/test_state_tracking.py`

- [ ] **Step 1: Write failing tests for early fatal failure and writer-mediated lifecycle commits**

```python
# tests/agent/test_run_engine.py
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, AgentConfig
from agiwo.agent.hooks import HookPhase, HookRegistry, transform
from agiwo.llm.base import Model, StreamChunk


class _NeverCalledModel(Model):
    def __init__(self) -> None:
        super().__init__(id="never", name="never", temperature=0.0)

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        raise AssertionError("llm should not run when prepare fails")


@pytest.mark.asyncio
async def test_prepare_failure_after_run_started_writes_run_failed() -> None:
    async def explode(payload: dict) -> dict:
        raise RuntimeError("prepare boom")

    agent = Agent(
        AgentConfig(name="strict-run", description="strict run"),
        model=_NeverCalledModel(),
        hooks=HookRegistry(
            [
                transform(
                    HookPhase.PREPARE,
                    "explode",
                    explode,
                    critical=True,
                )
            ]
        ),
    )

    with pytest.raises(RuntimeError, match="prepare boom"):
        await agent.run("hello", session_id="strict-run-session")

    entries = await agent.run_log_storage.list_entries(session_id="strict-run-session")
    assert [entry.kind.value for entry in entries][-1] == "run_failed"
```

```python
# tests/agent/test_state_tracking.py
from agiwo.agent.runtime.state_writer import RunStateWriter


@pytest.mark.asyncio
async def test_run_state_writer_commit_user_step_updates_state_and_returns_entry(
    _make_state,
) -> None:
    state = _make_state()
    writer = RunStateWriter(state)
    step = StepView.user(state, sequence=1, user_input="hello")

    committed = await writer.commit_step(step)

    assert committed[0].kind.value == "user_step_committed"
    assert state.ledger.steps.total == 1
    assert state.ledger.messages[-1]["role"] == "user"
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest tests/agent/test_run_engine.py::test_prepare_failure_after_run_started_writes_run_failed tests/agent/test_state_tracking.py::test_run_state_writer_commit_user_step_updates_state_and_returns_entry -v`
Expected: FAIL because `RunStateWriter` is not a real write coordinator and early `prepare` failures can still leave only `RunStarted`.

- [ ] **Step 3: Replace helper builders with a real `RunStateWriter` class and route lifecycle commits through it**

```python
# agiwo/agent/runtime/state_writer.py
class RunStateWriter:
    def __init__(self, state: RunContext) -> None:
        self._state = state

    async def append_entries(self, entries: list[RunLogEntry]) -> list[RunLogEntry]:
        await self._state.session_runtime.append_run_log_entries(entries)
        return entries

    async def start_run(self, user_input: UserInput) -> list[RunLogEntry]:
        entry = build_run_started_entry(
            self._state,
            sequence=await self._state.session_runtime.allocate_sequence(),
            user_input=user_input,
        )
        return await self.append_entries([entry])

    async def fail_run(self, error: Exception) -> list[RunLogEntry]:
        entry = build_run_failed_entry(
            self._state,
            sequence=await self._state.session_runtime.allocate_sequence(),
            error=error,
        )
        return await self.append_entries([entry])

    async def commit_step(
        self,
        step: StepView,
        *,
        llm: LLMCallContext | None = None,
        append_message: bool = True,
        track_state: bool = True,
    ) -> list[RunLogEntry]:
        if track_state:
            track_step_state(self._state, step, append_message=append_message)
        entry = build_step_log_entry(step)
        return await self.append_entries([entry])
```

```python
# agiwo/agent/run_loop.py
writer = RunStateWriter(self.context)
await writer.start_run(user_input)

try:
    bootstrap = await prepare_run_context(
        context=self.context,
        runtime=self.runtime,
        user_input=user_input,
        system_prompt=system_prompt,
        writer=writer,
    )
    ...
except Exception as error:
    await writer.fail_run(error)
    raise
```

```python
# agiwo/agent/run_bootstrap.py
async def prepare_run_context(..., writer: RunStateWriter) -> RunBootstrapResult:
    ...
    await writer.record_context_assembled(
        messages=context.snapshot_messages(),
        memory_count=len(memories),
    )
```

- [ ] **Step 4: Run the lifecycle and state suites**

Run: `uv run pytest tests/agent/test_run_engine.py tests/agent/test_run_loop_contracts.py tests/agent/test_state_tracking.py tests/agent/test_run_contracts.py -v`
Expected: PASS with early fatal failures ending in `RunFailed` and step/lifecycle commits flowing through `RunStateWriter`.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/runtime/state_writer.py agiwo/agent/run_loop.py agiwo/agent/run_bootstrap.py agiwo/agent/runtime/step_committer.py tests/agent/test_run_engine.py tests/agent/test_run_loop_contracts.py tests/agent/test_state_tracking.py tests/agent/test_run_contracts.py
git commit -m "refactor: route run lifecycle through state writer"
```

### Task 4: Route Tool, Compaction, And StepBack Writes Through The Writer

**Files:**
- Modify: `agiwo/agent/run_tool_batch.py`
- Modify: `agiwo/agent/compaction.py`
- Modify: `agiwo/agent/review/executor.py`
- Modify: `agiwo/agent/runtime/state_writer.py`
- Modify: `agiwo/agent/tool_executor.py`
- Test: `tests/agent/test_compact.py`
- Test: `tests/agent/test_step-back.py`
- Test: `tests/agent/test_tool_auth_runtime.py`
- Test: `tests/agent/test_run_engine.py`

- [ ] **Step 1: Write failing tests for `CompactionFailed` commits and writer-owned message rebuilds**

```python
# tests/agent/test_compact.py
import pytest

from agiwo.agent.compaction import CompactResult
from agiwo.agent.models.log import CompactionFailed


@pytest.mark.asyncio
async def test_run_loop_records_compaction_failed_entry_on_failed_attempt(
    monkeypatch,
    _make_context,
    _make_runtime,
) -> None:
    async def fake_compact_if_needed(**kwargs):
        del kwargs
        return CompactResult(failed=True, error="compact boom")

    monkeypatch.setattr("agiwo.agent.run_loop.compact_if_needed", fake_compact_if_needed)

    orchestrator = RunLoopOrchestrator(_make_context(), _make_runtime())
    await orchestrator._run_compaction_cycle()

    entries = await orchestrator.context.session_runtime.list_run_log_entries()
    assert any(isinstance(entry, CompactionFailed) for entry in entries)
```

```python
# tests/agent/test_step-back.py
@pytest.mark.asyncio
async def test_step-back_message_rebuild_goes_through_state_writer(
    monkeypatch, _make_context
) -> None:
    state = _make_context()
    writer_calls: list[str] = []

    class _WriterSpy(RunStateWriter):
        async def rebuild_messages(self, *, reason: str, messages: list[dict]):
            writer_calls.append(reason)
            return await super().rebuild_messages(reason=reason, messages=messages)

    ...

    assert writer_calls == ["step-back"]
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest tests/agent/test_compact.py::test_run_loop_records_compaction_failed_entry_on_failed_attempt tests/agent/test_step-back.py::test_step-back_message_rebuild_goes_through_state_writer -v`
Expected: FAIL because `compaction` failure does not append `CompactionFailed` and message rewrites still bypass the writer.

- [ ] **Step 3: Add writer methods for message rebuilds, `compaction` success/failure, tool-step commits, and step-back application**

```python
# agiwo/agent/runtime/state_writer.py
async def rebuild_messages(
    self,
    *,
    reason: str,
    messages: list[dict[str, Any]],
) -> list[RunLogEntry]:
    replace_messages(self._state, messages)
    entry = build_messages_rebuilt_entry(
        self._state,
        sequence=await self._state.session_runtime.allocate_sequence(),
        reason=reason,
        messages=self._state.snapshot_messages(),
    )
    return await self.append_entries([entry])


async def record_compaction_failed(
    self,
    *,
    error: str,
    attempt: int,
    max_attempts: int,
    terminal: bool,
) -> list[RunLogEntry]:
    self._state.ledger.compaction.failure_count = attempt
    entry = build_compaction_failed_entry(
        self._state,
        sequence=await self._state.session_runtime.allocate_sequence(),
        error=error,
        attempt=attempt,
        max_attempts=max_attempts,
        terminal=terminal,
    )
    return await self.append_entries([entry])
```

```python
# agiwo/agent/run_tool_batch.py
tool_entries = await writer.commit_step(tool_step)
await session_runtime.project_committed_entries(tool_entries, context=context)

if outcome.applied:
    rebuilt_entries = await writer.rebuild_messages(
        reason="step-back",
        messages=outcome.messages,
    )
    step-back_entries = await writer.record_step_back_applied(...)
    await session_runtime.project_committed_entries(
        [*rebuilt_entries, *step-back_entries],
        context=context,
    )
```

```python
# agiwo/agent/compaction.py
if result.failed:
    await writer.record_compaction_failed(
        error=result.error or "unknown",
        attempt=failure_count,
        max_attempts=retry_count + 1,
        terminal=failure_count >= 3,
    )
else:
    rebuilt_entries = await writer.rebuild_messages(
        reason="compaction",
        messages=compacted_messages,
    )
    applied_entries = await writer.record_compaction_applied(metadata)
```

- [ ] **Step 4: Run the compaction, step-back, and tool suites**

Run: `uv run pytest tests/agent/test_compact.py tests/agent/test_step-back.py tests/agent/test_tool_auth_runtime.py tests/agent/test_run_engine.py -v`
Expected: PASS with `CompactionFailed` committed, step-back rewrites mediated by the writer, and tool steps still producing stable results.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/runtime/state_writer.py agiwo/agent/run_tool_batch.py agiwo/agent/compaction.py agiwo/agent/review/executor.py agiwo/agent/tool_executor.py tests/agent/test_compact.py tests/agent/test_step-back.py tests/agent/test_tool_auth_runtime.py tests/agent/test_run_engine.py
git commit -m "refactor: route runtime decision writes through state writer"
```

### Task 5: Make Replayable Trace And Stream Pure RunLog Projections

**Files:**
- Modify: `agiwo/agent/runtime/session.py`
- Modify: `agiwo/agent/trace_writer.py`
- Modify: `agiwo/agent/models/stream.py`
- Modify: `agiwo/agent/run_loop.py`
- Modify: `agiwo/agent/runtime/step_committer.py`
- Modify: `agiwo/agent/compaction.py`
- Modify: `agiwo/agent/run_tool_batch.py`
- Modify: `AGENTS.md`
- Test: `tests/agent/test_run_log_replay_parity.py`
- Test: `tests/observability/test_collector.py`
- Test: `console/tests/test_runtime_replay_consistency.py`

- [ ] **Step 1: Write failing parity tests that prove live replayable outputs must come from committed entries**

```python
# tests/observability/test_collector.py
import pytest

from agiwo.agent.trace_writer import AgentTraceCollector


@pytest.mark.asyncio
async def test_trace_collector_builds_live_trace_from_run_log_entries_only() -> None:
    collector = AgentTraceCollector()
    collector.start(trace_id="trace-1", agent_id="agent-1", session_id="sess-1")

    with pytest.raises(RuntimeError, match="trace_not_started|direct callbacks disabled"):
        await collector.on_step(
            StepView.assistant(
                _ReplayContext("sess-1", "run-1", "agent-1"),
                sequence=1,
                content="hello",
            )
        )
```

```python
# tests/agent/test_run_log_replay_parity.py
def test_live_stream_matches_replayed_run_log_for_all_replayable_types() -> None:
    replayable = {
        "run_started",
        "step_completed",
        "messages_rebuilt",
        "compaction_applied",
        "compaction_failed",
        "step_back_applied",
        "termination_decided",
        "run_completed",
        "run_failed",
    }
    ...
```

- [ ] **Step 2: Run the parity tests to verify they fail**

Run: `uv run pytest tests/agent/test_run_log_replay_parity.py tests/observability/test_collector.py console/tests/test_runtime_replay_consistency.py -v`
Expected: FAIL because replayable live outputs still rely on direct runtime callbacks/publish calls instead of committed run-log projection.

- [ ] **Step 3: Add entry-projection dispatch in `SessionRuntime` and remove direct replayable trace/stream callbacks**

```python
# agiwo/agent/runtime/session.py
async def project_committed_entries(
    self,
    entries: list[RunLogEntry],
    *,
    context: RunContext | None = None,
) -> None:
    if self.trace_runtime is not None:
        await self.trace_runtime.on_run_log_entries(entries)

    run_contexts = None
    if context is not None:
        run_contexts = {
            context.run_id: {
                "parent_run_id": context.parent_run_id,
                "depth": context.depth,
            }
        }

    for item in stream_items_from_entries(entries, run_contexts=run_contexts):
        await self.publish(item)
```

```python
# agiwo/agent/trace_writer.py
class AgentTraceCollector:
    ...
    async def on_step(self, step: StepView, llm: LLMCallContext | None = None) -> None:
        raise RuntimeError("direct callbacks disabled; use on_run_log_entries")

    async def on_run_completed(self, output: RunOutput, *, run_id: str) -> None:
        raise RuntimeError("direct callbacks disabled; use on_run_log_entries")
```

```python
# agiwo/agent/run_loop.py
entries = await writer.start_run(user_input)
await self.context.session_runtime.project_committed_entries(entries, context=self.context)

...

entries = await writer.record_termination(...)
await self.context.session_runtime.project_committed_entries(entries, context=self.context)
```

```python
# AGENTS.md
- `RunStateWriter` is the only runtime-truth write path inside `agiwo.agent`; replayable trace/stream outputs are projected from committed `RunLog` entries, with `StepDeltaEvent` as the only live-only exception.
- Hook public phases are `before_tool_call` / `after_tool_call`; the older batch wording is retired.
```

- [ ] **Step 4: Run the replay, observability, console, and lint gates**

Run: `uv run pytest tests/agent/test_run_log_replay_parity.py tests/observability/test_collector.py console/tests/test_runtime_replay_consistency.py tests/scheduler/test_runtime_facts.py -v`
Expected: PASS with live replayable stream/trace outputs matching replayed `RunLog`.

Run: `uv run python scripts/lint.py ci`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/runtime/session.py agiwo/agent/trace_writer.py agiwo/agent/models/stream.py agiwo/agent/run_loop.py agiwo/agent/runtime/step_committer.py agiwo/agent/compaction.py agiwo/agent/run_tool_batch.py AGENTS.md tests/agent/test_run_log_replay_parity.py tests/observability/test_collector.py console/tests/test_runtime_replay_consistency.py tests/scheduler/test_runtime_facts.py
git commit -m "refactor: project replayable runtime views from run log"
```

## Spec Coverage Check

1. `RunLog` complete runtime truth and no missing `RunFailed`
   Covered by Task 3 and Task 4.
2. Hook contract fully aligned with spec
   Covered by Task 2.
3. `compaction` failure as first-class runtime fact
   Covered by Task 1 and Task 4.
4. Trace and stream as `RunLog`-only replayable view builders
   Covered by Task 5.
5. `RunStateWriter` as the only runtime-truth write path
   Covered by Task 3 and Task 4.
6. Public phase set updated to shipped better design
   Covered by Task 2 and Task 5.

## Self-Review

1. Placeholder scan
   No `TBD`, `TODO`, or deferred “implement later” steps remain.
2. Internal consistency
   The plan uses one architecture throughout: orchestrator decides, writer commits, builders project.
3. Scope check
   This remains one implementation plan because all work converges on the same runtime boundary.
4. Ambiguity check
   The plan explicitly preserves `StepDeltaEvent` as the only live-only exception and explicitly retires `before_tool_batch` / `after_tool_batch` public naming.
