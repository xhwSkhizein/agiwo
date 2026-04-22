# Agent Runtime Final Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the strict convergence of `agiwo.agent` so steer input is lossless, `RunStateWriter` is the only committed runtime-truth write path, internal LLM turns are canonical `RunLog` facts, and trace construction/persistence is entry-only.

**Architecture:** Keep `RunLoopOrchestrator` as the only phase decider, but push the remaining committed-state mutations and internal LLM facts behind `RunStateWriter`. Replace destructive steer draining with staged commit semantics, and delete `AgentTraceCollector`'s legacy callback path so live trace, replay trace, and replayable stream/query views all consume committed `RunLog` entries only.

**Tech Stack:** Python 3.10+, dataclasses, `RunLogStorage`, `RunStateWriter`, `SessionRuntime`, `AgentTraceCollector`, pytest, ruff

---

## File Structure

- `agiwo/agent/run_loop.py`
  Replace destructive steer drain with staged commit semantics, remove direct compaction-failure counter mutation, and reuse a single projection helper.
- `agiwo/agent/runtime/state_writer.py`
  Add explicit writer-owned mutations for tool schemas, compaction metadata, `run_start_seq`, compaction failure count, and canonical internal LLM facts.
- `agiwo/agent/runtime/session.py`
  Add staged steer snapshot/ack semantics so accepted steer input is only consumed after committed message rewrite succeeds.
- `agiwo/agent/prompt.py`
  Remove the destructive queue-drain helper or reduce it to a pure message-shaping helper that does not own queue semantics.
- `agiwo/agent/run_bootstrap.py`
  Route remaining committed bootstrap state through writer methods.
- `agiwo/agent/compaction.py`
  Emit canonical `LLMCallStarted` / `LLMCallCompleted` for the internal compaction turn and stop mutating committed compaction state outside the writer.
- `agiwo/agent/termination/summarizer.py`
  Emit canonical LLM-call facts for the termination summary turn.
- `agiwo/agent/trace_writer.py`
  Delete legacy direct-callback APIs, bound committed caches, clear correlation state on completion, and persist live trace snapshots after committed entry batches.
- `tests/agent/test_run_loop_contracts.py`
  Add lossless steer and staged-consumption coverage.
- `tests/agent/test_run_engine.py`
  Cover `before_llm` failure after accepted steer input and bootstrap writer ownership.
- `tests/agent/test_compact.py`
  Cover canonical compaction LLM facts.
- `tests/agent/test_termination.py`
  Cover canonical termination-summary LLM facts.
- `tests/observability/test_collector.py`
  Switch fully to committed-entry trace construction and assert bounded cache / persistence behavior.
- `tests/agent/test_run_log_replay_parity.py`
  Keep live stream and replay stream parity after the steer and internal-LLM changes.
- `console/tests/test_runtime_replay_consistency.py`
  Verify console replay/query semantics still match committed entries.
- `AGENTS.md`
  Update stable runtime wording if implementation changes the documented contract surface again.

### Task 1: Make Steer Consumption Lossless

**Files:**
- Modify: `agiwo/agent/runtime/session.py`
- Modify: `agiwo/agent/run_loop.py`
- Modify: `agiwo/agent/prompt.py`
- Test: `tests/agent/test_run_loop_contracts.py`
- Test: `tests/agent/test_run_engine.py`

- [ ] **Step 1: Write failing tests for staged steer consumption**

```python
# tests/agent/test_run_loop_contracts.py
import asyncio

import pytest

from agiwo.agent.models.input import UserMessage
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage


@pytest.mark.asyncio
async def test_session_runtime_peek_steer_does_not_consume_until_ack() -> None:
    runtime = SessionRuntime(
        session_id="sess-1",
        run_log_storage=InMemoryRunLogStorage(),
    )
    await runtime.enqueue_steer("follow up")

    pending = runtime.peek_pending_steer_inputs()

    assert [UserMessage.from_value(item).extract_text() for item in pending] == [
        "follow up"
    ]
    assert [UserMessage.from_value(item).extract_text() for item in runtime.peek_pending_steer_inputs()] == [
        "follow up"
    ]

    runtime.ack_pending_steer_inputs(len(pending))
    assert runtime.peek_pending_steer_inputs() == []
```

```python
# tests/agent/test_run_engine.py
import pytest

from agiwo.agent import Agent, AgentConfig
from agiwo.agent.hooks import HookPhase, HookRegistry, transform
from agiwo.agent.models.input import UserMessage


@pytest.mark.asyncio
async def test_before_llm_failure_does_not_lose_accepted_steer_input() -> None:
    gate = {"calls": 0}

    async def explode_once(payload: dict) -> dict:
        gate["calls"] += 1
        if gate["calls"] == 1:
            raise RuntimeError("before llm boom")
        return payload

    agent = Agent(
        AgentConfig(name="steer-lossless", description="steer-lossless"),
        model=_FixedResponseModel(response="ok"),
        hooks=HookRegistry(
            [transform(HookPhase.BEFORE_LLM, "explode_once", explode_once, critical=True)]
        ),
    )

    handle = agent.start("hello", session_id="sess-steer-lossless")
    accepted = await handle.steer(UserMessage.from_value("follow up"))
    assert accepted is True

    with pytest.raises(RuntimeError, match="before llm boom"):
        await handle.wait()

    retry = await agent.run("retry", session_id="sess-steer-lossless")
    assert retry.response is not None
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest tests/agent/test_run_loop_contracts.py::test_session_runtime_peek_steer_does_not_consume_until_ack tests/agent/test_run_engine.py::test_before_llm_failure_does_not_lose_accepted_steer_input -v`
Expected: FAIL because `SessionRuntime` has only destructive queue consumption and `before_llm` failure can still lose accepted steer input.

- [ ] **Step 3: Implement staged steer snapshot and ack semantics**

```python
# agiwo/agent/runtime/session.py
class SessionRuntime:
    def __init__(..., steering_queue: asyncio.Queue[object] | None = None) -> None:
        ...
        self.steering_queue = steering_queue or asyncio.Queue()
        self._pending_steer_inputs: list[UserMessage] = []

    async def enqueue_steer(self, user_input: UserInput) -> bool:
        ...
        self._pending_steer_inputs.append(message)
        return True

    def peek_pending_steer_inputs(self) -> list[UserMessage]:
        return [UserMessage.from_value(item) for item in self._pending_steer_inputs]

    def ack_pending_steer_inputs(self, count: int) -> None:
        if count <= 0:
            return
        del self._pending_steer_inputs[:count]
```

```python
# agiwo/agent/prompt.py
def apply_steering_inputs(
    messages: list[dict[str, Any]],
    steer_inputs: list[UserMessage],
) -> list[dict[str, Any]]:
    for normalized in steer_inputs:
        if normalized.has_content():
            messages.append(
                {
                    "role": "user",
                    "content": normalized.to_message_content(),
                }
            )
    return messages
```

```python
# agiwo/agent/run_loop.py
pending_steer = self.context.session_runtime.peek_pending_steer_inputs()
request_messages = apply_steering_inputs(
    self.context.snapshot_messages(),
    pending_steer,
)
modified = await self.context.hooks.before_llm_call(request_messages, self.context)
if modified is not None:
    request_messages = modified
if request_messages != self.context.snapshot_messages():
    rebuilt_entries = await self.writer.rebuild_messages(
        reason="before_llm",
        messages=request_messages,
    )
    await self._project_entries(rebuilt_entries)
    self.context.session_runtime.ack_pending_steer_inputs(len(pending_steer))
```

- [ ] **Step 4: Run the focused tests and parity check**

Run: `uv run pytest tests/agent/test_run_loop_contracts.py tests/agent/test_run_engine.py::test_before_llm_failure_does_not_lose_accepted_steer_input tests/agent/test_run_log_replay_parity.py -v`
Expected: PASS with steer inputs preserved until the `MessagesRebuilt(reason="before_llm")` commit succeeds and live/replay parity unchanged.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/runtime/session.py agiwo/agent/run_loop.py agiwo/agent/prompt.py tests/agent/test_run_loop_contracts.py tests/agent/test_run_engine.py tests/agent/test_run_log_replay_parity.py
git commit -m "refactor: make steer consumption lossless"
```

### Task 2: Move Remaining Bootstrap And Compaction State Behind The Writer

**Files:**
- Modify: `agiwo/agent/runtime/state_writer.py`
- Modify: `agiwo/agent/run_bootstrap.py`
- Modify: `agiwo/agent/run_loop.py`
- Modify: `agiwo/agent/compaction.py`
- Test: `tests/agent/test_state_tracking.py`
- Test: `tests/agent/test_run_engine.py`
- Test: `tests/agent/test_compact.py`

- [ ] **Step 1: Write failing tests for remaining writer ownership**

```python
# tests/agent/test_state_tracking.py
import pytest

from agiwo.agent.runtime.state_writer import RunStateWriter


@pytest.mark.asyncio
async def test_run_state_writer_owns_bootstrap_state_mutations() -> None:
    state = _make_state()
    writer = RunStateWriter(state)

    await writer.record_bootstrap_state(
        run_start_seq=7,
        tool_schemas=[{"type": "function", "function": {"name": "bash"}}],
        latest_compaction=None,
    )

    assert state.ledger.run_start_seq == 7
    assert state.ledger.tool_schemas == [{"type": "function", "function": {"name": "bash"}}]
    assert state.ledger.compaction.last_metadata is None
```

```python
# tests/agent/test_compact.py
from agiwo.agent.models.log import LLMCallCompleted, LLMCallStarted


@pytest.mark.asyncio
async def test_compaction_writes_canonical_llm_call_facts(tmp_path) -> None:
    ...
    result = await compact_if_needed(...)
    assert result.metadata is not None

    entries = await session_runtime.list_run_log_entries(run_id="run-1")
    assert any(isinstance(entry, LLMCallStarted) for entry in entries)
    assert any(isinstance(entry, LLMCallCompleted) for entry in entries)
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest tests/agent/test_state_tracking.py::test_run_state_writer_owns_bootstrap_state_mutations tests/agent/test_compact.py::test_compaction_writes_canonical_llm_call_facts -v`
Expected: FAIL because bootstrap still mutates state directly and compaction still commits internal steps without canonical LLM-call facts.

- [ ] **Step 3: Add explicit writer-owned bootstrap and internal-LLM methods**

```python
# agiwo/agent/runtime/state_writer.py
class RunStateWriter:
    async def record_bootstrap_state(
        self,
        *,
        run_start_seq: int,
        tool_schemas: list[dict[str, Any]] | None,
        latest_compaction: CompactMetadata | None,
    ) -> None:
        self._state.ledger.run_start_seq = run_start_seq
        self._state.ledger.tool_schemas = (
            copy.deepcopy(tool_schemas) if tool_schemas is not None else None
        )
        self._state.ledger.compaction.last_metadata = latest_compaction

    async def record_llm_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        step: StepView,
        llm: LLMCallContext,
    ) -> list[RunLogEntry]:
        started = build_llm_call_started_entry(...)
        completed = build_llm_call_completed_entry(...)
        return await self.append_entries([started, completed])

    async def set_compaction_failure_count(self, attempt: int) -> None:
        self._state.ledger.compaction.failure_count = attempt
```

```python
# agiwo/agent/run_bootstrap.py
user_step = await _build_user_step(context, user_input)
await writer.record_bootstrap_state(
    run_start_seq=user_step.sequence,
    tool_schemas=_build_tool_schemas(runtime),
    latest_compaction=latest_compact,
)
await writer.record_context_assembled(
    messages=assembled_messages,
    memory_count=len(memories),
)
```

```python
# agiwo/agent/compaction.py
request_entries = await writer.record_llm_call_started(
    messages=state.snapshot_messages(),
    tools=None,
)
await state.session_runtime.project_run_log_entries(request_entries, ...)
step, llm_context = await stream_assistant_step(...)
await commit_step(state, step, llm=llm_context, append_message=False)
completion_entries = await writer.record_llm_call_completed(step=step, llm=llm_context)
await state.session_runtime.project_run_log_entries(completion_entries, ...)
```

- [ ] **Step 4: Run the focused tests and engine coverage**

Run: `uv run pytest tests/agent/test_state_tracking.py tests/agent/test_run_engine.py tests/agent/test_compact.py -v`
Expected: PASS with bootstrap state owned by the writer, compaction failure count updated by writer methods, and compaction internal LLM turns represented by canonical LLM facts.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/runtime/state_writer.py agiwo/agent/run_bootstrap.py agiwo/agent/run_loop.py agiwo/agent/compaction.py tests/agent/test_state_tracking.py tests/agent/test_run_engine.py tests/agent/test_compact.py
git commit -m "refactor: route bootstrap and compaction state through writer"
```

### Task 3: Make Termination Summary A Canonical Internal LLM Turn

**Files:**
- Modify: `agiwo/agent/termination/summarizer.py`
- Modify: `agiwo/agent/runtime/state_writer.py`
- Test: `tests/agent/test_termination.py`

- [ ] **Step 1: Write the failing test for termination-summary LLM facts**

```python
# tests/agent/test_termination.py
import pytest

from agiwo.agent import Agent, AgentConfig, TerminationReason
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.log import LLMCallCompleted, LLMCallStarted


@pytest.mark.asyncio
async def test_termination_summary_writes_canonical_llm_call_facts() -> None:
    agent = Agent(
        AgentConfig(
            name="termination-summary",
            description="termination-summary",
            options=AgentOptions(max_steps=0, enable_termination_summary=True),
        ),
        model=_FixedResponseModel(response="summary"),
    )

    result = await agent.run("hello", session_id="termination-summary-session")

    assert result.termination_reason == TerminationReason.MAX_STEPS
    entries = await agent.run_log_storage.list_entries(
        session_id="termination-summary-session"
    )
    assert any(
        isinstance(entry, LLMCallStarted) and entry.run_id == result.run_id
        for entry in entries
    )
    assert any(
        isinstance(entry, LLMCallCompleted) and entry.run_id == result.run_id
        for entry in entries
    )
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `uv run pytest tests/agent/test_termination.py::test_termination_summary_writes_canonical_llm_call_facts -v`
Expected: FAIL because termination summary currently commits only steps and no canonical LLM-call facts.

- [ ] **Step 3: Route termination-summary request and completion through the writer**

```python
# agiwo/agent/termination/summarizer.py
writer = RunStateWriter(state)
summary_user_step = StepView.user(
    state,
    sequence=await state.session_runtime.allocate_sequence(),
    content=user_prompt,
    name="summary_request",
)
await commit_step(state, summary_user_step, append_message=True)
started_entries = await writer.record_llm_call_started(
    messages=state.snapshot_messages(),
    tools=None,
)
await state.session_runtime.project_run_log_entries(started_entries, ...)
step, llm_context = await stream_assistant_step(
    model,
    state,
    abort_signal,
    messages=state.snapshot_messages(),
    use_state_tools=False,
)
step.name = "summary"
await commit_step(state, step, llm=llm_context, append_message=False)
completed_entries = await writer.record_llm_call_completed(step=step, llm=llm_context)
await state.session_runtime.project_run_log_entries(completed_entries, ...)
```

- [ ] **Step 4: Run the focused termination tests**

Run: `uv run pytest tests/agent/test_termination.py -v`
Expected: PASS with termination summary preserving existing best-effort behavior while also writing canonical internal LLM facts.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/termination/summarizer.py agiwo/agent/runtime/state_writer.py tests/agent/test_termination.py
git commit -m "refactor: make termination summary a canonical llm turn"
```

### Task 4: Delete Legacy Trace Callbacks And Persist Trace From Entry Batches

**Files:**
- Modify: `agiwo/agent/trace_writer.py`
- Modify: `agiwo/agent/runtime/session.py`
- Test: `tests/observability/test_collector.py`
- Test: `tests/agent/test_run_log_replay_parity.py`
- Test: `console/tests/test_runtime_replay_consistency.py`

- [ ] **Step 1: Write failing tests for entry-only trace behavior, bounded cache, and save-on-entry**

```python
# tests/observability/test_collector.py
from collections import OrderedDict

import pytest

from agiwo.agent.models.log import AssistantStepCommitted, LLMCallCompleted, LLMCallStarted, RunFinished, RunStarted
from agiwo.agent.trace_writer import AgentTraceCollector
from agiwo.observability.memory_store import InMemoryTraceStorage


@pytest.mark.asyncio
async def test_collector_persists_trace_after_run_log_entry_batch() -> None:
    store = InMemoryTraceStorage()
    collector = AgentTraceCollector(store=store)
    collector.start(trace_id="trace-1", agent_id="agent-1", session_id="sess-1")

    await collector.on_run_log_entries(
        [
            RunStarted(sequence=1, session_id="sess-1", run_id="run-1", agent_id="agent-1", user_input="hello"),
            RunFinished(sequence=2, session_id="sess-1", run_id="run-1", agent_id="agent-1", response="done"),
        ]
    )

    saved = await store.get_trace("trace-1")
    assert saved is not None
    assert saved.final_output == "done"


def test_collector_no_longer_exposes_legacy_trace_callbacks() -> None:
    collector = AgentTraceCollector()
    assert not hasattr(collector, "on_run_started")
    assert not hasattr(collector, "on_step")
    assert not hasattr(collector, "on_run_completed")
    assert not hasattr(collector, "on_run_failed")
```

```python
# tests/observability/test_collector.py
@pytest.mark.asyncio
async def test_collector_bounds_committed_caches() -> None:
    collector = AgentTraceCollector()
    collector.start(trace_id="trace-2", agent_id="agent-1", session_id="sess-1")

    entries = [
        RunStarted(sequence=1, session_id="sess-1", run_id="run-1", agent_id="agent-1", user_input="hello")
    ]
    entries.extend(
        AssistantStepCommitted(
            sequence=idx + 2,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            step_id=f"step-{idx}",
            role=MessageRole.ASSISTANT,
            content="tool call",
            tool_calls=[{"id": f"call-{idx}", "function": {"name": "bash", "arguments": "{}"}}],
        )
        for idx in range(collector._CACHE_MAX_SIZE + 5)
    )

    await collector.on_run_log_entries(entries)

    assert len(collector._assistant_committed_cache) == collector._CACHE_MAX_SIZE
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `uv run pytest tests/observability/test_collector.py::test_collector_persists_trace_after_run_log_entry_batch tests/observability/test_collector.py::test_collector_no_longer_exposes_legacy_trace_callbacks tests/observability/test_collector.py::test_collector_bounds_committed_caches -v`
Expected: FAIL because the legacy callback methods still exist, committed caches are unbounded, and entry-batch application does not persist the trace snapshot immediately.

- [ ] **Step 3: Delete legacy callbacks and make entry batches the only live trace path**

```python
# agiwo/agent/trace_writer.py
class AgentTraceCollector:
    _CACHE_MAX_SIZE = 10_000

    async def on_run_log_entries(self, entries: list[RunLogEntry]) -> None:
        trace = self._trace
        if trace is None:
            return
        for entry in entries:
            _apply_entry_to_trace(
                trace,
                entry,
                run_spans=self._run_spans,
                llm_started=self._llm_started,
                assistant_cache=self._assistant_committed_cache,
                preview_length=self.PREVIEW_LENGTH,
            )
            if isinstance(entry, LLMCallCompleted):
                self._llm_started.pop(entry.run_id, None)
            if isinstance(entry, RunFinished | RunFailedEntry):
                self._llm_started.pop(entry.run_id, None)
        while len(self._assistant_committed_cache) > self._CACHE_MAX_SIZE:
            self._assistant_committed_cache.popitem(last=False)
        await self._save_trace()
```

```python
# agiwo/agent/trace_writer.py
# delete:
# - on_run_started(...)
# - on_step(...)
# - on_run_completed(...)
# - on_run_failed(...)
```

- [ ] **Step 4: Run trace, parity, and console replay coverage**

Run: `uv run pytest tests/observability/test_collector.py tests/agent/test_run_log_replay_parity.py console/tests/test_runtime_replay_consistency.py -v`
Expected: PASS with trace built and persisted only from committed entries, live/replay parity preserved, and no production dependency on direct callback trace construction.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/trace_writer.py agiwo/agent/runtime/session.py tests/observability/test_collector.py tests/agent/test_run_log_replay_parity.py console/tests/test_runtime_replay_consistency.py
git commit -m "refactor: make trace entry-only and persist live projections"
```

### Task 5: Run Full Verification And Update Runtime Docs

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update runtime wording if implementation changed contract details**

```markdown
### Storage & Observability

- accepted steer input is staged and only consumed after the writer commits the resulting `MessagesRebuilt` fact
- internal canonical LLM turns such as `compaction` and termination summary also emit `LLMCallStarted` / `LLMCallCompleted`
- `AgentTraceCollector` no longer exposes legacy direct callbacks; live and replay trace both consume committed `RunLog` entries only
```

- [ ] **Step 2: Run the full lint and affected test suites**

Run: `uv run python scripts/lint.py ci`
Expected: PASS with formatting, lint, import-linter, and repo guard all green.

Run: `uv run pytest tests/agent/test_run_loop_contracts.py tests/agent/test_run_engine.py tests/agent/test_state_tracking.py tests/agent/test_compact.py tests/agent/test_termination.py tests/observability/test_collector.py tests/agent/test_run_log_replay_parity.py tests/scheduler/test_runtime_facts.py -v`
Expected: PASS with steer, writer, internal-LLM, trace, and replay coverage green.

Run: `uv run pytest console/tests/test_runtime_replay_consistency.py -v`
Expected: PASS with console replay/query semantics still matching committed `RunLog`.

Run: `uv run python scripts/check.py console-tests`
Expected: PASS for the console backend suite.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: align runtime contract with final convergence"
```

## Self-Review

Spec coverage:
- Lossless steer consumption is covered by Task 1.
- Remaining writer bypass removal is covered by Task 2.
- Internal canonical LLM turns are covered by Tasks 2 and 3.
- Entry-only trace construction and live trace persistence are covered by Task 4.
- Final contract wording and full verification are covered by Task 5.

Placeholder scan:
- No `TODO`, `TBD`, or deferred implementation placeholders remain.

Type consistency:
- The plan consistently uses `peek_pending_steer_inputs`, `ack_pending_steer_inputs`, `record_bootstrap_state`, and canonical `LLMCallStarted` / `LLMCallCompleted` naming across tasks.
