# Goal-Directed Review Cleanup & Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the review/step-back migration by removing the remaining legacy review surface, making review metadata persistently hidden from future prompt rebuilds, and aligning runtime, scheduler, console, and docs with the approved design.

**Architecture:** Add one append-only hidden-from-context run-log fact so review metadata remains committed and observable while never re-entering later prompt assembly. Then simplify the review runtime around one milestone state machine and one structured cleanup outcome, wire the missing review hooks, remove dead scheduler/tool surface, and finally clean every remaining legacy review term from current and archived documentation.

**Tech Stack:** Python 3.10+, dataclasses, asyncio, Pydantic, SQLite/in-memory run-log replay, React 19, TypeScript, Vitest, pytest

**Spec:** `docs/superpowers/specs/2026-04-24-goal-directed-review-and-stepback-design.md`

---

## File Map

- `agiwo/agent/models/log.py`: Canonical run-log kinds and entry dataclasses.
- `agiwo/agent/runtime/state_writer.py`: Only write path for new review-hidden facts and step-back facts.
- `agiwo/agent/storage/serialization.py`: Run-log round-trip plus step-view replay filtering.
- `agiwo/agent/storage/base.py`: In-memory replay API and the public `list_step_views(...)` contract.
- `agiwo/agent/storage/sqlite.py`: SQLite replay query path that must honor hidden-from-context facts.
- `agiwo/agent/run_bootstrap.py`: Prompt rebuild entrypoint; must explicitly exclude hidden review metadata.
- `agiwo/agent/models/review.py`: Run-scoped milestone and checkpoint state.
- `agiwo/agent/review/goal_manager.py`: Milestone declaration/update logic; remove the thin OO wrapper.
- `agiwo/agent/review/review_enforcer.py`: Trigger detection and `<system-review>` notice building.
- `agiwo/agent/review/step_back_executor.py`: Step-back payload generation; should return structured content updates.
- `agiwo/agent/review/__init__.py`: `ReviewBatch` public runtime façade used by `run_tool_batch.py`.
- `agiwo/agent/run_tool_batch.py`: Commits tool steps, applies review cleanup outcome, and emits runtime facts.
- `agiwo/agent/run_loop.py`: Needs to pass the assistant step id into the tool batch so hidden review assistant steps can be persisted.
- `agiwo/agent/hooks.py`: Missing `before_review(...)` and `after_step_back(...)` dispatch helpers.
- `agiwo/scheduler/runtime_tools.py`: Review tool schemas and runtime-tool constructors.
- `agiwo/scheduler/engine.py`: Runtime-tool injection point; remove unused scheduler control wiring for review tools.
- `console/web/src/lib/api.ts`: Frontend payload types still expose legacy review fields.
- `console/web/src/components/agent-form.tsx`: UI still renders and submits removed review config.
- `console/web/src/components/session-detail/session-observability-panel.tsx`: Frontend still checks the wrong event kind instead of `step_back`.
- `AGENTS.md` and `docs/**`: Current and archived docs still carry old legacy-review terminology.

### Task 1: Persist Hidden Review Metadata And Keep It Out Of Prompt Rebuilds

**Files:**
- Modify: `agiwo/agent/models/log.py`
- Modify: `agiwo/agent/runtime/state_writer.py`
- Modify: `agiwo/agent/storage/serialization.py`
- Modify: `agiwo/agent/storage/base.py`
- Modify: `agiwo/agent/storage/sqlite.py`
- Modify: `agiwo/agent/run_bootstrap.py`
- Test: `tests/agent/test_storage_serialization.py`
- Test: `tests/agent/test_run_log_replay_parity.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agent/test_storage_serialization.py
from agiwo.agent.models.log import (
    ContextStepsHidden,
    ToolStepCommitted,
    UserStepCommitted,
)
from agiwo.agent.models.step import MessageRole
from agiwo.agent.storage.serialization import (
    build_step_views_from_entries,
    deserialize_run_log_entry_from_storage,
    serialize_run_log_entry_for_storage,
)


def test_build_step_views_can_exclude_hidden_review_steps_from_context() -> None:
    entries = [
        UserStepCommitted(
            sequence=1,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            step_id="step-user",
            role=MessageRole.USER,
            content="hello",
            user_input="hello",
        ),
        ToolStepCommitted(
            sequence=2,
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
            sequence=3,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            step_ids=["step-review-result"],
            reason="review_metadata",
        ),
    ]

    visible = build_step_views_from_entries(entries, include_hidden_from_context=False)
    all_steps = build_step_views_from_entries(entries, include_hidden_from_context=True)

    assert [step.id for step in visible] == ["step-user"]
    assert [step.id for step in all_steps] == ["step-user", "step-review-result"]


def test_hidden_review_metadata_round_trips_through_storage() -> None:
    entry = ContextStepsHidden(
        sequence=9,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        step_ids=["step-review-call", "step-review-result"],
        reason="review_metadata",
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert isinstance(restored, ContextStepsHidden)
    assert restored.step_ids == ["step-review-call", "step-review-result"]
    assert restored.reason == "review_metadata"
```

```python
# tests/agent/test_run_log_replay_parity.py
from agiwo.agent.models.log import ContextStepsHidden
from agiwo.agent.models.stream import stream_items_from_entries


def test_hidden_context_fact_does_not_emit_public_stream_events() -> None:
    entries = [
        ContextStepsHidden(
            sequence=1,
            session_id="sess-1",
            run_id="run-1",
            agent_id="agent-1",
            step_ids=["step-review-call"],
            reason="review_metadata",
        )
    ]

    assert stream_items_from_entries(entries) == []
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

```bash
uv run pytest tests/agent/test_storage_serialization.py tests/agent/test_run_log_replay_parity.py -v
```

Expected: FAIL because `ContextStepsHidden` does not exist and the step replay API has no hidden-from-context filter.

- [ ] **Step 3: Add the new run-log fact and writer helper**

```python
# agiwo/agent/models/log.py
class RunLogEntryKind(str, Enum):
    STEP_BACK_APPLIED = "step_back_applied"
    STEP_CONDENSED_CONTENT_UPDATED = "step_condensed_content_updated"
    CONTEXT_STEPS_HIDDEN = "context_steps_hidden"
    TERMINATION_DECIDED = "termination_decided"
    HOOK_FAILED = "hook_failed"


@dataclass(frozen=True, kw_only=True)
class ContextStepsHidden(RunLogEntry):
    step_ids: list[str]
    reason: Literal["review_metadata"]
    kind: RunLogEntryKind = field(
        init=False,
        default=RunLogEntryKind.CONTEXT_STEPS_HIDDEN,
    )
```

```python
# agiwo/agent/runtime/state_writer.py
async def record_context_steps_hidden(
    self,
    *,
    step_ids: list[str],
    reason: str = "review_metadata",
) -> list[object]:
    return await self.append_entries(
        [
            ContextStepsHidden(
                sequence=await self._state.session_runtime.allocate_sequence(),
                session_id=self._state.session_id,
                run_id=self._state.run_id,
                agent_id=self._state.agent_id,
                step_ids=list(step_ids),
                reason=reason,
            )
        ]
    )
```

- [ ] **Step 4: Teach replay and bootstrap to honor hidden review metadata**

```python
# agiwo/agent/storage/serialization.py
def _build_hidden_step_ids(
    entries: list[RunLogEntry],
    *,
    include_hidden_from_context: bool,
) -> set[str]:
    if include_hidden_from_context:
        return set()
    hidden_step_ids: set[str] = set()
    for entry in entries:
        if isinstance(entry, ContextStepsHidden):
            hidden_step_ids.update(entry.step_ids)
    return hidden_step_ids


def build_step_views_from_entries(
    entries: list[RunLogEntry],
    *,
    include_rolled_back: bool = False,
    include_hidden_from_context: bool = True,
) -> list[StepView]:
    condensation_by_step_id = _build_condensation_map(entries)
    hidden_sequences = _build_hidden_sequences(
        entries,
        include_rolled_back=include_rolled_back,
    )
    hidden_step_ids = _build_hidden_step_ids(
        entries,
        include_hidden_from_context=include_hidden_from_context,
    )
    step_views: list[StepView] = []
    for entry in _iter_visible_committed_steps(
        entries,
        hidden_sequences=hidden_sequences,
    ):
        if entry.step_id in hidden_step_ids:
            continue
        step_view = build_step_view_from_entry(entry)
        condensed_content = condensation_by_step_id.get(step_view.id)
        if condensed_content is not None:
            step_view.condensed_content = condensed_content
        step_views.append(step_view)
    return step_views
```

```python
# agiwo/agent/storage/base.py
async def list_step_views(
    self,
    *,
    session_id: str,
    start_seq: int | None = None,
    end_seq: int | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    include_rolled_back: bool = False,
    include_hidden_from_context: bool = True,
    limit: int = 1000,
    order: Literal["asc", "desc"] = "asc",
) -> list[StepView]:
    entries = await self.list_entries(
        session_id=session_id,
        run_id=run_id,
        agent_id=agent_id,
        limit=100_000,
    )
    step_views = build_step_views_from_entries(
        entries,
        include_rolled_back=include_rolled_back,
        include_hidden_from_context=include_hidden_from_context,
    )
```

```python
# agiwo/agent/run_bootstrap.py
return await context.session_runtime.run_log_storage.list_step_views(
    session_id=context.session_id,
    agent_id=context.agent_id,
    start_seq=compact_start_seq if compact_start_seq > 0 else None,
    include_hidden_from_context=False,
)
```

Do not add a new stream event for `ContextStepsHidden`. It is an internal replay-control fact, not part of the public stream protocol.

- [ ] **Step 5: Re-run the focused tests**

```bash
uv run pytest tests/agent/test_storage_serialization.py tests/agent/test_run_log_replay_parity.py -v
```

Expected: PASS, with hidden review steps excluded only when `include_hidden_from_context=False` and no public stream event emitted for the new fact.

- [ ] **Step 6: Commit**

```bash
git add agiwo/agent/models/log.py agiwo/agent/runtime/state_writer.py agiwo/agent/storage/serialization.py agiwo/agent/storage/base.py agiwo/agent/storage/sqlite.py agiwo/agent/run_bootstrap.py tests/agent/test_storage_serialization.py tests/agent/test_run_log_replay_parity.py
git commit -m "feat: hide review metadata from prompt replay"
```

### Task 2: Simplify The Review State Machine And Unify Cleanup Outcomes

**Files:**
- Modify: `agiwo/agent/models/review.py`
- Modify: `agiwo/agent/review/goal_manager.py`
- Modify: `agiwo/agent/review/review_enforcer.py`
- Modify: `agiwo/agent/review/step_back_executor.py`
- Modify: `agiwo/agent/review/__init__.py`
- Modify: `agiwo/agent/run_tool_batch.py`
- Modify: `agiwo/agent/run_loop.py`
- Test: `tests/agent/test_review_batch.py`
- Test: `tests/agent/test_step_back_executor.py`

- [ ] **Step 1: Write the failing tests for the new milestone and cleanup semantics**

```python
# tests/agent/test_review_batch.py
import pytest

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.review import Milestone, ReviewCheckpoint, ReviewState
from agiwo.agent.models.run import RunLedger
from agiwo.agent.review import ReviewBatch, declare_milestones
from agiwo.agent.storage.base import InMemoryRunLogStorage

# These tests reuse the existing FakeTool and FakeToolResult helpers
# already defined at the top of tests/agent/test_review_batch.py.


def test_initial_milestone_declaration_does_not_schedule_review() -> None:
    state = ReviewState()
    declare_milestones(
        state,
        [Milestone(id="understand", description="Read auth flow")],
        current_seq=3,
    )
    assert state.pending_review_reason is None


def test_milestone_switch_marks_pending_review_for_next_non_review_tool_result() -> None:
    state = ReviewState(
        milestones=[Milestone(id="understand", description="Read auth flow", status="active")]
    )
    declare_milestones(
        state,
        [
            Milestone(id="understand", description="Read auth flow", status="completed"),
            Milestone(id="fix", description="Patch timeout path", status="active"),
        ],
        current_seq=6,
    )
    assert state.pending_review_reason == "milestone_switch"


@pytest.mark.asyncio
async def test_finalize_aligned_review_hides_review_steps_and_updates_checkpoint() -> None:
    config = AgentOptions(enable_goal_directed_review=True)
    ledger = RunLedger()
    ledger.review.milestones = [
        Milestone(id="fix", description="Patch timeout path", status="active")
    ]
    ledger.messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_review",
                    "type": "function",
                    "function": {
                        "name": "review_trajectory",
                        "arguments": '{"aligned": true}',
                    },
                }
            ],
            "_sequence": 8,
        },
        {
            "role": "tool",
            "tool_call_id": "tc_review",
            "content": "Trajectory review: aligned=True.",
            "_sequence": 9,
        },
    ]
    batch = ReviewBatch(
        config,
        ledger,
        {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        },
    )
    batch.process_result(
        FakeToolResult(
            "review_trajectory",
            "Trajectory review: aligned=True.",
            tool_call_id="tc_review",
            output={"aligned": True, "experience": ""},
        ),
        current_seq=9,
        assistant_step_id="step-review-call",
        tool_step_id="step-review-result",
    )
    outcome = await batch.finalize(
        storage=InMemoryRunLogStorage(),
        session_id="s1",
        run_id="r1",
        agent_id="a1",
    )
    assert outcome.hidden_step_ids == ["step-review-call", "step-review-result"]
    assert outcome.step_back_applied is None
    assert ledger.review.latest_checkpoint is not None
    assert ledger.review.latest_checkpoint.seq == 9


@pytest.mark.asyncio
async def test_finalize_misaligned_review_returns_structured_content_updates() -> None:
    config = AgentOptions(enable_goal_directed_review=True)
    ledger = RunLedger()
    ledger.review.milestones = [
        Milestone(id="fix", description="Patch timeout path", status="active")
    ]
    ledger.review.latest_checkpoint = ReviewCheckpoint(seq=2, milestone_id="fix")
    ledger.messages = [
        {
            "role": "tool",
            "tool_call_id": "tc_search",
            "content": "Verbose JWT search result",
            "_sequence": 5,
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc_review",
                    "type": "function",
                    "function": {
                        "name": "review_trajectory",
                        "arguments": '{"aligned": false, "experience": "JWT search was off-track"}',
                    },
                }
            ],
            "_sequence": 8,
        },
        {
            "role": "tool",
            "tool_call_id": "tc_review",
            "content": "Trajectory review: aligned=False.",
            "_sequence": 9,
        },
    ]
    batch = ReviewBatch(
        config,
        ledger,
        {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        },
    )
    batch.register_step("tc_search", "step-search", 5)
    batch.process_result(
        FakeToolResult(
            "review_trajectory",
            "Trajectory review: aligned=False. JWT search was off-track",
            tool_call_id="tc_review",
            output={"aligned": False, "experience": "JWT search was off-track"},
        ),
        current_seq=9,
        assistant_step_id="step-review-call",
        tool_step_id="step-review-result",
    )
    outcome = await batch.finalize(
        storage=InMemoryRunLogStorage(),
        session_id="s1",
        run_id="r1",
        agent_id="a1",
    )
    assert outcome.hidden_step_ids == ["step-review-call", "step-review-result"]
    assert outcome.content_updates[0].tool_call_id == "tc_search"
    assert outcome.content_updates[0].content.startswith("[EXPERIENCE]")
    assert outcome.step_back_applied is not None
```

```python
# tests/agent/test_step_back_executor.py
from unittest.mock import AsyncMock

import pytest

from agiwo.agent.review.step_back_executor import ContentUpdate, execute_step_back

@pytest.mark.asyncio
async def test_step_back_returns_updates_instead_of_rebuilt_messages() -> None:
    messages = _make_messages()
    storage = AsyncMock()
    storage.append_step_condensed_content = AsyncMock(return_value=True)

    outcome = await execute_step_back(
        messages=messages,
        checkpoint_seq=3,
        experience="JWT search was off-track.",
        step_lookup={"tc_2": {"id": "step_2", "sequence": 5}},
        storage=storage,
        session_id="s1",
        run_id="r1",
        agent_id="a1",
    )

    assert outcome.content_updates == [
        ContentUpdate(
            step_id="step_2",
            tool_call_id="tc_2",
            content="[EXPERIENCE] JWT search was off-track.",
        )
    ]
    assert outcome.affected_count == 1
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

```bash
uv run pytest tests/agent/test_review_batch.py tests/agent/test_step_back_executor.py -v
```

Expected: FAIL because `ReviewState` still uses `last_checkpoint_seq` and `is_review_pending`, `GoalManager` still exists, and step-back still returns a rebuilt message list.

- [ ] **Step 3: Replace the old review state fields and milestone helper surface**

```python
# agiwo/agent/models/review.py
@dataclass
class ReviewState:
    milestones: list[Milestone] = field(default_factory=list)
    last_review_seq: int = 0
    latest_checkpoint: ReviewCheckpoint | None = None
    consecutive_errors: int = 0
    pending_review_reason: Literal["milestone_switch"] | None = None
```

```python
# agiwo/agent/review/goal_manager.py
def declare_milestones(
    state: ReviewState,
    milestones: list[Milestone],
    *,
    current_seq: int = 0,
) -> list[str]:
    previous_active_id = _active_milestone_id(state.milestones)
    existing_by_id = {item.id: item for item in state.milestones}
    merged: list[Milestone] = []
    for milestone in milestones:
        existing = existing_by_id.get(milestone.id)
        if existing is not None:
            milestone.declared_at_seq = existing.declared_at_seq
            milestone.completed_at_seq = existing.completed_at_seq
        else:
            milestone.declared_at_seq = current_seq
        merged.append(milestone)
    state.milestones = merged
    if (
        previous_active_id is None
        and state.milestones
        and not any(item.status == "active" for item in state.milestones)
    ):
        for item in state.milestones:
            if item.status == "pending":
                item.status = "active"
                break
    current_active_id = _active_milestone_id(state.milestones)
    if previous_active_id is not None and current_active_id != previous_active_id:
        state.pending_review_reason = "milestone_switch"
    return [item.id for item in milestones]
```

Delete the `GoalManager` class entirely. All call sites should use the module-level helpers directly.

- [ ] **Step 4: Return one structured cleanup outcome and apply it through `run_tool_batch.py`**

```python
# agiwo/agent/review/step_back_executor.py
@dataclass(frozen=True)
class ContentUpdate:
    step_id: str
    tool_call_id: str
    content: str


@dataclass(frozen=True)
class StepBackAppliedPayload:
    affected_count: int
    checkpoint_seq: int
    experience: str


@dataclass
class ReviewCleanupOutcome:
    applied: bool = False
    review_tool_call_id: str | None = None
    hidden_step_ids: list[str] = field(default_factory=list)
    content_updates: list[ContentUpdate] = field(default_factory=list)
    step_back_applied: StepBackAppliedPayload | None = None
```

```python
# agiwo/agent/review/__init__.py
checkpoint_seq = (
    self._ledger.review.latest_checkpoint.seq
    if self._ledger.review.latest_checkpoint is not None
    else 0
)

if aligned is True:
    self._ledger.review.latest_checkpoint = ReviewCheckpoint(
        seq=current_seq,
        milestone_id=active_milestone.id if active_milestone is not None else "",
    )
    self._ledger.review.last_review_seq = current_seq
    self._ledger.review.pending_review_reason = None
    return ReviewCleanupOutcome(
        applied=True,
        review_tool_call_id=result.tool_call_id or None,
        hidden_step_ids=[assistant_step_id, tool_step_id],
    )
```

```python
# agiwo/agent/run_loop.py
return await execute_tool_batch_cycle(
    context=self.context,
    runtime=self.runtime,
    tool_calls=tool_calls,
    assistant_step_id=step.id,
    set_termination_reason=_set_tool_termination,
    commit_step=self._commit_step,
)
```

```python
# agiwo/agent/run_tool_batch.py
if outcome.hidden_step_ids:
    await writer.record_context_steps_hidden(step_ids=outcome.hidden_step_ids)

for update in outcome.content_updates:
    _replace_tool_message_content(
        context.ledger.messages,
        tool_call_id=update.tool_call_id,
        content=update.content,
    )

_remove_review_tool_call(
    context.ledger.messages,
    review_tool_call_id=outcome.review_tool_call_id,
)

if outcome.step_back_applied is not None:
    step_back_entries = await writer.record_step_back_applied(
        affected_count=outcome.step_back_applied.affected_count,
        checkpoint_seq=outcome.step_back_applied.checkpoint_seq,
        experience=outcome.step_back_applied.experience,
    )
    await context.session_runtime.project_run_log_entries(
        step_back_entries,
        run_id=context.run_id,
        agent_id=context.agent_id,
        parent_run_id=context.parent_run_id,
        depth=context.depth,
    )
    await context.hooks.after_step_back(
        outcome=outcome.step_back_applied,
        context=context,
    )
```

Add two private helpers in `agiwo/agent/run_tool_batch.py` as part of this step:

```python
def _replace_tool_message_content(
    messages: list[dict[str, Any]],
    *,
    tool_call_id: str,
    content: str,
) -> None:
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id") == tool_call_id:
            message["content"] = content


def _remove_review_tool_call(
    messages: list[dict[str, Any]],
    *,
    review_tool_call_id: str | None,
) -> None:
    if not review_tool_call_id:
        return
    kept_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "tool":
            if message.get("tool_call_id") != review_tool_call_id:
                kept_messages.append(message)
            continue
        if message.get("role") != "assistant":
            kept_messages.append(message)
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            kept_messages.append(message)
            continue
        remaining_tool_calls = [
            tool_call
            for tool_call in tool_calls
            if tool_call.get("id") != review_tool_call_id
        ]
        message["tool_calls"] = remaining_tool_calls
        content = message.get("content")
        if remaining_tool_calls or (isinstance(content, str) and content.strip()):
            kept_messages.append(message)
    messages[:] = kept_messages
```

The important behavior change is this: aligned and misaligned reviews both go through the same cleanup contract, and only the hidden fact controls future prompt replay.

- [ ] **Step 5: Re-run the focused tests**

```bash
uv run pytest tests/agent/test_review_batch.py tests/agent/test_step_back_executor.py -v
```

Expected: PASS, with aligned reviews producing only hidden metadata cleanup and misaligned reviews producing hidden cleanup plus structured step-back updates.

- [ ] **Step 6: Commit**

```bash
git add agiwo/agent/models/review.py agiwo/agent/review/goal_manager.py agiwo/agent/review/review_enforcer.py agiwo/agent/review/step_back_executor.py agiwo/agent/review/__init__.py agiwo/agent/run_tool_batch.py agiwo/agent/run_loop.py tests/agent/test_review_batch.py tests/agent/test_step_back_executor.py
git commit -m "refactor: unify review cleanup and checkpoint state"
```

### Task 3: Wire The Missing Review Hooks And Trim The Scheduler Review Tool Surface

**Files:**
- Modify: `agiwo/agent/hooks.py`
- Modify: `agiwo/agent/review/review_enforcer.py`
- Modify: `agiwo/agent/run_tool_batch.py`
- Modify: `agiwo/scheduler/runtime_tools.py`
- Modify: `agiwo/scheduler/engine.py`
- Test: `tests/agent/test_hook_dispatcher.py`
- Test: `tests/scheduler/test_review_tools.py`

- [ ] **Step 1: Write the failing tests for hook dispatch and tool schema cleanup**

```python
# tests/agent/test_hook_dispatcher.py
@pytest.mark.asyncio
async def test_before_review_returns_review_advice() -> None:
    async def advise(payload: dict) -> dict:
        return {"review_advice": "Prefer auth.py before token helpers"}

    registry = HookRegistry(
        registrations=[
            decision_support(HookPhase.BEFORE_REVIEW, "advise", advise),
        ]
    )

    advice = await registry.before_review(
        trigger="milestone_switch",
        milestone_description="Patch timeout path",
        step_count=4,
        context=object(),
    )

    assert advice == "Prefer auth.py before token helpers"


@pytest.mark.asyncio
async def test_after_step_back_dispatches_payload() -> None:
    seen = {}

    async def observe_after_step_back(payload: dict) -> None:
        seen["payload"] = payload

    registry = HookRegistry(
        registrations=[
            observe(HookPhase.AFTER_STEP_BACK, "observe", observe_after_step_back),
        ]
    )

    await registry.after_step_back(outcome={"affected_count": 1}, context=object())
    assert seen["payload"]["outcome"] == {"affected_count": 1}
```

```python
# tests/scheduler/test_review_tools.py
def test_review_tools_do_not_require_scheduler_control() -> None:
    assert DeclareMilestonesTool().name == "declare_milestones"
    assert ReviewTrajectoryTool().name == "review_trajectory"


@pytest.mark.asyncio
async def test_declare_milestones_accepts_optional_status() -> None:
    tool = DeclareMilestonesTool()
    result = await tool.execute(
        parameters={
            "milestones": [
                {"id": "fix", "description": "Patch timeout path", "status": "active"}
            ]
        },
        context=ToolContext(session_id="s1", tool_call_id="tc_1"),
    )

    assert result.is_success
    assert result.output == {
        "milestones": [
            {"id": "fix", "description": "Patch timeout path", "status": "active"}
        ]
    }
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

```bash
uv run pytest tests/agent/test_hook_dispatcher.py tests/scheduler/test_review_tools.py -v
```

Expected: FAIL because `HookRegistry` exposes no `before_review(...)` or `after_step_back(...)`, the review notice cannot carry hook advice, and the scheduler review tools still require `SchedulerToolControl`.

- [ ] **Step 3: Add explicit review hook helpers and include hook advice in the review notice**

```python
# agiwo/agent/hooks.py
async def before_review(
    self,
    *,
    trigger: str,
    milestone_description: str | None,
    step_count: int,
    context: object | None = None,
) -> str | None:
    payload = await self._dispatch(
        HookPhase.BEFORE_REVIEW,
        {
            "trigger": trigger,
            "milestone_description": milestone_description,
            "step_count": step_count,
            "context": context,
            "review_advice": None,
        },
        allow_transform=True,
    )
    advice = payload.get("review_advice")
    return advice if isinstance(advice, str) else None


async def after_step_back(
    self,
    *,
    outcome: object,
    context: object | None = None,
) -> None:
    await self._dispatch(
        HookPhase.AFTER_STEP_BACK,
        {"outcome": outcome, "context": context},
        allow_transform=False,
    )
```

```python
# agiwo/agent/review/review_enforcer.py
def inject_system_review(
    content: str,
    milestone: Milestone | None,
    step_count: int,
    *,
    trigger_reason: str,
    review_advice: str | None = None,
) -> str:
    milestone_text = (
        f'Active milestone: "{milestone.description}"'
        if milestone is not None
        else "No active milestone declared. Consider using declare_milestones."
    )
    advice_text = f"Hook advice: {review_advice}\n" if review_advice else ""
    inner_text = (
        f"{milestone_text}\n\n"
        f"Trigger: {trigger_reason}\n"
        f"Steps since last review: {step_count}\n"
        f"{advice_text}\n"
        "Question: Do the last N steps meaningfully advance the active milestone?\n"
        "If not, use review_trajectory to:\n"
        "  1. Indicate misalignment (aligned=false)\n"
        "  2. Provide a concise experience summary of what was learned\n"
    )
    return f"{content}\n\n<system-review>\n{inner_text}</system-review>"
```

- [ ] **Step 4: Remove the unused scheduler control dependency from review tools**

```python
# agiwo/scheduler/runtime_tools.py
class DeclareMilestonesTool(BaseTool):
    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "milestones": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "active", "completed", "abandoned"],
                            },
                        },
                        "required": ["id", "description"],
                    },
                },
            },
            "required": ["milestones"],
        }
```

```python
# agiwo/scheduler/engine.py
self._scheduling_tools = (
    SpawnChildAgentTool(self._tool_control),
    ForkChildAgentTool(self._tool_control),
    SleepAndWaitTool(self._tool_control),
    QuerySpawnedAgentTool(self._tool_control),
    CancelAgentTool(self._tool_control),
    ListAgentsTool(self._tool_control),
    DeclareMilestonesTool(),
    ReviewTrajectoryTool(),
)
```

Keep the runtime-tool behavior pure: validation and schema only. These review tools should not depend on scheduler ports they never use.

- [ ] **Step 5: Re-run the focused tests**

```bash
uv run pytest tests/agent/test_hook_dispatcher.py tests/scheduler/test_review_tools.py -v
```

Expected: PASS, with hook advice wired into review injection and no dead scheduler-tool dependency left behind.

- [ ] **Step 6: Commit**

```bash
git add agiwo/agent/hooks.py agiwo/agent/review/review_enforcer.py agiwo/agent/run_tool_batch.py agiwo/scheduler/runtime_tools.py agiwo/scheduler/engine.py tests/agent/test_hook_dispatcher.py tests/scheduler/test_review_tools.py
git commit -m "refactor: wire review hooks and slim review tools"
```

### Task 4: Align The Console API Types, Agent Form, And Observability UI

**Files:**
- Modify: `console/web/src/lib/api.ts`
- Modify: `console/web/src/components/agent-form.tsx`
- Modify: `console/web/src/components/session-detail/session-observability-panel.tsx`
- Create: `console/web/src/components/agent-form.test.tsx`
- Modify: `console/web/src/components/session-detail/session-observability-panel.test.tsx`

- [ ] **Step 1: Write the failing frontend tests**

```tsx
// console/web/src/components/agent-form.test.tsx
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";

import { renderWithProviders } from "@/test/render";
import { AgentForm } from "./agent-form";

describe("AgentForm", () => {
  test("submits goal-directed review options instead of legacy step-back options", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn().mockResolvedValue(undefined);

    renderWithProviders(
      <AgentForm
        submitLabel="Create Agent"
        submitting={false}
        error={null}
        onSubmit={onSubmit}
      />,
    );

    await user.click(screen.getByLabelText("Enable Goal-Directed Review"));
    await user.clear(screen.getByLabelText("Review Step Interval"));
    await user.type(screen.getByLabelText("Review Step Interval"), "5");
    await user.click(screen.getByRole("button", { name: "Create Agent" }));

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          options: expect.objectContaining({
            enable_goal_directed_review: false,
            review_step_interval: 5,
            review_on_error: true,
          }),
        }),
      ),
    );
  });
});
```

```tsx
// console/web/src/components/session-detail/session-observability-panel.test.tsx
test("renders a step_back runtime decision", () => {
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
            created_at: "2026-04-25T12:01:00Z",
            summary: "2 results condensed, checkpoint at seq 5",
            details: {
              affected_count: 2,
              checkpoint_seq: 5,
            },
          },
        ],
      }}
    />,
  );

  expect(screen.getByText("2 results condensed, checkpoint at seq 5")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the focused frontend tests and confirm they fail**

```bash
cd console/web && npm test -- src/components/agent-form.test.tsx src/components/session-detail/session-observability-panel.test.tsx
```

Expected: FAIL because the frontend API types and form state still use `enable_goal_directed_review` and related threshold fields.

- [ ] **Step 3: Replace the legacy frontend payload types and form fields**

```ts
// console/web/src/lib/api.ts
export interface RuntimeDecisionEvent {
  kind: "termination" | "compaction" | "step_back" | "rollback" | string;
  sequence: number;
  run_id: string;
  agent_id: string;
  created_at: string;
  summary: string;
  details: Record<string, unknown>;
}

export interface AgentOptionsPayload {
  config_root: string;
  max_steps: number;
  run_timeout: number;
  max_input_tokens_per_call: number | null;
  max_run_cost: number | null;
  enable_termination_summary: boolean;
  termination_summary_prompt: string;
  relevant_memory_max_token: number;
  stream_cleanup_timeout: number;
  compact_prompt: string;
  enable_context_rollback: boolean;
  enable_goal_directed_review: boolean;
  review_step_interval: number;
  review_on_error: boolean;
}
```

```tsx
// console/web/src/components/agent-form.tsx
// Replace the legacy workflow fields in AgentFormState:
enableContextRollback: boolean;
enableGoalDirectedReview: boolean;
reviewStepInterval: number;
reviewOnError: boolean;

// Replace the legacy workflow defaults in DEFAULT_FORM_STATE:
enableContextRollback: true,
enableGoalDirectedReview: true,
reviewStepInterval: 8,
reviewOnError: true,
```

```tsx
// submit payload
options: {
  enable_context_rollback: form.enableContextRollback,
  enable_goal_directed_review: form.enableGoalDirectedReview,
  review_step_interval: form.reviewStepInterval,
  review_on_error: form.reviewOnError,
},
```

- [ ] **Step 4: Replace the UI wording and observability branch**

```tsx
// console/web/src/components/agent-form.tsx
<ToggleCard
  id={fieldId("enable-goal-directed-review")}
  label="Enable Goal-Directed Review"
  description="Inject milestone-aware system review checkpoints instead of carrying dead-end tool trajectories forward."
  checked={form.enableGoalDirectedReview}
  onChange={(checked) => setField("enableGoalDirectedReview", checked)}
/>
```

```tsx
// console/web/src/components/session-detail/session-observability-panel.tsx
function DecisionIcon({ kind }: { kind: RuntimeDecisionEvent["kind"] }) {
  if (kind === "termination") return <ShieldCheck className="h-4 w-4 text-red-300" />;
  if (kind === "compaction") return <Scissors className="h-4 w-4 text-cyan-300" />;
  if (kind === "step_back") return <Activity className="h-4 w-4 text-amber-300" />;
  return <GitBranch className="h-4 w-4 text-zinc-300" />;
}
```

The frontend should expose only the shipped runtime knobs. Do not keep legacy review fields as dead compatibility shims.

- [ ] **Step 5: Run the frontend checks**

```bash
cd console/web && npm run lint
cd console/web && npm test
cd console/web && npm run build
```

Expected: PASS, with the form submitting review fields and the observability panel recognizing `step_back`.

- [ ] **Step 6: Commit**

```bash
git add console/web/src/lib/api.ts console/web/src/components/agent-form.tsx console/web/src/components/agent-form.test.tsx console/web/src/components/session-detail/session-observability-panel.tsx console/web/src/components/session-detail/session-observability-panel.test.tsx
git commit -m "fix: align console review config and observability"
```

### Task 5: Remove The Remaining StepBack Terminology From Current And Archived Docs

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/architecture/scheduler-console-runtime-refactor.md`
- Modify: `docs/concepts/scheduler.md`
- Modify: `docs/guides/context-optimization.md`
- Modify: `docs/guides/hooks.md`
- Modify: `docs/guides/multi-agent.md`
- Modify: `docs/guides/storage.md`
- Modify: `docs/guides/streaming.md`
- Modify: `docs/superpowers/reviews/2026-04-22-agent-runtime-audit.md`
- Modify: `docs/superpowers/plans/2026-04-20-public-docs-site-overhaul.md`
- Modify: `docs/superpowers/plans/2026-04-21-agent-runtime-phase1.md`
- Modify: `docs/superpowers/plans/2026-04-22-agent-runtime-hardening-review.md`
- Modify: `docs/superpowers/plans/2026-04-22-agent-runtime-strict-convergence.md`
- Modify: `docs/superpowers/plans/2026-04-23-runtime-tool-surface-hardening.md`
- Modify: `docs/superpowers/plans/2026-04-24-goal-directed-review-and-stepback.md`
- Modify: `docs/superpowers/specs/2026-04-20-public-docs-site-overhaul-design.md`
- Modify: `docs/superpowers/specs/2026-04-21-agent-runtime-refactor-design.md`
- Modify: `docs/superpowers/specs/2026-04-22-agent-runtime-strict-convergence-design.md`
- Modify: `docs/superpowers/specs/2026-04-23-agent-runtime-mainline-cleanup-design.md`
- Modify: `docs/superpowers/specs/2026-04-24-goal-directed-review-and-stepback-design.md`
- Rename and modify: `docs/plans/archived/2026-04-04-context-rollback-and-step-back-design.md`
- Rename and modify: `docs/plans/archived/2026-04-05-step-back-batch-refactor-design.md`

- [ ] **Step 1: Capture the exact remaining files before editing**

```bash
rg -l "retrospect|Retrospect|RETROSPECT|enable_tool_retrospect|retrospect_|BEFORE_RETROSPECT|AFTER_RETROSPECT" AGENTS.md docs -S | sort
```

Expected: the command lists the files above. Use that exact list as the edit checklist.

- [ ] **Step 2: Rewrite current docs and `AGENTS.md` to the shipped terminology**

```markdown
- `retrospect` / `Retrospect` -> `review` / `Review` when the sentence describes the new goal-directed checkpoint flow.
- `RetrospectToolResultTool` -> `review_trajectory`
- `BEFORE_RETROSPECT` / `AFTER_RETROSPECT` -> `BEFORE_REVIEW` / `AFTER_STEP_BACK`
- `RetrospectApplied` -> `StepBackApplied`
- `agiwo/agent/retrospect/` -> `agiwo/agent/review/`
- `enable_tool_retrospect`, `retrospect_token_threshold`, `retrospect_round_interval`, `retrospect_accumulated_token_threshold` -> `enable_goal_directed_review`, `review_step_interval`, `review_on_error`
```

In `AGENTS.md`, update the architecture rows so they describe `agiwo/agent/review/` as the current context-optimization owner. Do not leave the old directory name in the stable-boundary docs.

- [ ] **Step 3: Rename archived docs so the filenames also stop carrying the old term**

```bash
mv docs/plans/archived/2026-04-04-context-rollback-and-step-back-design.md \
  docs/plans/archived/2026-04-04-context-rollback-and-step-back-predecessor-design.md

mv docs/plans/archived/2026-04-05-step-back-batch-refactor-design.md \
  docs/plans/archived/2026-04-05-review-batch-refactor-predecessor-design.md
```

Then rewrite the document titles and opening paragraphs so they read as historical predecessors of the current review/step-back design, instead of continuing to normalize the old vocabulary.

- [ ] **Step 4: Run the terminology sweep again**

```bash
rg -n "retrospect|Retrospect|RETROSPECT|enable_tool_retrospect|retrospect_|BEFORE_RETROSPECT|AFTER_RETROSPECT" AGENTS.md docs console/web/src agiwo tests console/tests -S
```

Expected: the remaining hits are only deliberate legacy literals that must stay for compatibility tests, such as the raw `"step_back_applied"` input in `tests/agent/test_storage_serialization.py`.

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md docs
git commit -m "docs: remove remaining legacy review terminology"
```

### Task 6: Run The Full Verification Sweep

**Files:**
- No file edits in this task. This is the final gate before implementation is considered complete.

- [ ] **Step 1: Run the agent and scheduler regression suite**

```bash
uv run pytest tests/agent/test_storage_serialization.py tests/agent/test_run_log_replay_parity.py tests/agent/test_review_batch.py tests/agent/test_step_back_executor.py tests/agent/test_hook_dispatcher.py tests/scheduler/test_review_tools.py -v
```

Expected: PASS, covering hidden-from-context replay, review cleanup, hook wiring, and scheduler review tool behavior.

- [ ] **Step 2: Run the repository lint gate**

```bash
uv run python scripts/lint.py ci
```

Expected: PASS.

- [ ] **Step 3: Run the Console backend test gate**

```bash
uv run python scripts/check.py console-tests
```

Expected: PASS.

- [ ] **Step 4: Re-run the Console web gate**

```bash
cd console/web && npm run lint && npm test && npm run build
```

Expected: PASS.

- [ ] **Step 5: Verify the worktree and commit stack are clean**

```bash
git status --short
git log --oneline -n 5
```

Expected: no unexpected modified files, and the recent commit stack matches the five task commits above.
