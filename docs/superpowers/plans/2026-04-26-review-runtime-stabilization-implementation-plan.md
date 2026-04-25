# Review Runtime Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement first-class review RunLog facts, replayable review state, and non-`review_trajectory` tool-count interval semantics.

**Architecture:** Add structured review facts to `agiwo.agent.models.log` and make SDK runtime write those facts whenever milestones, review triggers, checkpoints, and outcomes change. Replay helpers rebuild `ReviewState` from facts, while Console/trace observability consumes facts instead of parsing `declare_milestones`, `review_trajectory`, or `<system-review>` text as authoritative state.

**Tech Stack:** Python 3.10+, dataclasses, existing RunLog storage/serialization, pytest, FastAPI Console service tests, repository lint/import contracts.

---

## File Structure

- Modify `agiwo/agent/models/review.py`: add `PendingReviewNotice`, extend `ReviewState`, and add small serialization helpers for milestone board payloads.
- Modify `agiwo/agent/models/log.py`: add `RunLogEntryKind` values and dataclasses for `ReviewMilestonesUpdated`, `ReviewTriggerDecided`, `ReviewCheckpointRecorded`, and `ReviewOutcomeRecorded`.
- Modify `agiwo/agent/storage/serialization.py`: register the new RunLog dataclasses and restore nested milestone/review payloads during deserialization.
- Create `agiwo/agent/review/replay.py`: rebuild `ReviewState` from RunLog entries and compute current review count from committed tool facts.
- Modify `agiwo/agent/runtime/state_writer.py`: add writer methods for the new review facts.
- Modify `agiwo/agent/review/review_enforcer.py`: change interval trigger input from sequence delta to explicit review count.
- Modify `agiwo/agent/review/__init__.py`: make `ReviewBatch` count non-`review_trajectory` tools, write facts, preserve one-shot notice cleanup, and record review outcomes.
- Modify `agiwo/agent/run_tool_batch.py`: pass committed step ids/sequences to review fact writers and apply returned cleanup/step-back updates.
- Modify `agiwo/agent/run_bootstrap.py`: replay `ReviewState` from persisted facts for persistent sessions.
- Modify `agiwo/agent/trace_writer.py`: project review facts into runtime trace spans for Console trace views.
- Modify `console/server/services/runtime/runtime_observability.py`: build milestone board and review cycles from first-class review facts.
- Add tests in `tests/agent/test_review_run_log_facts.py`, `tests/agent/test_review_replay.py`, `tests/agent/test_review_state_writer.py`, `tests/agent/test_review_bootstrap.py`, `tests/agent/test_run_tool_batch.py`, and `console/tests/test_runtime_observability.py`.

## Task 1: Add Review RunLog Fact Models And Serialization

**Files:**
- Modify: `agiwo/agent/models/review.py`
- Modify: `agiwo/agent/models/log.py`
- Modify: `agiwo/agent/storage/serialization.py`
- Test: `tests/agent/test_review_run_log_facts.py`
- Test: `tests/agent/test_storage_serialization.py`

- [ ] **Step 1: Write failing round-trip tests for review facts**

Create `tests/agent/test_review_run_log_facts.py` with these tests:

```python
from agiwo.agent.models.log import (
    ReviewCheckpointRecorded,
    ReviewMilestonesUpdated,
    ReviewOutcomeRecorded,
    ReviewTriggerDecided,
    RunLogEntryKind,
)
from agiwo.agent.models.review import Milestone
from agiwo.agent.storage.serialization import (
    deserialize_run_log_entry_from_storage,
    serialize_run_log_entry_for_storage,
)


def test_review_milestones_updated_round_trips() -> None:
    entry = ReviewMilestonesUpdated(
        sequence=10,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        milestones=[
            Milestone(id="inspect", description="Inspect auth flow", status="active")
        ],
        active_milestone_id="inspect",
        source_tool_call_id="tc-milestones",
        source_step_id="step-milestones",
        reason="declared",
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert payload["kind"] == RunLogEntryKind.REVIEW_MILESTONES_UPDATED.value
    assert isinstance(restored, ReviewMilestonesUpdated)
    assert restored.milestones[0].id == "inspect"
    assert restored.active_milestone_id == "inspect"
    assert restored.reason == "declared"


def test_review_trigger_decided_round_trips() -> None:
    entry = ReviewTriggerDecided(
        sequence=11,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        trigger_reason="step_interval",
        active_milestone_id="inspect",
        review_count_since_checkpoint=8,
        trigger_tool_call_id="tc-search",
        trigger_tool_step_id="step-search",
        notice_step_id="step-search",
    )

    restored = deserialize_run_log_entry_from_storage(
        serialize_run_log_entry_for_storage(entry)
    )

    assert isinstance(restored, ReviewTriggerDecided)
    assert restored.trigger_reason == "step_interval"
    assert restored.review_count_since_checkpoint == 8


def test_review_checkpoint_recorded_round_trips() -> None:
    entry = ReviewCheckpointRecorded(
        sequence=12,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        checkpoint_seq=42,
        milestone_id="inspect",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
    )

    restored = deserialize_run_log_entry_from_storage(
        serialize_run_log_entry_for_storage(entry)
    )

    assert isinstance(restored, ReviewCheckpointRecorded)
    assert restored.checkpoint_seq == 42
    assert restored.milestone_id == "inspect"


def test_review_outcome_recorded_round_trips() -> None:
    entry = ReviewOutcomeRecorded(
        sequence=13,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        aligned=False,
        mode="step_back",
        experience="JWT search was not useful",
        active_milestone_id="inspect",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
        hidden_step_ids=["step-review-call", "step-review"],
        notice_cleaned_step_ids=["step-search"],
        condensed_step_ids=["step-search"],
    )

    restored = deserialize_run_log_entry_from_storage(
        serialize_run_log_entry_for_storage(entry)
    )

    assert isinstance(restored, ReviewOutcomeRecorded)
    assert restored.aligned is False
    assert restored.mode == "step_back"
    assert restored.condensed_step_ids == ["step-search"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_review_run_log_facts.py -q
```

Expected: FAIL with import errors for `ReviewMilestonesUpdated`, `ReviewTriggerDecided`, `ReviewCheckpointRecorded`, and `ReviewOutcomeRecorded`.

- [ ] **Step 3: Add review runtime models**

Update `agiwo/agent/models/review.py`:

```python
@dataclass
class PendingReviewNotice:
    """Outstanding one-shot review notice in prompt-visible context."""

    trigger_reason: str
    active_milestone_id: str | None
    review_count_since_checkpoint: int
    trigger_tool_call_id: str
    trigger_tool_step_id: str
    notice_step_id: str


@dataclass
class ReviewState:
    """Per-run review tracking state, stored on RunLedger."""

    milestones: list[Milestone] = field(default_factory=list)
    last_review_seq: int = 0
    latest_checkpoint: ReviewCheckpoint | None = None
    consecutive_errors: int = 0
    pending_review_reason: Literal["milestone_switch"] | None = None
    review_count_since_checkpoint: int = 0
    pending_review_notice: PendingReviewNotice | None = None
```

Keep `last_review_seq` for the staged implementation, but do not use it for new interval decisions after Task 4.

- [ ] **Step 4: Add review RunLog entry dataclasses**

Update `agiwo/agent/models/log.py`:

```python
class RunLogEntryKind(str, Enum):
    # existing values...
    REVIEW_MILESTONES_UPDATED = "review_milestones_updated"
    REVIEW_TRIGGER_DECIDED = "review_trigger_decided"
    REVIEW_CHECKPOINT_RECORDED = "review_checkpoint_recorded"
    REVIEW_OUTCOME_RECORDED = "review_outcome_recorded"


@dataclass(frozen=True, kw_only=True)
class ReviewMilestonesUpdated(RunLogEntry):
    milestones: list[Milestone] = field(default_factory=list)
    active_milestone_id: str | None = None
    source_tool_call_id: str | None = None
    source_step_id: str | None = None
    reason: Literal["declared", "updated", "completed", "activated"] = "updated"
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.REVIEW_MILESTONES_UPDATED
    )


@dataclass(frozen=True, kw_only=True)
class ReviewTriggerDecided(RunLogEntry):
    trigger_reason: Literal[
        "step_interval", "consecutive_errors", "milestone_switch"
    ]
    active_milestone_id: str | None = None
    review_count_since_checkpoint: int = 0
    trigger_tool_call_id: str = ""
    trigger_tool_step_id: str = ""
    notice_step_id: str = ""
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.REVIEW_TRIGGER_DECIDED
    )


@dataclass(frozen=True, kw_only=True)
class ReviewCheckpointRecorded(RunLogEntry):
    checkpoint_seq: int
    milestone_id: str | None = None
    review_tool_call_id: str | None = None
    review_step_id: str | None = None
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.REVIEW_CHECKPOINT_RECORDED
    )


@dataclass(frozen=True, kw_only=True)
class ReviewOutcomeRecorded(RunLogEntry):
    aligned: bool | None
    mode: Literal["metadata_only", "step_back"]
    experience: str | None = None
    active_milestone_id: str | None = None
    review_tool_call_id: str | None = None
    review_step_id: str | None = None
    hidden_step_ids: list[str] = field(default_factory=list)
    notice_cleaned_step_ids: list[str] = field(default_factory=list)
    condensed_step_ids: list[str] = field(default_factory=list)
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.REVIEW_OUTCOME_RECORDED
    )
```

Import `Literal` and `Milestone` at the top of `agiwo/agent/models/log.py`, and add the new classes to `__all__`.

- [ ] **Step 5: Register serialization support**

Update `agiwo/agent/storage/serialization.py` imports and `_RUN_LOG_TYPES`:

```python
from agiwo.agent.models.log import (
    # existing imports...
    ReviewCheckpointRecorded,
    ReviewMilestonesUpdated,
    ReviewOutcomeRecorded,
    ReviewTriggerDecided,
)
from agiwo.agent.models.review import Milestone

_RUN_LOG_TYPES: dict[RunLogEntryKind, type[RunLogEntry]] = {
    # existing mappings...
    RunLogEntryKind.REVIEW_MILESTONES_UPDATED: ReviewMilestonesUpdated,
    RunLogEntryKind.REVIEW_TRIGGER_DECIDED: ReviewTriggerDecided,
    RunLogEntryKind.REVIEW_CHECKPOINT_RECORDED: ReviewCheckpointRecorded,
    RunLogEntryKind.REVIEW_OUTCOME_RECORDED: ReviewOutcomeRecorded,
}
```

Add milestone restoration in `deserialize_run_log_entry_from_storage`:

```python
if entry_type is ReviewMilestonesUpdated:
    milestones = normalized.get("milestones")
    if isinstance(milestones, list):
        normalized["milestones"] = [
            item if isinstance(item, Milestone) else Milestone(**item)
            for item in milestones
            if isinstance(item, (dict, Milestone))
        ]
```

- [ ] **Step 6: Run fact tests**

Run:

```bash
uv run pytest tests/agent/test_review_run_log_facts.py tests/agent/test_storage_serialization.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add agiwo/agent/models/review.py agiwo/agent/models/log.py agiwo/agent/storage/serialization.py tests/agent/test_review_run_log_facts.py tests/agent/test_storage_serialization.py
git commit -m "feat: add review run log facts"
```

## Task 2: Add Review State Replay From RunLog Facts

**Files:**
- Create: `agiwo/agent/review/replay.py`
- Modify: `agiwo/agent/review/__init__.py`
- Test: `tests/agent/test_review_replay.py`

- [ ] **Step 1: Write failing replay tests**

Create `tests/agent/test_review_replay.py`:

```python
from agiwo.agent.models.log import (
    ReviewCheckpointRecorded,
    ReviewMilestonesUpdated,
    ReviewOutcomeRecorded,
    ReviewTriggerDecided,
    ToolStepCommitted,
)
from agiwo.agent.models.review import Milestone
from agiwo.agent.models.step import MessageRole
from agiwo.agent.review.replay import build_review_state_from_entries


def test_review_replay_restores_milestones_checkpoint_and_count() -> None:
    state = build_review_state_from_entries(
        [
            ReviewMilestonesUpdated(
                sequence=1,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                milestones=[
                    Milestone(id="inspect", description="Inspect", status="active")
                ],
                active_milestone_id="inspect",
                reason="declared",
            ),
            ToolStepCommitted(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                step_id="step-search",
                role=MessageRole.TOOL,
                tool_call_id="tc-search",
                name="web_search",
                content="results",
            ),
            ReviewCheckpointRecorded(
                sequence=3,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                checkpoint_seq=3,
                milestone_id="inspect",
                review_tool_call_id="tc-review",
                review_step_id="step-review",
            ),
            ToolStepCommitted(
                sequence=4,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                step_id="step-read",
                role=MessageRole.TOOL,
                tool_call_id="tc-read",
                name="web_reader",
                content="page",
            ),
            ToolStepCommitted(
                sequence=5,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                step_id="step-review-tool",
                role=MessageRole.TOOL,
                tool_call_id="tc-review-2",
                name="review_trajectory",
                content="Trajectory review",
            ),
        ]
    )

    assert [m.id for m in state.milestones] == ["inspect"]
    assert state.latest_checkpoint is not None
    assert state.latest_checkpoint.seq == 3
    assert state.review_count_since_checkpoint == 1


def test_review_replay_tracks_pending_notice_until_outcome() -> None:
    state = build_review_state_from_entries(
        [
            ReviewTriggerDecided(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                trigger_reason="step_interval",
                active_milestone_id="inspect",
                review_count_since_checkpoint=8,
                trigger_tool_call_id="tc-search",
                trigger_tool_step_id="step-search",
                notice_step_id="step-search",
            )
        ]
    )

    assert state.pending_review_notice is not None
    assert state.pending_review_notice.trigger_reason == "step_interval"

    state = build_review_state_from_entries(
        [
            ReviewTriggerDecided(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                trigger_reason="step_interval",
                active_milestone_id="inspect",
                review_count_since_checkpoint=8,
                trigger_tool_call_id="tc-search",
                trigger_tool_step_id="step-search",
                notice_step_id="step-search",
            ),
            ReviewOutcomeRecorded(
                sequence=3,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                aligned=True,
                mode="metadata_only",
            ),
        ]
    )

    assert state.pending_review_notice is None
    assert state.review_count_since_checkpoint == 0
```

- [ ] **Step 2: Run replay tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_review_replay.py -q
```

Expected: FAIL because `agiwo.agent.review.replay` does not exist.

- [ ] **Step 3: Implement replay helper**

Create `agiwo/agent/review/replay.py`:

```python
"""Replay helpers for goal-directed review state."""

from collections.abc import Iterable

from agiwo.agent.models.log import (
    ReviewCheckpointRecorded,
    ReviewMilestonesUpdated,
    ReviewOutcomeRecorded,
    ReviewTriggerDecided,
    RunLogEntry,
    ToolStepCommitted,
)
from agiwo.agent.models.review import (
    PendingReviewNotice,
    ReviewCheckpoint,
    ReviewState,
)


def build_review_state_from_entries(entries: Iterable[RunLogEntry]) -> ReviewState:
    state = ReviewState()
    checkpoint_seq = 0
    for entry in sorted(entries, key=lambda item: item.sequence):
        if isinstance(entry, ReviewMilestonesUpdated):
            state.milestones = list(entry.milestones)
            continue
        if isinstance(entry, ReviewCheckpointRecorded):
            state.latest_checkpoint = ReviewCheckpoint(
                seq=entry.checkpoint_seq,
                milestone_id=entry.milestone_id or "",
                confirmed_at=entry.created_at,
            )
            state.review_count_since_checkpoint = 0
            checkpoint_seq = entry.sequence
            continue
        if isinstance(entry, ReviewTriggerDecided):
            state.pending_review_notice = PendingReviewNotice(
                trigger_reason=entry.trigger_reason,
                active_milestone_id=entry.active_milestone_id,
                review_count_since_checkpoint=entry.review_count_since_checkpoint,
                trigger_tool_call_id=entry.trigger_tool_call_id,
                trigger_tool_step_id=entry.trigger_tool_step_id,
                notice_step_id=entry.notice_step_id,
            )
            continue
        if isinstance(entry, ReviewOutcomeRecorded):
            state.pending_review_notice = None
            state.pending_review_reason = None
            state.review_count_since_checkpoint = 0
            checkpoint_seq = entry.sequence
            continue
        if isinstance(entry, ToolStepCommitted):
            if entry.name == "review_trajectory":
                continue
            if entry.sequence > checkpoint_seq:
                state.review_count_since_checkpoint += 1
            continue
    return state


__all__ = ["build_review_state_from_entries"]
```

- [ ] **Step 4: Export replay helper from review package**

Update `agiwo/agent/review/__init__.py`:

```python
from agiwo.agent.review.replay import build_review_state_from_entries

__all__ = [
    # existing exports...
    "build_review_state_from_entries",
]
```

- [ ] **Step 5: Run replay tests**

Run:

```bash
uv run pytest tests/agent/test_review_replay.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add agiwo/agent/review/replay.py agiwo/agent/review/__init__.py tests/agent/test_review_replay.py
git commit -m "feat: replay review state from run log facts"
```

## Task 3: Add StateWriter Methods For Review Facts

**Files:**
- Modify: `agiwo/agent/runtime/state_writer.py`
- Test: `tests/agent/test_review_state_writer.py`

- [ ] **Step 1: Write failing state writer tests**

Create `tests/agent/test_review_state_writer.py`:

```python
import pytest

from agiwo.agent.models.log import (
    ReviewCheckpointRecorded,
    ReviewMilestonesUpdated,
    ReviewOutcomeRecorded,
    ReviewTriggerDecided,
)
from agiwo.agent.models.review import Milestone
from agiwo.agent.models.run import RunIdentity
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_writer import RunStateWriter
from agiwo.agent.storage.base import InMemoryRunLogStorage


def _context() -> RunContext:
    return RunContext(
        identity=RunIdentity(
            run_id="run-1",
            agent_id="agent-1",
            agent_name="agent",
        ),
        session_runtime=SessionRuntime(
            session_id="sess-1",
            run_log_storage=InMemoryRunLogStorage(),
        ),
    )


@pytest.mark.asyncio
async def test_state_writer_records_review_facts() -> None:
    context = _context()
    writer = RunStateWriter(context)

    entries = []
    entries.extend(
        await writer.record_review_milestones_updated(
            milestones=[
                Milestone(id="inspect", description="Inspect auth", status="active")
            ],
            active_milestone_id="inspect",
            source_tool_call_id="tc-milestones",
            source_step_id="step-milestones",
            reason="declared",
        )
    )
    entries.extend(
        await writer.record_review_trigger_decided(
            trigger_reason="step_interval",
            active_milestone_id="inspect",
            review_count_since_checkpoint=8,
            trigger_tool_call_id="tc-search",
            trigger_tool_step_id="step-search",
            notice_step_id="step-search",
        )
    )
    entries.extend(
        await writer.record_review_checkpoint_recorded(
            checkpoint_seq=9,
            milestone_id="inspect",
            review_tool_call_id="tc-review",
            review_step_id="step-review",
        )
    )
    entries.extend(
        await writer.record_review_outcome_recorded(
            aligned=True,
            mode="metadata_only",
            experience=None,
            active_milestone_id="inspect",
            review_tool_call_id="tc-review",
            review_step_id="step-review",
            hidden_step_ids=["step-review-call", "step-review"],
            notice_cleaned_step_ids=["step-search"],
            condensed_step_ids=[],
        )
    )

    assert [type(entry) for entry in entries] == [
        ReviewMilestonesUpdated,
        ReviewTriggerDecided,
        ReviewCheckpointRecorded,
        ReviewOutcomeRecorded,
    ]
```

- [ ] **Step 2: Run state writer test and verify it fails**

Run:

```bash
uv run pytest tests/agent/test_review_state_writer.py -q
```

Expected: FAIL because the `RunStateWriter.record_review_*` methods do not exist.

- [ ] **Step 3: Add builder functions and writer methods**

Update `agiwo/agent/runtime/state_writer.py` imports:

```python
from agiwo.agent.models.log import (
    # existing imports...
    ReviewCheckpointRecorded,
    ReviewMilestonesUpdated,
    ReviewOutcomeRecorded,
    ReviewTriggerDecided,
)
from agiwo.agent.models.review import Milestone
```

Add methods to `RunStateWriter`:

```python
async def record_review_milestones_updated(
    self,
    *,
    milestones: list[Milestone],
    active_milestone_id: str | None,
    source_tool_call_id: str | None,
    source_step_id: str | None,
    reason: str,
) -> list[object]:
    return await self.append_entries(
        [
            ReviewMilestonesUpdated(
                sequence=await self._state.session_runtime.allocate_sequence(),
                session_id=self._state.session_id,
                run_id=self._state.run_id,
                agent_id=self._state.agent_id,
                milestones=list(milestones),
                active_milestone_id=active_milestone_id,
                source_tool_call_id=source_tool_call_id,
                source_step_id=source_step_id,
                reason=reason,
            )
        ]
    )
```

Add analogous methods:

```python
async def record_review_trigger_decided(...)
async def record_review_checkpoint_recorded(...)
async def record_review_outcome_recorded(...)
```

Each method allocates exactly one sequence and appends exactly one corresponding entry.

- [ ] **Step 4: Run state writer tests**

Run:

```bash
uv run pytest tests/agent/test_review_state_writer.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add agiwo/agent/runtime/state_writer.py tests/agent/test_review_state_writer.py
git commit -m "feat: record review facts through state writer"
```

## Task 4: Change Review Interval To Non-review Tool Count

**Files:**
- Modify: `agiwo/agent/review/review_enforcer.py`
- Modify: `agiwo/agent/review/__init__.py`
- Test: `tests/agent/test_review_enforcer.py`
- Test: `tests/agent/test_review_batch.py`

- [ ] **Step 1: Write failing interval tests**

Add to `tests/agent/test_review_batch.py`:

```python
def test_process_result_counts_non_review_tools_for_interval() -> None:
    config = AgentOptions(enable_goal_directed_review=True, review_step_interval=2)
    ledger = RunLedger()
    tools_map = {
        "review_trajectory": FakeTool("review_trajectory"),
        "declare_milestones": FakeTool("declare_milestones"),
    }
    batch = ReviewBatch(config, ledger, tools_map)

    first = batch.process_result(
        FakeToolResult("web_search", "results", tool_call_id="tc-1"),
        current_seq=100,
    )
    second = batch.process_result(
        FakeToolResult("declare_milestones", "Milestones declared: a", tool_call_id="tc-2"),
        current_seq=101,
    )

    assert "<system-review>" not in first
    assert "<system-review>" in second
    assert ledger.review.review_count_since_checkpoint == 2


def test_process_result_does_not_count_review_trajectory_for_interval() -> None:
    config = AgentOptions(enable_goal_directed_review=True, review_step_interval=1)
    ledger = RunLedger()
    tools_map = {
        "review_trajectory": FakeTool("review_trajectory"),
        "declare_milestones": FakeTool("declare_milestones"),
    }
    batch = ReviewBatch(config, ledger, tools_map)

    batch.process_result(
        FakeToolResult(
            "review_trajectory",
            "Trajectory review: aligned=True.",
            tool_call_id="tc-review",
            output={"aligned": True},
        ),
        current_seq=9,
    )

    assert ledger.review.review_count_since_checkpoint == 0
```

Update line wrapping after writing; the formatter will split long lines.

- [ ] **Step 2: Run interval tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_review_batch.py::test_process_result_counts_non_review_tools_for_interval tests/agent/test_review_batch.py::test_process_result_does_not_count_review_trajectory_for_interval -q
```

Expected: FAIL because `review_count_since_checkpoint` is not used for interval triggering.

- [ ] **Step 3: Change trigger function signature**

Update `agiwo/agent/review/review_enforcer.py`:

```python
def check_review_trigger(
    *,
    state: ReviewState,
    enabled: bool,
    is_error: bool,
    step_interval: int,
    error_threshold: int,
    tool_name: str = "",
) -> ReviewTrigger:
    if not enabled:
        return ReviewTrigger.NONE
    if tool_name == "review_trajectory":
        return ReviewTrigger.NONE
    if (
        state.pending_review_reason == "milestone_switch"
        and tool_name != "declare_milestones"
    ):
        return ReviewTrigger.MILESTONE_SWITCH
    if is_error and state.consecutive_errors >= error_threshold:
        return ReviewTrigger.CONSECUTIVE_ERRORS
    if state.review_count_since_checkpoint >= step_interval:
        return ReviewTrigger.STEP_INTERVAL
    return ReviewTrigger.NONE
```

Remove `current_seq` from callers after tests guide the update.

- [ ] **Step 4: Count non-review tools in ReviewBatch**

Update `ReviewBatch.process_result` in `agiwo/agent/review/__init__.py`:

```python
if result.tool_name != "review_trajectory":
    self._ledger.review.review_count_since_checkpoint += 1
```

Place this after `declare_milestones` state update and before `check_review_trigger` for non-review tools. Do not count `review_trajectory`.

- [ ] **Step 5: Reset count when review is consumed**

In `_handle_review_result`, set:

```python
self._ledger.review.review_count_since_checkpoint = 0
```

Do this for `aligned is True`, `aligned is False`, and malformed successful review results.

- [ ] **Step 6: Run review batch and enforcer tests**

Run:

```bash
uv run pytest tests/agent/test_review_batch.py tests/agent/test_review_enforcer.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add agiwo/agent/review/review_enforcer.py agiwo/agent/review/__init__.py tests/agent/test_review_batch.py tests/agent/test_review_enforcer.py
git commit -m "fix: count review interval by non-review tools"
```

## Task 5: Write Runtime Review Facts From Tool Batch Execution

**Files:**
- Modify: `agiwo/agent/review/__init__.py`
- Modify: `agiwo/agent/run_tool_batch.py`
- Modify: `agiwo/agent/review/step_back_executor.py`
- Test: `tests/agent/test_run_tool_batch.py`

- [ ] **Step 1: Write failing runtime fact tests**

Add to `tests/agent/test_run_tool_batch.py`:

```python
@pytest.mark.asyncio
async def test_execute_tool_batch_cycle_records_review_trigger_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_execute_tool_batch(*args, **kwargs):
        del args, kwargs
        return [
            ToolResult.success(
                tool_name="search",
                tool_call_id="tc-search",
                content="Found results",
                output={},
            )
        ]

    monkeypatch.setattr(
        run_tool_batch_module, "execute_tool_batch", fake_execute_tool_batch
    )

    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(session_id="sess-1", run_log_storage=storage)
    hooks = _FakeHooks(review_advice=None)
    ledger = RunLedger()
    ledger.review.milestones = [
        Milestone(id="inspect", description="Inspect auth", status="active")
    ]
    context = _FakeContext(
        config=AgentOptions(enable_goal_directed_review=True, review_step_interval=1),
        ledger=ledger,
        hooks=hooks,
        session_runtime=session_runtime,
    )
    runtime = _FakeRuntime(tools_map=_review_tools_map())

    async def commit_step(step):
        return await _commit_step_to_ledger_and_storage(context, step)

    async def set_termination_reason(reason: TerminationReason, tool_name: str) -> None:
        del reason, tool_name

    await run_tool_batch_module.execute_tool_batch_cycle(
        context=context,
        runtime=runtime,
        tool_calls=[
            {"id": "tc-search", "type": "function", "function": {"name": "search"}}
        ],
        assistant_step_id="assistant-step",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )

    entries = await storage.get_entries("sess-1")
    assert any(entry.kind.value == "review_trigger_decided" for entry in entries)
```

Add a second test:

```python
@pytest.mark.asyncio
async def test_aligned_review_records_checkpoint_and_outcome_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [
        [
            ToolResult.success(
                tool_name="search",
                tool_call_id="tc-search",
                content="Found results",
                output={},
            )
        ],
        [
            ToolResult.success(
                tool_name="review_trajectory",
                tool_call_id="tc-review",
                content="Trajectory review: aligned=True.",
                output={"aligned": True, "experience": ""},
            )
        ],
    ]

    async def fake_execute_tool_batch(*args, **kwargs):
        del args, kwargs
        return batches.pop(0)

    monkeypatch.setattr(
        run_tool_batch_module, "execute_tool_batch", fake_execute_tool_batch
    )

    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(session_id="sess-1", run_log_storage=storage)
    hooks = _FakeHooks(review_advice=None)
    ledger = RunLedger()
    ledger.review.milestones = [
        Milestone(id="inspect", description="Inspect auth", status="active")
    ]
    context = _FakeContext(
        config=AgentOptions(enable_goal_directed_review=True, review_step_interval=1),
        ledger=ledger,
        hooks=hooks,
        session_runtime=session_runtime,
    )
    runtime = _FakeRuntime(tools_map=_review_tools_map())

    async def commit_step(step):
        return await _commit_step_to_ledger_and_storage(context, step)

    async def set_termination_reason(reason: TerminationReason, tool_name: str) -> None:
        del reason, tool_name

    await run_tool_batch_module.execute_tool_batch_cycle(
        context=context,
        runtime=runtime,
        tool_calls=[
            {"id": "tc-search", "type": "function", "function": {"name": "search"}}
        ],
        assistant_step_id="assistant-step-search",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )
    await run_tool_batch_module.execute_tool_batch_cycle(
        context=context,
        runtime=runtime,
        tool_calls=[
            {
                "id": "tc-review",
                "type": "function",
                "function": {"name": "review_trajectory"},
            }
        ],
        assistant_step_id="assistant-step-review",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )

    kinds = [entry.kind.value for entry in await storage.get_entries("sess-1")]
    assert "review_checkpoint_recorded" in kinds
    assert "review_outcome_recorded" in kinds
```

- [ ] **Step 2: Run runtime fact tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_run_tool_batch.py::test_execute_tool_batch_cycle_records_review_trigger_fact tests/agent/test_run_tool_batch.py::test_aligned_review_records_checkpoint_and_outcome_facts -q
```

Expected: FAIL because review facts are not written during tool batch execution.

- [ ] **Step 3: Extend ReviewBatch outcome objects**

Update `ReviewNoticeRequest` in `agiwo/agent/review/__init__.py`:

```python
@dataclass(frozen=True)
class ReviewNoticeRequest:
    content: str
    milestone: Milestone | None
    step_count: int
    trigger: ReviewTrigger
    trigger_tool_call_id: str
    trigger_tool_step_id: str
```

When creating it, include `result.tool_call_id or ""` and `tool_step_id or ""`.

- [ ] **Step 4: Write milestone update facts after commit**

In `execute_tool_batch_cycle`, after `committed_step = await commit_step(tool_step)`, check whether the processed result declared milestones. If using a `ReviewBatch` method is cleaner, add:

```python
milestone_update = batch.consume_milestone_update()
```

The update should include the full board, active milestone id, source tool call id, source step id, and reason. Then call:

```python
await writer.record_review_milestones_updated(
    milestones=milestone_update.milestones,
    active_milestone_id=milestone_update.active_milestone_id,
    source_tool_call_id=call_id,
    source_step_id=committed_step.id,
    reason=milestone_update.reason,
)
```

- [ ] **Step 5: Write trigger fact after notice injection**

After `committed_step = await commit_step(tool_step)`, if `review_notice is not None`, call:

```python
await writer.record_review_trigger_decided(
    trigger_reason=review_notice.trigger.value,
    active_milestone_id=(
        review_notice.milestone.id if review_notice.milestone is not None else None
    ),
    review_count_since_checkpoint=context.ledger.review.review_count_since_checkpoint,
    trigger_tool_call_id=call_id,
    trigger_tool_step_id=committed_step.id,
    notice_step_id=committed_step.id,
)
```

Then store `context.ledger.review.pending_review_notice` with the same values.

- [ ] **Step 6: Write checkpoint and outcome facts in apply phase**

In `_apply_review_outcome`, after hidden step and content update handling, record facts before returning:

```python
if outcome.checkpoint_seq:
    await writer.record_review_checkpoint_recorded(
        checkpoint_seq=outcome.checkpoint_seq,
        milestone_id=outcome.active_milestone_id,
        review_tool_call_id=outcome.review_tool_call_id,
        review_step_id=outcome.review_step_id,
    )

await writer.record_review_outcome_recorded(
    aligned=outcome.aligned,
    mode=outcome.mode,
    experience=outcome.experience,
    active_milestone_id=outcome.active_milestone_id,
    review_tool_call_id=outcome.review_tool_call_id,
    review_step_id=outcome.review_step_id,
    hidden_step_ids=outcome.hidden_step_ids,
    notice_cleaned_step_ids=[update.step_id for update in outcome.content_updates],
    condensed_step_ids=outcome.condensed_step_ids,
)
```

Add these exact fields to `StepBackOutcome`:

```python
aligned: bool | None = None
active_milestone_id: str | None = None
review_step_id: str | None = None
condensed_step_ids: list[str] = field(default_factory=list)
```

- [ ] **Step 7: Run runtime fact tests**

Run:

```bash
uv run pytest tests/agent/test_run_tool_batch.py tests/agent/test_review_batch.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

Run:

```bash
git add agiwo/agent/review/__init__.py agiwo/agent/run_tool_batch.py agiwo/agent/review/step_back_executor.py tests/agent/test_run_tool_batch.py
git commit -m "feat: write review facts during tool execution"
```

## Task 6: Bootstrap Persistent ReviewState From Facts

**Files:**
- Modify: `agiwo/agent/run_bootstrap.py`
- Modify: `agiwo/agent/runtime/session.py` if session helpers are needed
- Test: `tests/agent/test_run_bootstrap.py` or create `tests/agent/test_review_bootstrap.py`

- [ ] **Step 1: Write failing bootstrap replay test**

Create `tests/agent/test_review_bootstrap.py`:

```python
import pytest

from agiwo.agent.models.log import ReviewMilestonesUpdated, ToolStepCommitted
from agiwo.agent.models.review import Milestone
from agiwo.agent.models.run import RunIdentity
from agiwo.agent.models.step import MessageRole
from agiwo.agent.review.replay import build_review_state_from_entries
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage


@pytest.mark.asyncio
async def test_review_state_replay_uses_persisted_run_log_entries() -> None:
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            ReviewMilestonesUpdated(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                milestones=[
                    Milestone(id="inspect", description="Inspect auth", status="active")
                ],
                active_milestone_id="inspect",
                reason="declared",
            ),
            ToolStepCommitted(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                step_id="step-tool",
                role=MessageRole.TOOL,
                name="web_search",
                tool_call_id="tc-search",
                content="results",
            ),
        ]
    )
    context = RunContext(
        identity=RunIdentity(
            run_id="run-2",
            agent_id="agent-1",
            agent_name="agent",
        ),
        session_runtime=SessionRuntime(
            session_id="sess-1",
            run_log_storage=storage,
        ),
    )

    entries = await storage.get_entries("sess-1")
    context.ledger.review = build_review_state_from_entries(entries)

    assert context.ledger.review.milestones[0].id == "inspect"
    assert context.ledger.review.review_count_since_checkpoint == 1
```

This test validates the helper path first. If existing bootstrap tests can instantiate `prepare_run_context` cheaply, add a second test that calls bootstrap directly.

- [ ] **Step 2: Add bootstrap integration**

In `agiwo/agent/run_bootstrap.py`, before assembling run messages, load session entries and assign review state:

```python
from agiwo.agent.review.replay import build_review_state_from_entries

entries = await context.session_runtime.run_log_storage.get_entries(
    context.session_id
)
context.ledger.review = build_review_state_from_entries(
    [
        entry
        for entry in entries
        if entry.agent_id == context.agent_id
    ]
)
```

Keep this close to existing step loading so persistent sessions restore review facts before new tool calls occur.

- [ ] **Step 3: Run bootstrap/replay tests**

Run:

```bash
uv run pytest tests/agent/test_review_bootstrap.py tests/agent/test_review_replay.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit Task 6**

Run:

```bash
git add agiwo/agent/run_bootstrap.py tests/agent/test_review_bootstrap.py
git commit -m "feat: restore review state during run bootstrap"
```

## Task 7: Project Review Facts To Trace And Convert Console Review Views

**Files:**
- Modify: `agiwo/agent/trace_writer.py`
- Modify: `console/server/services/runtime/runtime_observability.py`
- Test: `tests/agent/test_trace_writer.py`
- Test: `console/tests/test_runtime_observability.py`
- Test: `console/tests/test_traces_api.py`
- Test: `console/tests/test_sessions_api.py`

- [ ] **Step 1: Write failing trace projection test for review facts**

Add to `tests/agent/test_trace_writer.py`:

```python
from agiwo.agent.models.log import ReviewMilestonesUpdated, RunStarted
from agiwo.agent.models.review import Milestone
from agiwo.agent.trace_writer import AgentTraceCollector


def test_trace_writer_projects_review_milestone_fact_to_runtime_span() -> None:
    trace = AgentTraceCollector().build_from_entries(
        [
            RunStarted(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
            ),
            ReviewMilestonesUpdated(
                sequence=2,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                milestones=[
                    Milestone(id="inspect", description="Inspect auth", status="active")
                ],
                active_milestone_id="inspect",
                reason="declared",
            ),
        ]
    )

    review_span = next(span for span in trace.spans if span.name == "review_milestones")

    assert review_span.kind.value == "runtime"
    assert review_span.attributes["active_milestone_id"] == "inspect"
    assert review_span.attributes["milestones"][0]["id"] == "inspect"
```

- [ ] **Step 2: Project review facts in trace writer**

Update `agiwo/agent/trace_writer.py` imports and runtime-entry type unions to include:

```python
ReviewCheckpointRecorded
ReviewMilestonesUpdated
ReviewOutcomeRecorded
ReviewTriggerDecided
```

Extend `_build_runtime_span_from_entry`:

```python
elif isinstance(entry, ReviewMilestonesUpdated):
    name = "review_milestones"
    attributes.update(
        {
            "milestones": [
                {
                    "id": milestone.id,
                    "description": milestone.description,
                    "status": milestone.status,
                    "declared_at_seq": milestone.declared_at_seq,
                    "completed_at_seq": milestone.completed_at_seq,
                }
                for milestone in entry.milestones
            ],
            "active_milestone_id": entry.active_milestone_id,
            "source_tool_call_id": entry.source_tool_call_id,
            "source_step_id": entry.source_step_id,
            "reason": entry.reason,
        }
    )
elif isinstance(entry, ReviewTriggerDecided):
    name = "review_trigger"
    attributes.update(
        {
            "trigger_reason": entry.trigger_reason,
            "active_milestone_id": entry.active_milestone_id,
            "review_count_since_checkpoint": entry.review_count_since_checkpoint,
            "trigger_tool_call_id": entry.trigger_tool_call_id,
            "trigger_tool_step_id": entry.trigger_tool_step_id,
            "notice_step_id": entry.notice_step_id,
        }
    )
elif isinstance(entry, ReviewCheckpointRecorded):
    name = "review_checkpoint"
    attributes.update(
        {
            "checkpoint_seq": entry.checkpoint_seq,
            "milestone_id": entry.milestone_id,
            "review_tool_call_id": entry.review_tool_call_id,
            "review_step_id": entry.review_step_id,
        }
    )
elif isinstance(entry, ReviewOutcomeRecorded):
    name = "review_outcome"
    attributes.update(
        {
            "aligned": entry.aligned,
            "mode": entry.mode,
            "experience": entry.experience,
            "active_milestone_id": entry.active_milestone_id,
            "review_tool_call_id": entry.review_tool_call_id,
            "review_step_id": entry.review_step_id,
            "hidden_step_ids": entry.hidden_step_ids,
            "notice_cleaned_step_ids": entry.notice_cleaned_step_ids,
            "condensed_step_ids": entry.condensed_step_ids,
        }
    )
```

Also add the four review fact classes to `_apply_runtime_entry_to_trace` and `_append_runtime_entry_to_trace` type checks.

- [ ] **Step 3: Run trace projection test**

Run:

```bash
uv run pytest tests/agent/test_trace_writer.py::test_trace_writer_projects_review_milestone_fact_to_runtime_span -q
```

Expected: PASS.

- [ ] **Step 4: Write failing observability test for fact-backed board**

Add to `console/tests/test_runtime_observability.py`:

```python
from datetime import datetime, timezone

from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace


def _trace_with_review_fact_spans() -> Trace:
    trace = Trace(
        trace_id="trace-review-facts",
        agent_id="agent-1",
        session_id="sess-1",
        start_time=datetime(2026, 4, 26, tzinfo=timezone.utc),
    )
    trace.spans = [
        Span(
            trace_id=trace.trace_id,
            span_id="review-milestones",
            kind=SpanKind.RUNTIME,
            name="review_milestones",
            run_id="run-1",
            status=SpanStatus.OK,
            start_time=trace.start_time,
            attributes={
                "sequence": 10,
                "milestones": [
                    {
                        "id": "inspect",
                        "description": "Inspect auth",
                        "status": "active",
                        "declared_at_seq": 7,
                        "completed_at_seq": None,
                    }
                ],
                "active_milestone_id": "inspect",
            },
        ),
        Span(
            trace_id=trace.trace_id,
            span_id="review-checkpoint",
            kind=SpanKind.RUNTIME,
            name="review_checkpoint",
            run_id="run-1",
            status=SpanStatus.OK,
            start_time=trace.start_time,
            attributes={
                "sequence": 11,
                "checkpoint_seq": 9,
                "milestone_id": "inspect",
                "review_tool_call_id": "tc-review",
                "review_step_id": "step-review",
            },
        ),
        Span(
            trace_id=trace.trace_id,
            span_id="review-outcome",
            kind=SpanKind.RUNTIME,
            name="review_outcome",
            run_id="run-1",
            status=SpanStatus.OK,
            start_time=trace.start_time,
            attributes={
                "sequence": 12,
                "aligned": True,
                "mode": "metadata_only",
                "active_milestone_id": "inspect",
            },
        ),
    ]
    return trace


def test_milestone_board_uses_review_fact_spans_not_tool_payloads() -> None:
    trace = _trace_with_review_fact_spans()

    board = build_session_milestone_board(
        session_id="sess-1",
        trace=trace,
        review_cycles=build_trace_review_cycles(trace),
    )

    assert board is not None
    assert board.active_milestone_id == "inspect"
    assert board.latest_checkpoint is not None
    assert board.latest_review_outcome is not None
```

The test should fail until runtime observability reads runtime spans projected from review facts.

- [ ] **Step 5: Add fact-span collection helpers**

In `console/server/services/runtime/runtime_observability.py`, add helpers:

```python
def _collect_milestones_from_review_fact_spans(trace: Trace) -> list[MilestoneRecord]:
    records: list[MilestoneRecord] = []
    latest = _latest_runtime_span(trace, name="review_milestones")
    if latest is None:
        return []
    raw_milestones = latest.attributes.get("milestones")
    if not isinstance(raw_milestones, list):
        return []
    for raw in raw_milestones:
        if not isinstance(raw, dict):
            continue
        records.append(
            MilestoneRecord(
                id=str(raw["id"]),
                description=str(raw["description"]),
                status=str(raw["status"]),
                declared_at_seq=_as_int(raw.get("declared_at_seq")),
                completed_at_seq=_as_int(raw.get("completed_at_seq")),
            )
        )
    return records
```

Add analogous helpers for latest checkpoint, latest outcome, and review cycles. Use runtime spans named `review_trigger`, `review_checkpoint`, and `review_outcome`.

- [ ] **Step 6: Remove authoritative tool-span fallback**

Change milestone board and review cycle builders so they do not call `_collect_milestones_from_trace` as authoritative fallback. For legacy traces without review facts:

```python
if not review_fact_entries:
    return None
```

For API responses, keep existing nullable fields as `None` and return `review_cycles=[]`.

- [ ] **Step 7: Run Console observability tests**

Run:

```bash
uv run python scripts/check.py console-tests
```

Expected: PASS after updating tests that expected legacy tool-span inference.

- [ ] **Step 8: Commit Task 7**

Run:

```bash
git add agiwo/agent/trace_writer.py console/server/services/runtime/runtime_observability.py tests/agent/test_trace_writer.py console/tests/test_runtime_observability.py console/tests/test_traces_api.py console/tests/test_sessions_api.py
git commit -m "feat: build review observability from run log facts"
```

## Task 8: Documentation And Final Guardrails

**Files:**
- Modify: `docs/guides/context-optimization.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update context optimization guide**

In `docs/guides/context-optimization.md`, update the review section to state:

```markdown
`review_step_interval` counts completed tool results except `review_trajectory`.
`declare_milestones` and scheduler control tools count toward the interval.
Milestone boards and review cycles are persisted as first-class RunLog facts:
`review_milestones_updated`, `review_trigger_decided`,
`review_checkpoint_recorded`, and `review_outcome_recorded`.
Console and trace views use these facts as the authoritative source.
```

- [ ] **Step 2: Run full required verification**

Run:

```bash
uv run pytest tests/agent/ -v
uv run python scripts/check.py console-tests
uv run python scripts/lint.py ci
```

Expected: all commands pass.

- [ ] **Step 3: Inspect final status**

Run:

```bash
git status --short
```

Expected: only intended documentation changes are unstaged or staged for this task.

- [ ] **Step 4: Commit Task 8**

Run:

```bash
git add docs/guides/context-optimization.md AGENTS.md
git commit -m "docs: document review runtime facts"
```

## Self-Review Checklist

- Spec coverage: The plan covers interval semantics, first-class facts, replay, prompt notice lifecycle, Console/trace read-side changes, and no migration for old sessions.
- Scope control: The plan does not implement backoff, budget, soft review, or `review_trajectory` schema redesign.
- Type consistency: The plan consistently uses `ReviewMilestonesUpdated`, `ReviewTriggerDecided`, `ReviewCheckpointRecorded`, `ReviewOutcomeRecorded`, `PendingReviewNotice`, and `review_count_since_checkpoint`.
- Testing coverage: SDK unit tests cover model serialization, replay, runtime fact writing, interval semantics, and prompt cleanup. Console tests cover fact-backed milestone/review observability and legacy no-fallback behavior.
