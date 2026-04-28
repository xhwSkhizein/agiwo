# Agent Introspect Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `agiwo.agent.review` implementation with a cleaner `agiwo.agent.introspect` subsystem backed only by first-class RunLog facts.

**Architecture:** Keep `run_tool_batch.py` as the explicit tool-batch execution owner, but move goal, trajectory, and context-repair rules into focused `agiwo/agent/introspect/*` modules. All committed facts go through `RunStateWriter` and are projected through `SessionRuntime`; Console and trace views consume facts/spans only.

**Tech Stack:** Python 3.10+, dataclasses, pytest, existing RunLog storage, existing Agent runtime/session/trace abstractions.

---

## File Map

- Create `agiwo/agent/introspect/__init__.py`: public exports for the new subsystem.
- Create `agiwo/agent/introspect/models.py`: pure introspect data structures.
- Create `agiwo/agent/introspect/goal.py`: milestone parsing, validation, and goal state updates.
- Create `agiwo/agent/introspect/trajectory.py`: trigger detection, notice rendering, and `review_trajectory` outcome parsing.
- Create `agiwo/agent/introspect/repair.py`: pure context repair planning.
- Create `agiwo/agent/introspect/apply.py`: commit helpers that call `RunStateWriter` and project entries.
- Create `agiwo/agent/introspect/replay.py`: rebuild `GoalState` and `IntrospectionState` from RunLog facts.
- Modify `agiwo/agent/models/run.py`: replace `ReviewState` on `RunLedger` with `GoalState` and `IntrospectionState`.
- Modify `agiwo/agent/models/log.py`: replace review fact dataclasses with introspect fact dataclasses.
- Modify `agiwo/agent/runtime/state_writer.py`: add writer methods for new facts and `StepCondensedContentUpdated`.
- Modify `agiwo/agent/storage/serialization.py`, `agiwo/agent/storage/base.py`, `agiwo/agent/storage/sqlite.py`: serialize new facts and keep condensed-content replay.
- Modify `agiwo/agent/run_bootstrap.py`: restore introspect state from RunLog entries.
- Modify `agiwo/agent/run_tool_batch.py`: replace `ReviewBatch` with explicit calls into `introspect`.
- Modify `agiwo/agent/trace_writer.py`: project new facts into runtime spans.
- Modify `console/server/services/runtime/runtime_observability.py`: remove `<system-review>` parsing and consume new spans.
- Modify `agiwo/scheduler/runtime_tools.py`: keep tool names, update descriptions/errors for stricter validation where needed.
- Modify `agiwo/agent/prompt.py`: rename prose from goal-directed review to introspection if public prompt wording needs it.
- Delete `agiwo/agent/review/`.
- Delete `agiwo/agent/models/review.py`.
- Update `AGENTS.md`.
- Replace review tests under `tests/agent/` with introspect tests.
- Update Console observability tests under `console/tests/`.

## Task 1: Add Introspect Models

**Files:**
- Create: `agiwo/agent/introspect/__init__.py`
- Create: `agiwo/agent/introspect/models.py`
- Test: `tests/agent/test_introspect_models.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/agent/test_introspect_models.py`:

```python
from agiwo.agent.introspect.models import (
    ContentUpdate,
    ContextRepairPlan,
    GoalState,
    IntrospectionOutcome,
    IntrospectionState,
    Milestone,
)


def test_goal_state_tracks_active_milestone() -> None:
    state = GoalState(
        milestones=[Milestone(id="inspect", description="Inspect", status="active")],
        active_milestone_id="inspect",
    )

    assert state.active_milestone is not None
    assert state.active_milestone.description == "Inspect"


def test_introspection_state_defaults_to_clean_boundary() -> None:
    state = IntrospectionState()

    assert state.review_count_since_boundary == 0
    assert state.consecutive_errors == 0
    assert state.last_boundary_seq == 0
    assert state.pending_trigger is None
    assert state.latest_aligned_checkpoint is None


def test_context_repair_plan_reports_affected_steps() -> None:
    plan = ContextRepairPlan(
        mode="step_back",
        start_seq=3,
        end_seq=8,
        experience="Search drifted into unrelated JWT code.",
        content_updates=[
            ContentUpdate(
                step_id="step-search",
                tool_call_id="tc-search",
                content="[EXPERIENCE] Search drifted into unrelated JWT code.",
            )
        ],
    )

    assert plan.affected_count == 1


def test_introspection_outcome_advances_boundary() -> None:
    outcome = IntrospectionOutcome(
        aligned=False,
        mode="step_back",
        boundary_seq=12,
        experience="The search drifted.",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
    )

    assert outcome.boundary_seq == 12
    assert outcome.mode == "step_back"
```

- [ ] **Step 2: Run model tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_introspect_models.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agiwo.agent.introspect'`.

- [ ] **Step 3: Add model implementation**

Create `agiwo/agent/introspect/models.py`:

```python
"""Data models for agent goal, trajectory introspection, and context repair."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


MilestoneStatus = Literal["pending", "active", "completed", "abandoned"]
GoalUpdateReason = Literal["declared", "updated", "completed", "activated"]
IntrospectionTriggerReason = Literal[
    "step_interval", "consecutive_errors", "milestone_switch"
]
IntrospectionMode = Literal["metadata_only", "step_back"]
ContextRepairMode = Literal["step_back"]


@dataclass
class Milestone:
    id: str
    description: str
    status: MilestoneStatus = "pending"
    declared_at_seq: int = 0
    completed_at_seq: int | None = None


@dataclass
class GoalState:
    milestones: list[Milestone] = field(default_factory=list)
    active_milestone_id: str | None = None

    @property
    def active_milestone(self) -> Milestone | None:
        if self.active_milestone_id is None:
            return None
        for milestone in self.milestones:
            if milestone.id == self.active_milestone_id:
                return milestone
        return None


@dataclass
class IntrospectionCheckpoint:
    seq: int
    milestone_id: str
    confirmed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PendingIntrospectionNotice:
    trigger_reason: IntrospectionTriggerReason
    active_milestone_id: str | None
    review_count_since_boundary: int
    trigger_tool_call_id: str | None
    trigger_tool_step_id: str | None
    notice_step_id: str | None


@dataclass
class IntrospectionState:
    review_count_since_boundary: int = 0
    consecutive_errors: int = 0
    pending_trigger: PendingIntrospectionNotice | None = None
    last_boundary_seq: int = 0
    latest_aligned_checkpoint: IntrospectionCheckpoint | None = None
    pending_milestone_switch: bool = False


@dataclass(frozen=True)
class GoalUpdate:
    milestones: list[Milestone]
    active_milestone_id: str | None
    source_tool_call_id: str | None
    reason: GoalUpdateReason
    milestone_switch: bool = False


@dataclass(frozen=True)
class IntrospectionNotice:
    content: str
    active_milestone: Milestone | None
    step_count: int
    trigger_reason: IntrospectionTriggerReason


@dataclass(frozen=True)
class ContentUpdate:
    step_id: str
    tool_call_id: str
    content: str


@dataclass
class ContextRepairPlan:
    mode: ContextRepairMode
    start_seq: int
    end_seq: int
    experience: str
    content_updates: list[ContentUpdate] = field(default_factory=list)
    notice_cleaned_step_ids: list[str] = field(default_factory=list)

    @property
    def affected_count(self) -> int:
        return len(self.content_updates)

    @property
    def condensed_step_ids(self) -> list[str]:
        return [update.step_id for update in self.content_updates]


@dataclass
class IntrospectionOutcome:
    aligned: bool | None
    mode: IntrospectionMode
    boundary_seq: int
    experience: str | None = None
    active_milestone_id: str | None = None
    review_tool_call_id: str | None = None
    review_step_id: str | None = None
    hidden_step_ids: list[str] = field(default_factory=list)
    repair_plan: ContextRepairPlan | None = None


__all__ = [
    "ContentUpdate",
    "ContextRepairMode",
    "ContextRepairPlan",
    "GoalState",
    "GoalUpdate",
    "GoalUpdateReason",
    "IntrospectionCheckpoint",
    "IntrospectionMode",
    "IntrospectionNotice",
    "IntrospectionOutcome",
    "IntrospectionState",
    "IntrospectionTriggerReason",
    "Milestone",
    "MilestoneStatus",
    "PendingIntrospectionNotice",
]
```

Create `agiwo/agent/introspect/__init__.py`:

```python
"""Agent introspection subsystem."""

from agiwo.agent.introspect.models import (
    ContentUpdate,
    ContextRepairPlan,
    GoalState,
    GoalUpdate,
    IntrospectionCheckpoint,
    IntrospectionNotice,
    IntrospectionOutcome,
    IntrospectionState,
    Milestone,
    PendingIntrospectionNotice,
)

__all__ = [
    "ContentUpdate",
    "ContextRepairPlan",
    "GoalState",
    "GoalUpdate",
    "IntrospectionCheckpoint",
    "IntrospectionNotice",
    "IntrospectionOutcome",
    "IntrospectionState",
    "Milestone",
    "PendingIntrospectionNotice",
]
```

- [ ] **Step 4: Run model tests and verify they pass**

Run:

```bash
uv run pytest tests/agent/test_introspect_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/introspect/__init__.py agiwo/agent/introspect/models.py tests/agent/test_introspect_models.py
git commit -m "feat: add introspect data models"
```

## Task 2: Implement Goal Milestone Rules

**Files:**
- Create: `agiwo/agent/introspect/goal.py`
- Test: `tests/agent/test_introspect_goal.py`

- [ ] **Step 1: Write failing goal tests**

Create `tests/agent/test_introspect_goal.py`:

```python
import pytest

from agiwo.agent.introspect.goal import (
    GoalValidationError,
    handle_goal_tool_result,
    update_goal_milestones,
)
from agiwo.agent.introspect.models import GoalState, Milestone
from agiwo.tool.base import ToolResult


def test_declare_milestones_activates_first_pending() -> None:
    state = GoalState()
    result = ToolResult.success(
        tool_name="declare_milestones",
        tool_call_id="tc-declare",
        content="ok",
        output={"milestones": [{"id": "inspect", "description": "Inspect auth"}]},
    )

    update = handle_goal_tool_result(result, state, current_seq=4)

    assert update is not None
    assert update.active_milestone_id == "inspect"
    assert [(m.id, m.status, m.declared_at_seq) for m in state.milestones] == [
        ("inspect", "active", 4)
    ]


def test_duplicate_milestone_ids_fail_fast() -> None:
    state = GoalState()

    with pytest.raises(GoalValidationError, match="duplicate milestone id"):
        update_goal_milestones(
            state,
            [
                Milestone(id="inspect", description="Inspect"),
                Milestone(id="inspect", description="Inspect again"),
            ],
            current_seq=1,
            source_tool_call_id="tc",
        )


def test_multiple_active_milestones_fail_fast() -> None:
    state = GoalState()

    with pytest.raises(GoalValidationError, match="at most one active"):
        update_goal_milestones(
            state,
            [
                Milestone(id="a", description="A", status="active"),
                Milestone(id="b", description="B", status="active"),
            ],
            current_seq=1,
            source_tool_call_id="tc",
        )


def test_active_switch_marks_milestone_switch() -> None:
    state = GoalState(
        milestones=[
            Milestone(id="inspect", description="Inspect", status="active"),
            Milestone(id="fix", description="Fix", status="pending"),
        ],
        active_milestone_id="inspect",
    )

    update = update_goal_milestones(
        state,
        [
            Milestone(id="inspect", description="Inspect", status="completed"),
            Milestone(id="fix", description="Fix", status="active"),
        ],
        current_seq=9,
        source_tool_call_id="tc",
        reason="activated",
    )

    assert update.milestone_switch is True
    assert state.active_milestone_id == "fix"
```

- [ ] **Step 2: Run goal tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_introspect_goal.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agiwo.agent.introspect.goal'`.

- [ ] **Step 3: Implement `goal.py`**

Create `agiwo/agent/introspect/goal.py`:

```python
"""Goal and milestone rules for agent introspection."""

from agiwo.agent.introspect.models import (
    GoalState,
    GoalUpdate,
    GoalUpdateReason,
    Milestone,
    MilestoneStatus,
)
from agiwo.tool.base import ToolResult

_VALID_STATUSES: set[str] = {"pending", "active", "completed", "abandoned"}


class GoalValidationError(ValueError):
    """Raised when a milestone declaration is not a valid goal contract."""


def parse_declared_milestones(output: object) -> list[Milestone]:
    if not isinstance(output, dict):
        return []
    raw_milestones = output.get("milestones")
    if not isinstance(raw_milestones, list):
        return []

    milestones: list[Milestone] = []
    seen_ids: set[str] = set()
    active_count = 0
    for raw in raw_milestones:
        if not isinstance(raw, dict):
            raise GoalValidationError("milestones must contain objects")
        milestone_id = raw.get("id")
        description = raw.get("description")
        if not isinstance(milestone_id, str) or not milestone_id.strip():
            raise GoalValidationError("milestone id must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise GoalValidationError("milestone description must be a non-empty string")
        milestone_id = milestone_id.strip()
        if milestone_id in seen_ids:
            raise GoalValidationError(f"duplicate milestone id: {milestone_id}")
        seen_ids.add(milestone_id)
        status = raw.get("status", "pending")
        if not isinstance(status, str) or status not in _VALID_STATUSES:
            raise GoalValidationError(f"invalid milestone status: {status}")
        if status == "active":
            active_count += 1
        milestones.append(
            Milestone(
                id=milestone_id,
                description=description.strip(),
                status=status,  # type: ignore[arg-type]
            )
        )
    if active_count > 1:
        raise GoalValidationError("milestones may contain at most one active item")
    return milestones


def _active_milestone_id(milestones: list[Milestone]) -> str | None:
    for milestone in milestones:
        if milestone.status == "active":
            return milestone.id
    return None


def _validate_milestones(milestones: list[Milestone]) -> None:
    seen_ids: set[str] = set()
    active_count = 0
    for milestone in milestones:
        if not milestone.id.strip():
            raise GoalValidationError("milestone id must be a non-empty string")
        if milestone.id in seen_ids:
            raise GoalValidationError(f"duplicate milestone id: {milestone.id}")
        seen_ids.add(milestone.id)
        if milestone.status == "active":
            active_count += 1
    if active_count > 1:
        raise GoalValidationError("milestones may contain at most one active item")


def update_goal_milestones(
    state: GoalState,
    milestones: list[Milestone],
    *,
    current_seq: int,
    source_tool_call_id: str | None,
    reason: GoalUpdateReason = "declared",
) -> GoalUpdate:
    _validate_milestones(milestones)
    previous_active_id = state.active_milestone_id or _active_milestone_id(
        state.milestones
    )
    existing_by_id = {milestone.id: milestone for milestone in state.milestones}
    next_by_id = dict(existing_by_id)

    for milestone in milestones:
        existing = existing_by_id.get(milestone.id)
        if existing is not None:
            milestone.declared_at_seq = existing.declared_at_seq
            milestone.completed_at_seq = existing.completed_at_seq
        else:
            milestone.declared_at_seq = current_seq
        if milestone.status == "completed" and milestone.completed_at_seq is None:
            milestone.completed_at_seq = current_seq
        next_by_id[milestone.id] = milestone

    next_milestones = list(next_by_id.values())
    if _active_milestone_id(next_milestones) is None:
        for milestone in next_milestones:
            if milestone.status == "pending":
                milestone.status = "active"
                break

    active_id = _active_milestone_id(next_milestones)
    state.milestones = next_milestones
    state.active_milestone_id = active_id
    return GoalUpdate(
        milestones=list(state.milestones),
        active_milestone_id=active_id,
        source_tool_call_id=source_tool_call_id,
        reason=reason,
        milestone_switch=previous_active_id is not None
        and active_id != previous_active_id,
    )


def handle_goal_tool_result(
    result: ToolResult,
    state: GoalState,
    *,
    current_seq: int,
) -> GoalUpdate | None:
    if result.tool_name != "declare_milestones" or not result.is_success:
        return None
    milestones = parse_declared_milestones(result.output)
    if not milestones:
        return None
    return update_goal_milestones(
        state,
        milestones,
        current_seq=current_seq,
        source_tool_call_id=result.tool_call_id or None,
    )


__all__ = [
    "GoalValidationError",
    "handle_goal_tool_result",
    "parse_declared_milestones",
    "update_goal_milestones",
]
```

- [ ] **Step 4: Run goal tests**

Run:

```bash
uv run pytest tests/agent/test_introspect_goal.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/introspect/goal.py tests/agent/test_introspect_goal.py
git commit -m "feat: add introspect goal rules"
```

## Task 3: Implement Trajectory Trigger And Outcome Rules

**Files:**
- Create: `agiwo/agent/introspect/trajectory.py`
- Test: `tests/agent/test_introspect_trajectory.py`

- [ ] **Step 1: Write failing trajectory tests**

Create `tests/agent/test_introspect_trajectory.py`:

```python
from agiwo.agent.introspect.models import GoalState, IntrospectionState, Milestone
from agiwo.agent.introspect.trajectory import (
    maybe_build_introspection_notice,
    parse_introspection_outcome,
    strip_system_review_notices,
)
from agiwo.tool.base import ToolResult


def test_step_interval_builds_notice() -> None:
    goal = GoalState(
        milestones=[Milestone(id="inspect", description="Inspect auth", status="active")],
        active_milestone_id="inspect",
    )
    state = IntrospectionState()
    first = ToolResult.success(tool_name="search", tool_call_id="tc1", content="one")
    second = ToolResult.success(tool_name="read", tool_call_id="tc2", content="two")

    assert maybe_build_introspection_notice(
        first, goal, state, step_interval=2, review_on_error=True
    ) is None
    notice = maybe_build_introspection_notice(
        second, goal, state, step_interval=2, review_on_error=True
    )

    assert notice is not None
    assert notice.trigger_reason == "step_interval"
    assert notice.step_count == 2
    assert "<system-review>" in notice.content


def test_review_trajectory_does_not_increment_counter() -> None:
    goal = GoalState()
    state = IntrospectionState(review_count_since_boundary=5)
    result = ToolResult.success(
        tool_name="review_trajectory",
        tool_call_id="tc-review",
        content="review",
        output={"aligned": True, "experience": "ok"},
    )

    notice = maybe_build_introspection_notice(
        result, goal, state, step_interval=6, review_on_error=True
    )

    assert notice is None
    assert state.review_count_since_boundary == 5


def test_parse_misaligned_outcome_advances_boundary() -> None:
    goal = GoalState(
        milestones=[Milestone(id="inspect", description="Inspect", status="active")],
        active_milestone_id="inspect",
    )
    result = ToolResult.success(
        tool_name="review_trajectory",
        tool_call_id="tc-review",
        content="Trajectory review: aligned=False. JWT drifted",
        output={"aligned": False, "experience": "JWT drifted"},
    )

    outcome = parse_introspection_outcome(
        result,
        goal,
        current_seq=12,
        assistant_step_id="step-call",
        tool_step_id="step-review",
    )

    assert outcome is not None
    assert outcome.aligned is False
    assert outcome.mode == "step_back"
    assert outcome.boundary_seq == 12
    assert outcome.hidden_step_ids == ["step-call", "step-review"]


def test_strip_system_review_notices() -> None:
    content = "result\n\n<system-review>\ncheck\n</system-review>"

    assert strip_system_review_notices(content) == "result"
```

- [ ] **Step 2: Run trajectory tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_introspect_trajectory.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agiwo.agent.introspect.trajectory'`.

- [ ] **Step 3: Implement `trajectory.py`**

Create `agiwo/agent/introspect/trajectory.py` with these public functions:

```python
"""Trajectory introspection trigger and outcome rules."""

import re

from agiwo.agent.introspect.models import (
    GoalState,
    IntrospectionNotice,
    IntrospectionOutcome,
    IntrospectionState,
    IntrospectionTriggerReason,
    Milestone,
)
from agiwo.tool.base import ToolResult

_SYSTEM_REVIEW_BLOCK_RE = re.compile(
    r"\n*<system-review>\s*.*?\s*</system-review>\s*",
    re.DOTALL,
)


def _build_review_notice(
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
    inner_text = (
        f"{milestone_text}\n\n"
        f"Trigger: {trigger_reason}\n"
        f"Steps since last review: {step_count}\n"
    )
    if review_advice:
        inner_text += f"Hook advice: {review_advice}\n"
    inner_text += (
        "\n"
        "Question: Do your recent steps meaningfully advance the current goal?\n"
        "If not, use review_trajectory with aligned=false and a concise "
        "experience summary. If aligned, use review_trajectory with aligned=true."
    )
    return f"\n\n<system-review>\n{inner_text}\n</system-review>"


def append_system_review_notice(
    content: str,
    milestone: Milestone | None,
    step_count: int,
    *,
    trigger_reason: str,
    review_advice: str | None = None,
) -> str:
    return content + _build_review_notice(
        milestone,
        step_count,
        trigger_reason=trigger_reason,
        review_advice=review_advice,
    )


def strip_system_review_notices(content: str) -> str:
    return _SYSTEM_REVIEW_BLOCK_RE.sub("", content).rstrip()


def maybe_build_introspection_notice(
    result: ToolResult,
    goal: GoalState,
    state: IntrospectionState,
    *,
    step_interval: int,
    review_on_error: bool,
    error_threshold: int = 2,
) -> IntrospectionNotice | None:
    if result.tool_name == "review_trajectory":
        return None
    state.review_count_since_boundary += 1
    if result.is_success:
        state.consecutive_errors = 0
    else:
        state.consecutive_errors += 1

    trigger_reason: IntrospectionTriggerReason | None = None
    if state.pending_milestone_switch and result.tool_name != "declare_milestones":
        trigger_reason = "milestone_switch"
    elif (
        review_on_error
        and not result.is_success
        and state.consecutive_errors >= error_threshold
    ):
        trigger_reason = "consecutive_errors"
    elif state.review_count_since_boundary >= step_interval:
        trigger_reason = "step_interval"

    if trigger_reason is None:
        return None
    state.pending_milestone_switch = False
    milestone = goal.active_milestone
    return IntrospectionNotice(
        content=append_system_review_notice(
            result.content or "",
            milestone,
            state.review_count_since_boundary,
            trigger_reason=trigger_reason,
        ),
        active_milestone=milestone,
        step_count=state.review_count_since_boundary,
        trigger_reason=trigger_reason,
    )


def parse_introspection_outcome(
    result: ToolResult,
    goal: GoalState,
    *,
    current_seq: int,
    assistant_step_id: str | None,
    tool_step_id: str | None,
) -> IntrospectionOutcome | None:
    if result.tool_name != "review_trajectory" or not result.is_success:
        return None
    output = result.output if isinstance(result.output, dict) else {}
    aligned = output.get("aligned")
    experience_value = output.get("experience")
    experience = experience_value if isinstance(experience_value, str) else None
    if aligned is True:
        mode = "metadata_only"
    elif aligned is False:
        mode = "step_back"
        experience = experience or (result.content or "")
    else:
        mode = "metadata_only"
    return IntrospectionOutcome(
        aligned=aligned if isinstance(aligned, bool) else None,
        mode=mode,
        boundary_seq=current_seq,
        experience=experience,
        active_milestone_id=goal.active_milestone_id,
        review_tool_call_id=result.tool_call_id or None,
        review_step_id=tool_step_id,
        hidden_step_ids=[
            step_id
            for step_id in (assistant_step_id, tool_step_id)
            if step_id is not None
        ],
    )


__all__ = [
    "append_system_review_notice",
    "maybe_build_introspection_notice",
    "parse_introspection_outcome",
    "strip_system_review_notices",
]
```

- [ ] **Step 4: Run trajectory tests**

Run:

```bash
uv run pytest tests/agent/test_introspect_trajectory.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/introspect/trajectory.py tests/agent/test_introspect_trajectory.py
git commit -m "feat: add trajectory introspection rules"
```

## Task 4: Implement Context Repair Planning

**Files:**
- Create: `agiwo/agent/introspect/repair.py`
- Test: `tests/agent/test_introspect_repair.py`

- [ ] **Step 1: Write failing repair tests**

Create `tests/agent/test_introspect_repair.py`:

```python
from agiwo.agent.introspect.models import IntrospectionOutcome
from agiwo.agent.introspect.repair import build_context_repair_plan


def test_step_back_repairs_only_after_previous_boundary() -> None:
    messages = [
        {"role": "tool", "tool_call_id": "tc-old", "content": "old", "_sequence": 2},
        {"role": "tool", "tool_call_id": "tc-new", "content": "new", "_sequence": 5},
        {
            "role": "tool",
            "tool_call_id": "tc-review",
            "content": "review",
            "_sequence": 6,
        },
    ]
    outcome = IntrospectionOutcome(
        aligned=False,
        mode="step_back",
        boundary_seq=6,
        experience="new search drifted",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
    )

    plan = build_context_repair_plan(
        messages,
        outcome,
        previous_boundary_seq=3,
        step_lookup={"tc-new": {"id": "step-new", "sequence": 5}},
    )

    assert plan is not None
    assert plan.start_seq == 4
    assert plan.end_seq == 5
    assert [(u.tool_call_id, u.content) for u in plan.content_updates] == [
        ("tc-new", "[EXPERIENCE] new search drifted")
    ]


def test_aligned_review_cleans_prompt_notice() -> None:
    messages = [
        {
            "role": "tool",
            "tool_call_id": "tc-search",
            "content": "result\n<system-review>check</system-review>",
            "_sequence": 4,
        }
    ]
    outcome = IntrospectionOutcome(
        aligned=True,
        mode="metadata_only",
        boundary_seq=5,
        review_tool_call_id="tc-review",
        review_step_id="step-review",
    )

    plan = build_context_repair_plan(
        messages,
        outcome,
        previous_boundary_seq=0,
        step_lookup={"tc-search": {"id": "step-search", "sequence": 4}},
    )

    assert plan is not None
    assert plan.content_updates[0].content == "result"
    assert plan.notice_cleaned_step_ids == ["step-search"]
```

- [ ] **Step 2: Run repair tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_introspect_repair.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agiwo.agent.introspect.repair'`.

- [ ] **Step 3: Implement `repair.py`**

Create `agiwo/agent/introspect/repair.py`:

```python
"""Pure context repair planning for introspection outcomes."""

from typing import Any

from agiwo.agent.introspect.models import (
    ContentUpdate,
    ContextRepairPlan,
    IntrospectionOutcome,
)
from agiwo.agent.introspect.trajectory import strip_system_review_notices


def _step_id_for_tool_call(
    tool_call_id: str,
    step_lookup: dict[str, dict[str, Any]],
) -> str:
    info = step_lookup.get(tool_call_id)
    step_id = info.get("id", "") if info is not None else ""
    return step_id if isinstance(step_id, str) else ""


def build_context_repair_plan(
    messages: list[dict[str, Any]],
    outcome: IntrospectionOutcome,
    *,
    previous_boundary_seq: int,
    step_lookup: dict[str, dict[str, Any]],
) -> ContextRepairPlan | None:
    updates: list[ContentUpdate] = []
    cleaned_step_ids: list[str] = []
    start_seq = previous_boundary_seq + 1
    end_seq = max(outcome.boundary_seq - 1, previous_boundary_seq)

    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        if outcome.review_tool_call_id and tool_call_id == outcome.review_tool_call_id:
            continue
        sequence = message.get("_sequence", 0)
        if not isinstance(sequence, int):
            sequence = 0
        content = message.get("content")
        if not isinstance(content, str) or not content:
            continue
        step_id = _step_id_for_tool_call(tool_call_id, step_lookup)
        if not step_id:
            continue
        if outcome.mode == "step_back" and previous_boundary_seq < sequence < outcome.boundary_seq:
            updates.append(
                ContentUpdate(
                    step_id=step_id,
                    tool_call_id=tool_call_id,
                    content=f"[EXPERIENCE] {outcome.experience or ''}",
                )
            )
            continue
        if "<system-review>" in content:
            updates.append(
                ContentUpdate(
                    step_id=step_id,
                    tool_call_id=tool_call_id,
                    content=strip_system_review_notices(content),
                )
            )
            cleaned_step_ids.append(step_id)

    if not updates:
        return None
    return ContextRepairPlan(
        mode="step_back",
        start_seq=start_seq,
        end_seq=end_seq,
        experience=outcome.experience or "",
        content_updates=updates,
        notice_cleaned_step_ids=cleaned_step_ids,
    )


__all__ = ["build_context_repair_plan"]
```

- [ ] **Step 4: Run repair tests**

Run:

```bash
uv run pytest tests/agent/test_introspect_repair.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/introspect/repair.py tests/agent/test_introspect_repair.py
git commit -m "feat: add introspect context repair planning"
```

## Task 5: Replace Review RunLog Facts With Introspect Facts

**Files:**
- Modify: `agiwo/agent/models/log.py`
- Modify: `agiwo/agent/storage/serialization.py`
- Test: `tests/agent/test_introspect_run_log_facts.py`

- [ ] **Step 1: Write failing fact serialization tests**

Create `tests/agent/test_introspect_run_log_facts.py`:

```python
from agiwo.agent.introspect.models import Milestone
from agiwo.agent.models.log import (
    ContextRepairApplied,
    GoalMilestonesUpdated,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    RunLogEntryKind,
)
from agiwo.agent.storage.serialization import (
    deserialize_run_log_entry_from_storage,
    serialize_run_log_entry_for_storage,
)


def test_goal_milestones_updated_round_trips() -> None:
    entry = GoalMilestonesUpdated(
        sequence=1,
        session_id="sess",
        run_id="run",
        agent_id="agent",
        milestones=[Milestone(id="inspect", description="Inspect", status="active")],
        active_milestone_id="inspect",
        source_tool_call_id="tc",
        source_step_id="step",
        reason="declared",
    )

    payload = serialize_run_log_entry_for_storage(entry)
    restored = deserialize_run_log_entry_from_storage(payload)

    assert payload["kind"] == RunLogEntryKind.GOAL_MILESTONES_UPDATED.value
    assert isinstance(restored, GoalMilestonesUpdated)
    assert restored.milestones[0].id == "inspect"


def test_introspection_outcome_round_trips_boundary_and_repair_range() -> None:
    entry = IntrospectionOutcomeRecorded(
        sequence=2,
        session_id="sess",
        run_id="run",
        agent_id="agent",
        aligned=False,
        mode="step_back",
        boundary_seq=12,
        repair_start_seq=4,
        repair_end_seq=11,
        condensed_step_ids=["step-search"],
    )

    restored = deserialize_run_log_entry_from_storage(
        serialize_run_log_entry_for_storage(entry)
    )

    assert isinstance(restored, IntrospectionOutcomeRecorded)
    assert restored.boundary_seq == 12
    assert restored.repair_start_seq == 4


def test_context_repair_applied_round_trips() -> None:
    entry = ContextRepairApplied(
        sequence=3,
        session_id="sess",
        run_id="run",
        agent_id="agent",
        mode="step_back",
        affected_count=1,
        start_seq=4,
        end_seq=11,
        experience="drifted",
    )

    restored = deserialize_run_log_entry_from_storage(
        serialize_run_log_entry_for_storage(entry)
    )

    assert isinstance(restored, ContextRepairApplied)
    assert restored.mode == "step_back"
    assert restored.affected_count == 1


def test_introspection_trigger_round_trips() -> None:
    entry = IntrospectionTriggered(
        sequence=4,
        session_id="sess",
        run_id="run",
        agent_id="agent",
        trigger_reason="step_interval",
        active_milestone_id="inspect",
        review_count_since_boundary=8,
        trigger_tool_call_id="tc",
        trigger_tool_step_id="step",
        notice_step_id="step",
    )

    restored = deserialize_run_log_entry_from_storage(
        serialize_run_log_entry_for_storage(entry)
    )

    assert isinstance(restored, IntrospectionTriggered)
    assert restored.review_count_since_boundary == 8
```

- [ ] **Step 2: Run fact tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_introspect_run_log_facts.py -v
```

Expected: FAIL because the new RunLog classes are not defined.

- [ ] **Step 3: Modify `models/log.py`**

In `agiwo/agent/models/log.py`:

```python
from agiwo.agent.introspect.models import Milestone
```

Replace review enum members with:

```python
GOAL_MILESTONES_UPDATED = "goal_milestones_updated"
INTROSPECTION_TRIGGERED = "introspection_triggered"
INTROSPECTION_CHECKPOINT_RECORDED = "introspection_checkpoint_recorded"
INTROSPECTION_OUTCOME_RECORDED = "introspection_outcome_recorded"
CONTEXT_REPAIR_APPLIED = "context_repair_applied"
```

Add dataclasses:

```python
@dataclass(frozen=True, kw_only=True)
class GoalMilestonesUpdated(RunLogEntry):
    milestones: list[Milestone] = field(default_factory=list)
    active_milestone_id: str | None = None
    source_tool_call_id: str | None = None
    source_step_id: str | None = None
    reason: Literal["declared", "updated", "completed", "activated"] = "updated"
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.GOAL_MILESTONES_UPDATED
    )


@dataclass(frozen=True, kw_only=True)
class IntrospectionTriggered(RunLogEntry):
    trigger_reason: Literal["step_interval", "consecutive_errors", "milestone_switch"]
    active_milestone_id: str | None = None
    review_count_since_boundary: int = 0
    trigger_tool_call_id: str | None = None
    trigger_tool_step_id: str | None = None
    notice_step_id: str | None = None
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.INTROSPECTION_TRIGGERED
    )


@dataclass(frozen=True, kw_only=True)
class IntrospectionCheckpointRecorded(RunLogEntry):
    checkpoint_seq: int
    milestone_id: str | None = None
    review_tool_call_id: str | None = None
    review_step_id: str | None = None
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.INTROSPECTION_CHECKPOINT_RECORDED
    )


@dataclass(frozen=True, kw_only=True)
class IntrospectionOutcomeRecorded(RunLogEntry):
    mode: Literal["metadata_only", "step_back"]
    boundary_seq: int
    aligned: bool | None = None
    experience: str | None = None
    active_milestone_id: str | None = None
    review_tool_call_id: str | None = None
    review_step_id: str | None = None
    hidden_step_ids: list[str] = field(default_factory=list)
    notice_cleaned_step_ids: list[str] = field(default_factory=list)
    condensed_step_ids: list[str] = field(default_factory=list)
    repair_start_seq: int | None = None
    repair_end_seq: int | None = None
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.INTROSPECTION_OUTCOME_RECORDED
    )


@dataclass(frozen=True, kw_only=True)
class ContextRepairApplied(RunLogEntry):
    mode: Literal["step_back"]
    affected_count: int
    start_seq: int
    end_seq: int
    experience: str
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.CONTEXT_REPAIR_APPLIED
    )
```

Remove the old `ReviewMilestonesUpdated`, `ReviewTriggerDecided`,
`ReviewCheckpointRecorded`, `ReviewOutcomeRecorded`, and `StepBackApplied`
exports after all direct imports are migrated in later tasks.

- [ ] **Step 4: Modify serialization mapping**

In `agiwo/agent/storage/serialization.py`, import the new classes and add them
to the kind-to-class mapping:

```python
RunLogEntryKind.GOAL_MILESTONES_UPDATED: GoalMilestonesUpdated,
RunLogEntryKind.INTROSPECTION_TRIGGERED: IntrospectionTriggered,
RunLogEntryKind.INTROSPECTION_CHECKPOINT_RECORDED: IntrospectionCheckpointRecorded,
RunLogEntryKind.INTROSPECTION_OUTCOME_RECORDED: IntrospectionOutcomeRecorded,
RunLogEntryKind.CONTEXT_REPAIR_APPLIED: ContextRepairApplied,
```

Update milestone hydration to instantiate `agiwo.agent.introspect.models.Milestone`.

- [ ] **Step 5: Run fact tests**

Run:

```bash
uv run pytest tests/agent/test_introspect_run_log_facts.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agiwo/agent/models/log.py agiwo/agent/storage/serialization.py tests/agent/test_introspect_run_log_facts.py
git commit -m "feat: add introspect run log facts"
```

## Task 6: Add RunStateWriter Introspect Commit Methods

**Files:**
- Modify: `agiwo/agent/runtime/state_writer.py`
- Test: `tests/agent/test_introspect_state_writer.py`

- [ ] **Step 1: Write failing writer tests**

Create `tests/agent/test_introspect_state_writer.py`:

```python
import pytest

from agiwo.agent.introspect.models import Milestone
from agiwo.agent.models.log import (
    GoalMilestonesUpdated,
    IntrospectionOutcomeRecorded,
)
from agiwo.agent.models.run import RunIdentity
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_writer import RunStateWriter
from agiwo.agent.storage.base import InMemoryRunLogStorage


@pytest.mark.asyncio
async def test_writer_records_goal_update() -> None:
    storage = InMemoryRunLogStorage()
    context = RunContext(
        identity=RunIdentity(run_id="run", agent_id="agent", agent_name="agent"),
        session_runtime=SessionRuntime(session_id="sess", run_log_storage=storage),
    )
    writer = RunStateWriter(context)

    entries = await writer.record_goal_milestones_updated(
        milestones=[Milestone(id="inspect", description="Inspect", status="active")],
        active_milestone_id="inspect",
        source_tool_call_id="tc",
        source_step_id="step",
        reason="declared",
    )

    assert isinstance(entries[0], GoalMilestonesUpdated)
    stored = await storage.list_entries(session_id="sess")
    assert isinstance(stored[0], GoalMilestonesUpdated)


@pytest.mark.asyncio
async def test_writer_records_introspection_outcome_boundary() -> None:
    storage = InMemoryRunLogStorage()
    context = RunContext(
        identity=RunIdentity(run_id="run", agent_id="agent", agent_name="agent"),
        session_runtime=SessionRuntime(session_id="sess", run_log_storage=storage),
    )
    writer = RunStateWriter(context)

    entries = await writer.record_introspection_outcome_recorded(
        aligned=False,
        mode="step_back",
        experience="drifted",
        active_milestone_id="inspect",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
        hidden_step_ids=["step-call", "step-review"],
        notice_cleaned_step_ids=[],
        condensed_step_ids=["step-search"],
        boundary_seq=12,
        repair_start_seq=4,
        repair_end_seq=11,
    )

    assert isinstance(entries[0], IntrospectionOutcomeRecorded)
    assert entries[0].boundary_seq == 12
```

- [ ] **Step 2: Run writer tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_introspect_state_writer.py -v
```

Expected: FAIL because writer methods are missing.

- [ ] **Step 3: Add writer methods and builders**

In `agiwo/agent/runtime/state_writer.py`, add methods:

```python
async def record_goal_milestones_updated(
    self,
    *,
    milestones: list[Milestone],
    active_milestone_id: str | None,
    source_tool_call_id: str | None,
    source_step_id: str | None,
    reason: Literal["declared", "updated", "completed", "activated"],
) -> list[object]:
    return await self.append_entries(
        [
            build_goal_milestones_updated_entry(
                self._state,
                sequence=await self._state.session_runtime.allocate_sequence(),
                milestones=milestones,
                active_milestone_id=active_milestone_id,
                source_tool_call_id=source_tool_call_id,
                source_step_id=source_step_id,
                reason=reason,
            )
        ]
    )


async def record_introspection_triggered(
    self,
    *,
    trigger_reason: Literal["step_interval", "consecutive_errors", "milestone_switch"],
    active_milestone_id: str | None,
    review_count_since_boundary: int,
    trigger_tool_call_id: str | None,
    trigger_tool_step_id: str | None,
    notice_step_id: str | None,
) -> list[object]:
    return await self.append_entries(
        [
            build_introspection_triggered_entry(
                self._state,
                sequence=await self._state.session_runtime.allocate_sequence(),
                trigger_reason=trigger_reason,
                active_milestone_id=active_milestone_id,
                review_count_since_boundary=review_count_since_boundary,
                trigger_tool_call_id=trigger_tool_call_id,
                trigger_tool_step_id=trigger_tool_step_id,
                notice_step_id=notice_step_id,
            )
        ]
    )


async def record_introspection_checkpoint_recorded(
    self,
    *,
    checkpoint_seq: int,
    milestone_id: str | None,
    review_tool_call_id: str | None,
    review_step_id: str | None,
) -> list[object]:
    return await self.append_entries(
        [
            build_introspection_checkpoint_recorded_entry(
                self._state,
                sequence=await self._state.session_runtime.allocate_sequence(),
                checkpoint_seq=checkpoint_seq,
                milestone_id=milestone_id,
                review_tool_call_id=review_tool_call_id,
                review_step_id=review_step_id,
            )
        ]
    )


async def record_introspection_outcome_recorded(
    self,
    *,
    aligned: bool | None,
    mode: Literal["metadata_only", "step_back"],
    experience: str | None,
    active_milestone_id: str | None,
    review_tool_call_id: str | None,
    review_step_id: str | None,
    hidden_step_ids: list[str],
    notice_cleaned_step_ids: list[str],
    condensed_step_ids: list[str],
    boundary_seq: int,
    repair_start_seq: int | None,
    repair_end_seq: int | None,
) -> list[object]:
    return await self.append_entries(
        [
            build_introspection_outcome_recorded_entry(
                self._state,
                sequence=await self._state.session_runtime.allocate_sequence(),
                aligned=aligned,
                mode=mode,
                experience=experience,
                active_milestone_id=active_milestone_id,
                review_tool_call_id=review_tool_call_id,
                review_step_id=review_step_id,
                hidden_step_ids=hidden_step_ids,
                notice_cleaned_step_ids=notice_cleaned_step_ids,
                condensed_step_ids=condensed_step_ids,
                boundary_seq=boundary_seq,
                repair_start_seq=repair_start_seq,
                repair_end_seq=repair_end_seq,
            )
        ]
    )


async def record_context_repair_applied(
    self,
    *,
    mode: Literal["step_back"],
    affected_count: int,
    start_seq: int,
    end_seq: int,
    experience: str,
) -> list[object]:
    return await self.append_entries(
        [
            build_context_repair_applied_entry(
                self._state,
                sequence=await self._state.session_runtime.allocate_sequence(),
                mode=mode,
                affected_count=affected_count,
                start_seq=start_seq,
                end_seq=end_seq,
                experience=experience,
            )
        ]
    )


async def record_step_condensed_content_updated(
    self,
    *,
    step_id: str,
    condensed_content: str,
) -> list[object]:
    return await self.append_entries(
        [
            build_step_condensed_content_updated_entry(
                self._state,
                sequence=await self._state.session_runtime.allocate_sequence(),
                step_id=step_id,
                condensed_content=condensed_content,
            )
        ]
    )
```

Use existing `record_review_*` methods as the mechanical template, but build the
new dataclasses from Task 5. `record_step_condensed_content_updated` must append
`StepCondensedContentUpdated` through `append_entries` instead of calling storage
directly.

- [ ] **Step 4: Run writer tests**

Run:

```bash
uv run pytest tests/agent/test_introspect_state_writer.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/runtime/state_writer.py tests/agent/test_introspect_state_writer.py
git commit -m "feat: add introspect state writer methods"
```

## Task 7: Implement Replay For Goal And Introspection State

**Files:**
- Create: `agiwo/agent/introspect/replay.py`
- Modify: `agiwo/agent/models/run.py`
- Modify: `agiwo/agent/run_bootstrap.py`
- Test: `tests/agent/test_introspect_replay.py`
- Test: `tests/agent/test_introspect_bootstrap.py`

- [ ] **Step 1: Write replay tests**

Create `tests/agent/test_introspect_replay.py`:

```python
from agiwo.agent.introspect.models import Milestone
from agiwo.agent.introspect.replay import build_introspect_state_from_entries
from agiwo.agent.models.log import (
    GoalMilestonesUpdated,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    ToolStepCommitted,
)
from agiwo.agent.models.step import MessageRole


def test_replay_restores_goal_and_counter_since_boundary() -> None:
    goal, introspection = build_introspect_state_from_entries(
        [
            GoalMilestonesUpdated(
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
            IntrospectionOutcomeRecorded(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                aligned=True,
                mode="metadata_only",
                boundary_seq=2,
            ),
            ToolStepCommitted(
                sequence=3,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                step_id="step-search",
                role=MessageRole.TOOL,
                tool_call_id="tc-search",
                name="search",
                content="result",
            ),
        ]
    )

    assert goal.active_milestone_id == "inspect"
    assert introspection.last_boundary_seq == 2
    assert introspection.review_count_since_boundary == 1


def test_replay_tracks_pending_trigger_until_outcome() -> None:
    _, introspection = build_introspect_state_from_entries(
        [
            IntrospectionTriggered(
                sequence=1,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                trigger_reason="step_interval",
                review_count_since_boundary=8,
            )
        ]
    )

    assert introspection.pending_trigger is not None
    assert introspection.pending_trigger.trigger_reason == "step_interval"
```

- [ ] **Step 2: Run replay tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_introspect_replay.py -v
```

Expected: FAIL because `introspect.replay` is missing.

- [ ] **Step 3: Implement replay**

Create `agiwo/agent/introspect/replay.py`:

```python
"""Replay helpers for goal and introspection state."""

from collections.abc import Iterable

from agiwo.agent.introspect.models import (
    GoalState,
    IntrospectionCheckpoint,
    IntrospectionState,
    PendingIntrospectionNotice,
)
from agiwo.agent.models.log import (
    GoalMilestonesUpdated,
    IntrospectionCheckpointRecorded,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    RunLogEntry,
    ToolStepCommitted,
)


def build_introspect_state_from_entries(
    entries: Iterable[RunLogEntry],
) -> tuple[GoalState, IntrospectionState]:
    goal = GoalState()
    introspection = IntrospectionState()
    for entry in sorted(entries, key=lambda item: item.sequence):
        if isinstance(entry, GoalMilestonesUpdated):
            goal.milestones = list(entry.milestones)
            goal.active_milestone_id = entry.active_milestone_id
            if entry.reason in {"completed", "activated"}:
                introspection.pending_milestone_switch = True
            continue
        if isinstance(entry, IntrospectionTriggered):
            introspection.pending_milestone_switch = False
            introspection.pending_trigger = PendingIntrospectionNotice(
                trigger_reason=entry.trigger_reason,
                active_milestone_id=entry.active_milestone_id,
                review_count_since_boundary=entry.review_count_since_boundary,
                trigger_tool_call_id=entry.trigger_tool_call_id,
                trigger_tool_step_id=entry.trigger_tool_step_id,
                notice_step_id=entry.notice_step_id,
            )
            continue
        if isinstance(entry, IntrospectionCheckpointRecorded):
            introspection.latest_aligned_checkpoint = IntrospectionCheckpoint(
                seq=entry.checkpoint_seq,
                milestone_id=entry.milestone_id or "",
                confirmed_at=entry.created_at,
            )
            continue
        if isinstance(entry, IntrospectionOutcomeRecorded):
            introspection.pending_trigger = None
            introspection.pending_milestone_switch = False
            introspection.last_boundary_seq = entry.boundary_seq
            introspection.review_count_since_boundary = 0
            continue
        if isinstance(entry, ToolStepCommitted):
            if entry.name == "review_trajectory":
                continue
            if entry.sequence > introspection.last_boundary_seq:
                introspection.review_count_since_boundary += 1
    return goal, introspection


__all__ = ["build_introspect_state_from_entries"]
```

- [ ] **Step 4: Update `RunLedger`**

In `agiwo/agent/models/run.py`, replace `review: ReviewState` with:

```python
from agiwo.agent.introspect.models import GoalState, IntrospectionState

goal: GoalState = field(default_factory=GoalState)
introspection: IntrospectionState = field(default_factory=IntrospectionState)
```

- [ ] **Step 5: Update bootstrap restore**

In `agiwo/agent/run_bootstrap.py`, replace review restore with:

```python
from agiwo.agent.introspect.replay import build_introspect_state_from_entries


async def _restore_introspect_state(context: RunContext) -> None:
    entries = await context.session_runtime.list_run_log_entries(
        agent_id=context.agent_id,
        limit=100_000,
    )
    context.ledger.goal, context.ledger.introspection = (
        build_introspect_state_from_entries(entries)
    )
```

Call `_restore_introspect_state(context)` from `prepare_run_context`.

- [ ] **Step 6: Run replay and bootstrap tests**

Run:

```bash
uv run pytest tests/agent/test_introspect_replay.py tests/agent/test_review_bootstrap.py -v
```

Expected: introspect replay tests PASS; old bootstrap test fails until renamed or updated.

- [ ] **Step 7: Rename/update bootstrap test**

Rename `tests/agent/test_review_bootstrap.py` to `tests/agent/test_introspect_bootstrap.py` and update imports/classes from review facts to new introspect facts. Assertions must check:

```python
assert context.ledger.goal.active_milestone_id == "inspect"
assert context.ledger.introspection.last_boundary_seq == 2
assert context.ledger.introspection.review_count_since_boundary == 1
```

- [ ] **Step 8: Run updated tests**

Run:

```bash
uv run pytest tests/agent/test_introspect_replay.py tests/agent/test_introspect_bootstrap.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add agiwo/agent/introspect/replay.py agiwo/agent/models/run.py agiwo/agent/run_bootstrap.py tests/agent/test_introspect_replay.py tests/agent/test_introspect_bootstrap.py
git rm tests/agent/test_review_bootstrap.py
git commit -m "feat: replay introspect runtime state"
```

## Task 8: Add Apply Helpers And Refactor Tool Batch Flow

**Files:**
- Create: `agiwo/agent/introspect/apply.py`
- Modify: `agiwo/agent/run_tool_batch.py`
- Test: `tests/agent/test_run_tool_batch.py`

- [ ] **Step 1: Add an integration test for projection**

In `tests/agent/test_run_tool_batch.py`, replace review fact imports with new introspect fact imports and add:

```python
async def test_execute_tool_batch_cycle_projects_introspection_trigger(monkeypatch):
    projected = []

    async def fake_execute_tool_batch(*args, **kwargs):
        del args, kwargs
        return [
            ToolResult.success(
                tool_name="search",
                tool_call_id="tc_search",
                content="Found results",
                output={},
            )
        ]

    monkeypatch.setattr(
        run_tool_batch_module, "execute_tool_batch", fake_execute_tool_batch
    )
    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(session_id="sess-1", run_log_storage=storage)
    original_project = session_runtime.project_run_log_entries

    async def spy_project(entries, **kwargs):
        projected.extend(entries)
        await original_project(entries, **kwargs)

    session_runtime.project_run_log_entries = spy_project
    hooks = _FakeHooks()
    ledger = RunLedger()
    ledger.goal.milestones = [
        Milestone(id="locate", description="Locate the auth bug", status="active")
    ]
    ledger.goal.active_milestone_id = "locate"
    context = _FakeContext(
        config=AgentOptions(enable_goal_directed_review=True, review_step_interval=1),
        ledger=ledger,
        hooks=hooks,
        session_runtime=session_runtime,
    )

    async def commit_step(step):
        return await _commit_step_to_ledger_and_storage(context, step)

    async def set_termination_reason(reason, tool_name):
        del reason, tool_name

    await run_tool_batch_module.execute_tool_batch_cycle(
        context=context,
        runtime=_FakeRuntime(tools_map=_review_tools_map()),
        tool_calls=[
            {"id": "tc_search", "type": "function", "function": {"name": "search"}}
        ],
        assistant_step_id="assistant-step",
        set_termination_reason=set_termination_reason,
        commit_step=commit_step,
    )

    assert any(entry.__class__.__name__ == "IntrospectionTriggered" for entry in projected)
```

- [ ] **Step 2: Run integration test and verify it fails**

Run:

```bash
uv run pytest tests/agent/test_run_tool_batch.py::test_execute_tool_batch_cycle_projects_introspection_trigger -v
```

Expected: FAIL because `run_tool_batch.py` still uses `ReviewBatch`.

- [ ] **Step 3: Implement `apply.py`**

Create `agiwo/agent/introspect/apply.py` with helpers that call writer methods
and immediately project returned entries:

```python
"""Commit helpers for introspect facts and live context updates."""

from agiwo.agent.introspect.models import (
    ContextRepairPlan,
    GoalUpdate,
    IntrospectionNotice,
    IntrospectionOutcome,
)
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_writer import RunStateWriter


async def _project(context: RunContext, entries: list[object]) -> None:
    await context.session_runtime.project_run_log_entries(
        entries,
        run_id=context.run_id,
        agent_id=context.agent_id,
        parent_run_id=context.parent_run_id,
        depth=context.depth,
    )


async def commit_goal_update(
    context: RunContext,
    writer: RunStateWriter,
    update: GoalUpdate,
    *,
    source_step_id: str,
) -> None:
    entries = await writer.record_goal_milestones_updated(
        milestones=update.milestones,
        active_milestone_id=update.active_milestone_id,
        source_tool_call_id=update.source_tool_call_id,
        source_step_id=source_step_id,
        reason=update.reason,
    )
    await _project(context, entries)


async def commit_introspection_trigger(
    context: RunContext,
    writer: RunStateWriter,
    notice: IntrospectionNotice,
    *,
    trigger_tool_call_id: str | None,
    trigger_tool_step_id: str,
) -> None:
    entries = await writer.record_introspection_triggered(
        trigger_reason=notice.trigger_reason,
        active_milestone_id=notice.active_milestone.id
        if notice.active_milestone is not None
        else None,
        review_count_since_boundary=notice.step_count,
        trigger_tool_call_id=trigger_tool_call_id,
        trigger_tool_step_id=trigger_tool_step_id,
        notice_step_id=trigger_tool_step_id,
    )
    await _project(context, entries)


async def commit_introspection_outcome(
    context: RunContext,
    writer: RunStateWriter,
    outcome: IntrospectionOutcome,
) -> None:
    if outcome.hidden_step_ids:
        await _project(
            context,
            await writer.record_context_steps_hidden(step_ids=outcome.hidden_step_ids),
        )
    if outcome.aligned is True:
        await _project(
            context,
            await writer.record_introspection_checkpoint_recorded(
                checkpoint_seq=outcome.boundary_seq,
                milestone_id=outcome.active_milestone_id,
                review_tool_call_id=outcome.review_tool_call_id,
                review_step_id=outcome.review_step_id,
            ),
        )
    repair = outcome.repair_plan
    await _project(
        context,
        await writer.record_introspection_outcome_recorded(
            aligned=outcome.aligned,
            mode=outcome.mode,
            experience=outcome.experience,
            active_milestone_id=outcome.active_milestone_id,
            review_tool_call_id=outcome.review_tool_call_id,
            review_step_id=outcome.review_step_id,
            hidden_step_ids=outcome.hidden_step_ids,
            notice_cleaned_step_ids=repair.notice_cleaned_step_ids if repair else [],
            condensed_step_ids=repair.condensed_step_ids if repair else [],
            boundary_seq=outcome.boundary_seq,
            repair_start_seq=repair.start_seq if repair else None,
            repair_end_seq=repair.end_seq if repair else None,
        ),
    )


async def commit_context_repair(
    context: RunContext,
    writer: RunStateWriter,
    plan: ContextRepairPlan,
) -> None:
    for update in plan.content_updates:
        await _project(
            context,
            await writer.record_step_condensed_content_updated(
                step_id=update.step_id,
                condensed_content=update.content,
            ),
        )
        for message in context.ledger.messages:
            if (
                message.get("role") == "tool"
                and message.get("tool_call_id") == update.tool_call_id
            ):
                message["content"] = update.content
                break
    await _project(
        context,
        await writer.record_context_repair_applied(
            mode=plan.mode,
            affected_count=plan.affected_count,
            start_seq=plan.start_seq,
            end_seq=plan.end_seq,
            experience=plan.experience,
        ),
    )
```

- [ ] **Step 4: Refactor `run_tool_batch.py`**

Replace `ReviewBatch` usage with:

```python
goal_update = handle_goal_tool_result(
    result,
    context.ledger.goal,
    current_seq=seq,
)
notice = maybe_build_introspection_notice(
    result,
    context.ledger.goal,
    context.ledger.introspection,
    step_interval=context.config.review_step_interval,
    review_on_error=context.config.review_on_error,
)
outcome = parse_introspection_outcome(
    result,
    context.ledger.goal,
    current_seq=committed_step.sequence,
    assistant_step_id=assistant_step_id,
    tool_step_id=committed_step.id,
)
repair_plan = build_context_repair_plan(
    context.ledger.messages,
    outcome,
    previous_boundary_seq=context.ledger.introspection.last_boundary_seq,
    step_lookup=step_lookup,
)
await commit_goal_update(
    context,
    writer,
    goal_update,
    source_step_id=committed_step.id,
)
await commit_introspection_trigger(
    context,
    writer,
    notice,
    trigger_tool_call_id=result.tool_call_id or None,
    trigger_tool_step_id=committed_step.id,
)
await commit_introspection_outcome(context, writer, outcome)
await commit_context_repair(context, writer, repair_plan)
```

Keep `_remove_review_tool_call` for this task, but rename it to
`_remove_introspection_tool_call` and call it after outcome application.

- [ ] **Step 5: Run tool batch tests**

Run:

```bash
uv run pytest tests/agent/test_run_tool_batch.py -v
```

Expected: PASS after imports/assertions are updated from review names to introspect names.

- [ ] **Step 6: Commit**

```bash
git add agiwo/agent/introspect/apply.py agiwo/agent/run_tool_batch.py tests/agent/test_run_tool_batch.py
git commit -m "feat: route tool batches through introspect"
```

## Task 9: Update Trace Writer And Console Observability

**Files:**
- Modify: `agiwo/agent/trace_writer.py`
- Modify: `console/server/services/runtime/runtime_observability.py`
- Modify: `console/tests/test_runtime_observability.py`
- Test: `tests/agent/test_introspect_trace_projection.py`

- [ ] **Step 1: Write trace projection tests**

Create `tests/agent/test_introspect_trace_projection.py`:

```python
from agiwo.agent.introspect.models import Milestone
from agiwo.agent.models.log import (
    ContextRepairApplied,
    GoalMilestonesUpdated,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    RunStarted,
)
from agiwo.agent.trace_writer import AgentTraceCollector


def test_introspect_facts_project_to_runtime_spans() -> None:
    trace = AgentTraceCollector().build_from_entries(
        [
            RunStarted(
                sequence=1,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                user_input="inspect",
            ),
            GoalMilestonesUpdated(
                sequence=2,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                milestones=[Milestone(id="inspect", description="Inspect", status="active")],
                active_milestone_id="inspect",
                reason="declared",
            ),
            IntrospectionTriggered(
                sequence=3,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                trigger_reason="step_interval",
                review_count_since_boundary=8,
            ),
            IntrospectionOutcomeRecorded(
                sequence=4,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                aligned=False,
                mode="step_back",
                boundary_seq=9,
                condensed_step_ids=["step-search"],
            ),
            ContextRepairApplied(
                sequence=5,
                session_id="sess",
                run_id="run",
                agent_id="agent",
                mode="step_back",
                affected_count=1,
                start_seq=2,
                end_seq=8,
                experience="drifted",
            ),
        ]
    )

    spans_by_name = {span.name: span for span in trace.spans}
    assert "goal_milestones" in spans_by_name
    assert "introspection_trigger" in spans_by_name
    assert "introspection_outcome" in spans_by_name
    assert "context_repair" in spans_by_name
```

- [ ] **Step 2: Run trace test and verify it fails**

Run:

```bash
uv run pytest tests/agent/test_introspect_trace_projection.py -v
```

Expected: FAIL because trace writer does not handle new fact classes.

- [ ] **Step 3: Update `trace_writer.py`**

Replace review runtime span handling with names:

```python
goal_milestones
introspection_trigger
introspection_checkpoint
introspection_outcome
context_repair
```

For `ContextRepairApplied`, attributes must include:

```python
{
    "mode": entry.mode,
    "affected_count": entry.affected_count,
    "start_seq": entry.start_seq,
    "end_seq": entry.end_seq,
    "experience": entry.experience,
}
```

- [ ] **Step 4: Remove Console text parser**

In `console/server/services/runtime/runtime_observability.py`:

- remove `_SYSTEM_REVIEW_BLOCK_RE`, `_TRIGGER_RE`, `_STEPS_RE`, `_MILESTONE_RE`, `_HOOK_ADVICE_RE`, and `_ALIGNED_RE`;
- remove `parse_system_review_notice`;
- remove `_update_review_cycles_from_tool_span`;
- make `build_trace_review_cycles` consume runtime spans named `introspection_trigger`, `introspection_outcome`, `goal_milestones`, and `context_repair`.

- [ ] **Step 5: Update Console tests**

In `console/tests/test_runtime_observability.py`, replace review span fixture names with new names. Assertions must use:

```python
assert cycles[0].trigger_reason == "step_interval"
assert cycles[0].step_back_applied is True
assert cycles[0].affected_count == 1
```

- [ ] **Step 6: Run trace and Console observability tests**

Run:

```bash
uv run pytest tests/agent/test_introspect_trace_projection.py console/tests/test_runtime_observability.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agiwo/agent/trace_writer.py console/server/services/runtime/runtime_observability.py tests/agent/test_introspect_trace_projection.py console/tests/test_runtime_observability.py
git commit -m "feat: project introspect facts to observability"
```

## Task 10: Remove Old Review Module And Update Public References

**Files:**
- Delete: `agiwo/agent/review/`
- Delete: `agiwo/agent/models/review.py`
- Modify: `agiwo/agent/__init__.py`
- Modify: `agiwo/agent/prompt.py`
- Modify: `agiwo/scheduler/runtime_tools.py`
- Modify: tests under `tests/agent/`

- [ ] **Step 1: Search old review imports**

Run:

```bash
rg -n "agiwo\\.agent\\.review|models\\.review|ReviewMilestones|ReviewTrigger|ReviewOutcome|ReviewCheckpoint|StepBackApplied|review_count_since_checkpoint|latest_checkpoint|ledger\\.review" agiwo tests console -S
```

Expected: output lists remaining old references.

- [ ] **Step 2: Replace imports and names**

Use these mappings:

```text
agiwo.agent.review -> agiwo.agent.introspect
agiwo.agent.models.review.Milestone -> agiwo.agent.introspect.models.Milestone
ReviewMilestonesUpdated -> GoalMilestonesUpdated
ReviewTriggerDecided -> IntrospectionTriggered
ReviewCheckpointRecorded -> IntrospectionCheckpointRecorded
ReviewOutcomeRecorded -> IntrospectionOutcomeRecorded
StepBackApplied -> ContextRepairApplied
ledger.review -> ledger.introspection or ledger.goal depending on field
review_count_since_checkpoint -> review_count_since_boundary
latest_checkpoint -> latest_aligned_checkpoint
```

- [ ] **Step 3: Delete old files**

Run:

```bash
mkdir -p trash/agent-review-refactor
mv agiwo/agent/review trash/agent-review-refactor/review
mv agiwo/agent/models/review.py trash/agent-review-refactor/review.py
```

Do not use `rm`; repository guidance sends deleted files to `trash/`.

- [ ] **Step 4: Update prompt wording**

In `agiwo/agent/prompt.py`, keep tool names unchanged but update section title:

```text
## Goal And Trajectory Introspection
```

Keep the behavior instruction that `<system-review>` requires `review_trajectory`.

- [ ] **Step 5: Run old-reference search again**

Run:

```bash
rg -n "agiwo\\.agent\\.review|models\\.review|ReviewMilestones|ReviewTrigger|ReviewOutcome|ReviewCheckpoint|StepBackApplied|ledger\\.review" agiwo tests console -S
```

Expected: no matches.

- [ ] **Step 6: Run focused agent tests**

Run:

```bash
uv run pytest tests/agent/test_introspect_models.py tests/agent/test_introspect_goal.py tests/agent/test_introspect_trajectory.py tests/agent/test_introspect_repair.py tests/agent/test_introspect_replay.py tests/agent/test_run_tool_batch.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agiwo tests trash
git commit -m "refactor: remove review module in favor of introspect"
```

## Task 11: Update AGENTS.md And Final Guards

**Files:**
- Modify: `AGENTS.md`
- Modify: any tests still failing after the focused run

- [ ] **Step 1: Update AGENTS.md**

In `AGENTS.md`, replace:

```text
目标导向 review / step-back 优化收口在 `review/`
```

with:

```text
目标声明、trajectory introspection 与 context repair 收口在 `introspect/`；
`introspect` 通过 first-class RunLog facts 与 replay/trace/Console 交互，
`<system-review>` 只作为 prompt 控制机制，不作为事实来源。
```

- [ ] **Step 2: Run SDK lint gate**

Run:

```bash
uv run python scripts/lint.py ci
```

Expected: PASS.

- [ ] **Step 3: Run affected SDK tests**

Run:

```bash
uv run pytest tests/agent tests/scheduler/test_review_tools.py tests/scheduler/test_runtime_facts.py -v
```

Expected: PASS.

- [ ] **Step 4: Run Console backend tests**

Run:

```bash
uv run python scripts/check.py console-tests
```

Expected: PASS.

- [ ] **Step 5: Search for stale text parser and review module references**

Run:

```bash
rg -n "parse_system_review_notice|_SYSTEM_REVIEW_BLOCK_RE|agiwo\\.agent\\.review|ReviewBatch|StepBackApplied|review_count_since_checkpoint|ledger\\.review" agiwo console tests AGENTS.md -S
```

Expected: no matches, except user-facing prompt text may still contain `<system-review>` and `review_trajectory`.

- [ ] **Step 6: Commit final docs and cleanup**

```bash
git add AGENTS.md agiwo console tests trash
git commit -m "docs: update introspect architecture guidance"
```

## Self-Review Checklist

- Spec coverage:
  - Package layout is covered in Tasks 1-4.
  - RunLog fact renaming is covered in Task 5.
  - Writer/projection path is covered in Tasks 6 and 8.
  - Replay/bootstrap is covered in Task 7.
  - Trace/Console fact-only read models are covered in Task 9.
  - Old module deletion and AGENTS.md update are covered in Tasks 10-11.
- Placeholder scan:
  - The plan intentionally avoids open-ended task text; commands and expected results are specified.
- Type consistency:
  - `GoalState`, `IntrospectionState`, `GoalMilestonesUpdated`,
    `IntrospectionTriggered`, `IntrospectionCheckpointRecorded`,
    `IntrospectionOutcomeRecorded`, and `ContextRepairApplied` are introduced
    before later tasks reference them.
