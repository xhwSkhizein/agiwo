# Goal-Directed Review & StepBack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy token/round-count-based tool-result rewrite flow with a goal-aligned review + step-back mechanism that preserves KV cache via targeted content replacement.

**Architecture:** New `agiwo/agent/review/` package (GoalManager → ReviewEnforcer → StepBackExecutor). System enforces review at checkpoints, agent provides alignment assessment and experience, system executes KV-cache-safe content condensation without message deletion or reordering.

**Tech Stack:** Python 3.10+, dataclasses, asyncio, aiofiles, pytest

**Spec:** `docs/superpowers/specs/2026-04-24-goal-directed-review-and-stepback-design.md`

---

## Phase 1: New Data Models

### Task 1: Add Milestone and ReviewCheckpoint data models

**Files:**
- Create: `agiwo/agent/models/review.py`
- Modify: `agiwo/agent/models/run.py` — replace the legacy review-tracking state with `ReviewState`
- Test: `tests/agent/test_review_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_review_models.py
import pytest
from datetime import datetime
from agiwo.agent.models.review import Milestone, ReviewCheckpoint, ReviewState


class TestMilestone:
    def test_milestone_creation(self):
        m = Milestone(
            id="understand",
            description="Understand session management",
            status="pending",
            declared_at_seq=5,
        )
        assert m.id == "understand"
        assert m.description == "Understand session management"
        assert m.status == "pending"
        assert m.declared_at_seq == 5
        assert m.completed_at_seq is None

    def test_milestone_defaults(self):
        m = Milestone(id="fix", description="Fix the bug", status="pending")
        assert m.declared_at_seq == 0
        assert m.completed_at_seq is None

    def test_milestone_equality(self):
        m1 = Milestone(id="a", description="desc a", status="pending")
        m2 = Milestone(id="a", description="desc a", status="pending")
        assert m1 == m2


class TestReviewCheckpoint:
    def test_checkpoint_creation(self):
        now = datetime.now()
        cp = ReviewCheckpoint(
            seq=10,
            milestone_id="understand",
            confirmed_at=now,
        )
        assert cp.seq == 10
        assert cp.milestone_id == "understand"
        assert cp.confirmed_at == now


class TestReviewState:
    def test_review_state_defaults(self):
        rs = ReviewState()
        assert rs.milestones == []
        assert rs.last_review_seq == 0
        assert rs.last_checkpoint_seq == 0
        assert rs.consecutive_errors == 0
        assert rs.is_review_pending is False

    def test_review_state_with_milestones(self):
        m = Milestone(id="a", description="desc", status="active")
        rs = ReviewState(
            milestones=[m],
            last_review_seq=5,
            last_checkpoint_seq=3,
            consecutive_errors=2,
            is_review_pending=True,
        )
        assert len(rs.milestones) == 1
        assert rs.last_review_seq == 5
        assert rs.last_checkpoint_seq == 3
        assert rs.consecutive_errors == 2
        assert rs.is_review_pending is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/agent/test_review_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'agiwo.agent.models.review'`

- [ ] **Step 3: Create `agiwo/agent/models/review.py`**

```python
"""Review and milestone data models for goal-directed review."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class Milestone:
    """A verifiable sub-goal declared by the agent."""

    id: str
    description: str
    status: Literal["pending", "active", "completed", "abandoned"] = "pending"
    declared_at_seq: int = 0
    completed_at_seq: int | None = None


@dataclass
class ReviewCheckpoint:
    """A confirmed-aligned checkpoint recorded after a successful review."""

    seq: int
    milestone_id: str
    confirmed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class ReviewState:
    """Per-run review tracking state, stored on RunLedger."""

    milestones: list[Milestone] = field(default_factory=list)
    last_review_seq: int = 0
    last_checkpoint_seq: int = 0
    consecutive_errors: int = 0
    is_review_pending: bool = False


__all__ = [
    "Milestone",
    "ReviewCheckpoint",
    "ReviewState",
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/agent/test_review_models.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Add ReviewState to RunLedger in `agiwo/agent/models/run.py`**

Read the file first to understand current layout. Replace the legacy review-tracking state import and usage:

```python
# Line 24: change import
from agiwo.agent.models.review import ReviewState

# Lines 97-103: replace the legacy review-tracking state with a comment that it's removed
# (keep it for now in Phase 1, will be fully cleaned up in Phase 9)

# Line 133: add new field to RunLedger
review: ReviewState = field(default_factory=ReviewState)
```

The legacy review-tracking field on `RunLedger` (line 133) stays for now — it will be removed in the cleanup phase.

- [ ] **Step 6: Commit**

```bash
git add agiwo/agent/models/review.py agiwo/agent/models/run.py tests/agent/test_review_models.py
git commit -m "feat: add Milestone, ReviewCheckpoint, and ReviewState data models"
```

---

### Task 2: Goal Manager — milestone CRUD + declaration logic

**Files:**
- Create: `agiwo/agent/review/__init__.py`
- Create: `agiwo/agent/review/goal_manager.py`
- Test: `tests/agent/test_goal_manager.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_goal_manager.py
import pytest
from agiwo.agent.models.review import Milestone, ReviewState
from agiwo.agent.review.goal_manager import (
    GoalManager,
    declare_milestones,
    activate_next_milestone,
    complete_active_milestone,
    get_active_milestone,
)


class TestDeclareMilestones:
    def test_first_declaration_sets_first_active(self):
        state = ReviewState()
        milestones = [
            Milestone(id="a", description="Step A"),
            Milestone(id="b", description="Step B"),
        ]
        result = declare_milestones(state, milestones)
        assert len(state.milestones) == 2
        assert state.milestones[0].status == "active"
        assert state.milestones[1].status == "pending"
        assert result == ["a", "b"]

    def test_redeclare_preserves_previous_if_no_active_change(self):
        m = Milestone(id="a", description="Step A", status="completed")
        m2 = Milestone(id="b", description="Step B", status="active")
        state = ReviewState(milestones=[m, m2])
        result = declare_milestones(state, [
            Milestone(id="c", description="Step C"),
        ])
        # state unchanged because active milestone wasn't in the new list
        assert len(state.milestones) == 2
        assert result == ["c"]


class TestCompleteActiveMilestone:
    def test_complete_active(self):
        m = Milestone(id="a", description="Step A", status="active")
        state = ReviewState(milestones=[m])
        result = complete_active_milestone(state, seq=10)
        assert result is True
        assert state.milestones[0].status == "completed"
        assert state.milestones[0].completed_at_seq == 10

    def test_complete_no_active_returns_false(self):
        state = ReviewState()
        result = complete_active_milestone(state, seq=10)
        assert result is False


class TestActivateNextMilestone:
    def test_activate_first_pending(self):
        m1 = Milestone(id="a", description="A", status="completed")
        m2 = Milestone(id="b", description="B", status="pending")
        state = ReviewState(milestones=[m1, m2])
        result = activate_next_milestone(state)
        assert result is not None
        assert result.id == "b"
        assert result.status == "active"

    def test_activate_none_pending_returns_none(self):
        m = Milestone(id="a", description="A", status="completed")
        state = ReviewState(milestones=[m])
        result = activate_next_milestone(state)
        assert result is None


class TestGetActiveMilestone:
    def test_returns_active(self):
        m = Milestone(id="a", description="A", status="active")
        state = ReviewState(milestones=[m])
        result = get_active_milestone(state)
        assert result is not None
        assert result.id == "a"

    def test_returns_none_when_none_active(self):
        m = Milestone(id="a", description="A", status="completed")
        state = ReviewState(milestones=[m])
        result = get_active_milestone(state)
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/agent/test_goal_manager.py -v
```

Expected: `ModuleNotFoundError: No module named 'agiwo.agent.review.goal_manager'`

- [ ] **Step 3: Create `agiwo/agent/review/goal_manager.py`**

```python
"""Goal Manager — milestone declaration, activation, completion, and querying."""

from agiwo.agent.models.review import Milestone, ReviewState


def declare_milestones(
    state: ReviewState,
    milestones: list[Milestone],
    *,
    current_seq: int = 0,
) -> list[str]:
    """Declare or update milestones. First pending becomes active if none is active.

    Returns the list of milestone ids that were declared.
    """
    has_active = any(m.status == "active" for m in state.milestones)

    new_ids = [m.id for m in milestones]
    existing_ids = {m.id for m in state.milestones}

    # Update existing or add new
    for m in milestones:
        m.declared_at_seq = current_seq
        if m.id in existing_ids:
            # Update existing milestone
            for i, existing in enumerate(state.milestones):
                if existing.id == m.id:
                    state.milestones[i] = m
                    break
        else:
            state.milestones.append(m)

    # If no milestone is currently active, activate the first one
    if not has_active and state.milestones:
        for m in state.milestones:
            if m.status == "pending":
                m.status = "active"
                break

    return new_ids


def complete_active_milestone(state: ReviewState, *, seq: int) -> bool:
    """Mark the active milestone as completed. Returns True if one was completed."""
    for m in state.milestones:
        if m.status == "active":
            m.status = "completed"
            m.completed_at_seq = seq
            return True
    return False


def activate_next_milestone(state: ReviewState) -> Milestone | None:
    """Activate the first pending milestone. Returns it, or None."""
    for m in state.milestones:
        if m.status == "pending":
            m.status = "active"
            return m
    return None


def get_active_milestone(state: ReviewState) -> Milestone | None:
    """Return the currently active milestone, or None."""
    for m in state.milestones:
        if m.status == "active":
            return m
    return None


__all__ = [
    "activate_next_milestone",
    "complete_active_milestone",
    "declare_milestones",
    "get_active_milestone",
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/agent/test_goal_manager.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Create `agiwo/agent/review/__init__.py`**

```python
"""Goal-directed review — replaces token/round-based step-back.

Public API consumed by ``run_tool_batch.py``:

* ``ReviewBatch`` — per-batch lifecycle object
* ``StepBackOutcome`` — result of a step-back pass
"""

from agiwo.agent.review.goal_manager import (
    activate_next_milestone,
    complete_active_milestone,
    declare_milestones,
    get_active_milestone,
)

__all__ = [
    "ReviewBatch",
    "StepBackOutcome",
    "activate_next_milestone",
    "complete_active_milestone",
    "declare_milestones",
    "get_active_milestone",
]
```

Note: `ReviewBatch` and `StepBackOutcome` are forward-referenced — they'll be added in later tasks.

- [ ] **Step 6: Commit**

```bash
git add agiwo/agent/review/__init__.py agiwo/agent/review/goal_manager.py tests/agent/test_goal_manager.py
git commit -m "feat: add GoalManager with milestone declare/complete/activate/query"
```

---

## Phase 2: Review Enforcer

### Task 3: Review Enforcer — trigger checks + system-review injection

**Files:**
- Create: `agiwo/agent/review/review_enforcer.py`
- Test: `tests/agent/test_review_enforcer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_review_enforcer.py
import pytest
from agiwo.agent.models.review import Milestone, ReviewState
from agiwo.agent.review.review_enforcer import (
    ReviewTrigger,
    check_review_trigger,
    inject_system_review,
)


class TestCheckReviewTrigger:
    def test_disabled_returns_none(self):
        state = ReviewState()
        trigger = check_review_trigger(
            state=state,
            enabled=False,
            is_error=False,
            step_interval=8,
            error_threshold=2,
        )
        assert trigger == ReviewTrigger.NONE

    def test_error_consecutive_trigger(self):
        state = ReviewState(consecutive_errors=2)
        trigger = check_review_trigger(
            state=state,
            enabled=True,
            is_error=True,
            step_interval=8,
            error_threshold=2,
        )
        assert trigger == ReviewTrigger.CONSECUTIVE_ERRORS

    def test_step_interval_trigger(self):
        state = ReviewState(
            last_review_seq=5,
            last_checkpoint_seq=5,
            consecutive_errors=0,
        )
        # current_seq=14, last_review_seq=5, diff=9 >= interval=8
        trigger = check_review_trigger(
            state=state,
            enabled=True,
            is_error=False,
            step_interval=8,
            error_threshold=2,
            current_seq=14,
        )
        assert trigger == ReviewTrigger.STEP_INTERVAL

    def test_pending_review_trigger(self):
        state = ReviewState(is_review_pending=True)
        trigger = check_review_trigger(
            state=state,
            enabled=True,
            is_error=False,
            step_interval=8,
            error_threshold=2,
        )
        assert trigger == ReviewTrigger.MILESTONE_SWITCH

    def test_no_trigger_for_review_tool_itself(self):
        state = ReviewState(is_review_pending=True)
        trigger = check_review_trigger(
            state=state,
            enabled=True,
            is_error=False,
            step_interval=8,
            error_threshold=2,
            tool_name="review_trajectory",
        )
        assert trigger == ReviewTrigger.NONE

    def test_below_interval_no_trigger(self):
        state = ReviewState(last_review_seq=5, last_checkpoint_seq=5)
        trigger = check_review_trigger(
            state=state,
            enabled=True,
            is_error=False,
            step_interval=8,
            error_threshold=2,
            current_seq=7,
        )
        assert trigger == ReviewTrigger.NONE


class TestInjectSystemReview:
    def test_injects_review_with_milestone(self):
        content = "Tool result content"
        milestone = Milestone(id="locate", description="定位超时根因", status="active")
        result = inject_system_review(content, milestone, step_count=3)
        assert "<system-review>" in result
        assert content in result
        assert "定位超时根因" in result
        assert "review_trajectory" in result

    def test_injects_review_without_milestone(self):
        content = "Tool result content"
        result = inject_system_review(content, None, step_count=5)
        assert "<system-review>" in result
        assert "No active milestone" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/agent/test_review_enforcer.py -v
```

Expected: `ModuleNotFoundError: No module named 'agiwo.agent.review.review_enforcer'`

- [ ] **Step 3: Create `agiwo/agent/review/review_enforcer.py`**

```python
"""Review Enforcer — trigger detection and system-review notice injection."""

from enum import Enum

from agiwo.agent.models.review import Milestone, ReviewState


class ReviewTrigger(Enum):
    """Which condition fired the review."""

    NONE = "none"
    STEP_INTERVAL = "step_interval"
    CONSECUTIVE_ERRORS = "consecutive_errors"
    MILESTONE_SWITCH = "milestone_switch"


def _build_review_notice(
    milestone: Milestone | None,
    step_count: int,
) -> str:
    """Build the <system-review> notice text."""
    milestone_text = (
        f'Active milestone: "{milestone.description}"'
        if milestone is not None
        else "No active milestone declared. Consider using declare_milestones."
    )

    inner_text = (
        f"{milestone_text}\n\n"
        f"Steps since last review: {step_count}\n\n"
        f"Question: Do your recent steps meaningfully advance the current goal?\n"
        f"If not, use review_trajectory to:\n"
        f"  1. Indicate misalignment (aligned=false)\n"
        f"  2. Provide a concise experience summary of what was learned\n"
        f"If aligned, use review_trajectory with aligned=true and a brief note."
    )

    return f"\n\n<system-review>\n{inner_text}\n</system-review>"


def check_review_trigger(
    *,
    state: ReviewState,
    enabled: bool,
    is_error: bool,
    step_interval: int,
    error_threshold: int,
    tool_name: str = "",
    current_seq: int = 0,
) -> ReviewTrigger:
    """Check if a review should be triggered. Returns the trigger type or NONE."""
    if not enabled:
        return ReviewTrigger.NONE
    if tool_name == "review_trajectory":
        return ReviewTrigger.NONE

    # Milestone switch (agent just declared/completed a milestone)
    if state.is_review_pending:
        return ReviewTrigger.MILESTONE_SWITCH

    # Consecutive errors
    if is_error and state.consecutive_errors >= error_threshold:
        return ReviewTrigger.CONSECUTIVE_ERRORS

    # Step interval since last review
    steps_since_review = current_seq - state.last_review_seq
    if steps_since_review >= step_interval:
        return ReviewTrigger.STEP_INTERVAL

    return ReviewTrigger.NONE


def inject_system_review(
    content: str,
    milestone: Milestone | None,
    step_count: int,
) -> str:
    """Append a <system-review> notice to the tool result content.

    Returns content unchanged if no notice should be injected.
    """
    notice = _build_review_notice(milestone, step_count)
    return content + notice


__all__ = [
    "ReviewTrigger",
    "check_review_trigger",
    "inject_system_review",
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/agent/test_review_enforcer.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Update `agiwo/agent/review/__init__.py` exports**

Add to imports:
```python
from agiwo.agent.review.review_enforcer import (
    ReviewTrigger,
    check_review_trigger,
    inject_system_review,
)
```

Add to `__all__`:
```python
    "ReviewTrigger",
    "check_review_trigger",
    "inject_system_review",
```

- [ ] **Step 6: Commit**

```bash
git add agiwo/agent/review/review_enforcer.py agiwo/agent/review/__init__.py tests/agent/test_review_enforcer.py
git commit -m "feat: add ReviewEnforcer with trigger detection and system-review injection"
```

---

## Phase 3: StepBack Executor

### Task 4: StepBack Executor — KV-cache-safe content condensation

**Files:**
- Create: `agiwo/agent/review/step_back_executor.py`
- Test: `tests/agent/test_step_back_executor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_step_back_executor.py
import pytest
from unittest.mock import AsyncMock
from agiwo.agent.models.review import ReviewState
from agiwo.agent.review.step_back_executor import (
    StepBackOutcome,
    execute_step_back,
)


def _make_messages():
    """Build a realistic message list for testing."""
    return [
        {"role": "system", "content": "You are an agent", "_sequence": 0},
        {"role": "user", "content": "Fix the bug", "_sequence": 1},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"session"}'},
                }
            ],
            "_sequence": 2,
        },
        {
            "role": "tool",
            "tool_call_id": "tc_1",
            "content": "Found SessionManager in auth.py",
            "_sequence": 3,
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_2",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"jwt"}'},
                }
            ],
            "_sequence": 4,
        },
        {
            "role": "tool",
            "tool_call_id": "tc_2",
            "content": "Found 15 JWT references",
            "_sequence": 5,
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc_review",
                    "type": "function",
                    "function": {
                        "name": "review_trajectory",
                        "arguments": '{"aligned":false,"experience":"JWT search was off-track"}',
                    },
                }
            ],
            "_sequence": 6,
        },
        {
            "role": "tool",
            "tool_call_id": "tc_review",
            "content": "Review acknowledged",
            "_sequence": 7,
        },
    ]


class TestExecuteStepBack:
    @pytest.mark.asyncio
    async def test_condenses_tool_results_after_checkpoint(self):
        messages = _make_messages()
        storage = AsyncMock()
        storage.append_step_condensed_content = AsyncMock(return_value=True)

        outcome = await execute_step_back(
            messages=messages,
            checkpoint_seq=3,
            experience="JWT search was off-track. Token validation lives in auth.py.",
            step_lookup={
                "tc_1": {"id": "step_1", "sequence": 3},
                "tc_2": {"id": "step_2", "sequence": 5},
            },
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        assert outcome.applied is True
        assert outcome.affected_count == 1  # only tc_2 (seq 5 > checkpoint 3)
        assert outcome.checkpoint_seq == 3
        assert len(outcome.messages) == len(messages) - 2  # review_trajectory removed

        # tc_1 (seq 3) is at or before checkpoint — should be unchanged
        assert messages[3]["content"] == "Found SessionManager in auth.py"

        # tc_2 (seq 5) is after checkpoint — should be condensed
        assert "[EXPERIENCE]" in messages[5]["content"]
        assert "off-track" in messages[5]["content"]

    @pytest.mark.asyncio
    async def test_preserves_tool_calls(self):
        messages = _make_messages()
        storage = AsyncMock()
        storage.append_step_condensed_content = AsyncMock(return_value=True)

        await execute_step_back(
            messages=messages,
            checkpoint_seq=2,
            experience="All steps were off-track.",
            step_lookup={
                "tc_1": {"id": "step_1", "sequence": 3},
                "tc_2": {"id": "step_2", "sequence": 5},
            },
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        # assistant messages with tool_calls are preserved
        assert messages[2]["tool_calls"][0]["function"]["name"] == "search"
        assert messages[4]["tool_calls"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_removes_review_trajectory_messages(self):
        messages = _make_messages()
        storage = AsyncMock()
        storage.append_step_condensed_content = AsyncMock(return_value=True)

        outcome = await execute_step_back(
            messages=messages,
            checkpoint_seq=2,
            experience="Done.",
            step_lookup={
                "tc_1": {"id": "step_1", "sequence": 3},
                "tc_2": {"id": "step_2", "sequence": 5},
            },
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        # No review_trajectory references remain
        for msg in outcome.messages:
            if msg.get("role") == "tool":
                assert msg.get("tool_call_id") != "tc_review"
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    assert tc["function"]["name"] != "review_trajectory"

    @pytest.mark.asyncio
    async def test_no_op_when_no_tool_results_after_checkpoint(self):
        messages = [
            {"role": "user", "content": "hello", "_sequence": 1},
            {"role": "assistant", "content": "hi", "_sequence": 2},
        ]
        storage = AsyncMock()

        outcome = await execute_step_back(
            messages=messages,
            checkpoint_seq=5,  # after all messages
            experience="Nothing to condense.",
            step_lookup={},
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        assert outcome.applied is True
        assert outcome.affected_count == 0
        assert len(outcome.messages) == len(messages)

    @pytest.mark.asyncio
    async def test_persists_condensed_content_to_storage(self):
        messages = _make_messages()
        storage = AsyncMock()
        storage.append_step_condensed_content = AsyncMock(return_value=True)

        await execute_step_back(
            messages=messages,
            checkpoint_seq=2,
            experience="Condensed.",
            step_lookup={
                "tc_1": {"id": "step_1", "sequence": 3},
                "tc_2": {"id": "step_2", "sequence": 5},
            },
            storage=storage,
            session_id="s1",
            run_id="r1",
            agent_id="a1",
        )

        # Should have called append_step_condensed_content for affected steps
        assert storage.append_step_condensed_content.call_count >= 2


class TestStepBackOutcome:
    def test_default_not_applied(self):
        outcome = StepBackOutcome()
        assert outcome.applied is False
        assert outcome.affected_count == 0
        assert outcome.messages == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/agent/test_step_back_executor.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `agiwo/agent/review/step_back_executor.py`**

```python
"""StepBack Executor — KV-cache-safe content condensation."""

from dataclasses import dataclass, field
from typing import Any

from agiwo.agent.storage.base import RunLogStorage
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StepBackOutcome:
    """Result of a step-back execution.

    When ``applied`` is True, content_updates contains (msg_index, new_content)
    pairs. The caller applies these as targeted in-place updates — no
    rebuild_messages, no message reordering.
    """

    applied: bool = False
    affected_count: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_seq: int = 0
    experience: str | None = None


async def execute_step_back(
    *,
    messages: list[dict[str, Any]],
    checkpoint_seq: int,
    experience: str,
    step_lookup: dict[str, dict[str, Any]],
    storage: RunLogStorage,
    session_id: str,
    run_id: str,
    agent_id: str,
) -> StepBackOutcome:
    """Condense tool results after *checkpoint_seq* to *experience*.

    Core invariants (KV-cache-safe):
    - Tool call assistant messages are NEVER removed or reordered
    - Only tool result content is replaced in-place
    - review_trajectory's own call + result are removed (they're at the tail)
    - No rebuild_messages call — caller applies targeted content updates
    """
    working = list(messages)  # shallow copy
    affected_count = 0

    # 1. Find and condense tool results after checkpoint
    for i, msg in enumerate(working):
        if msg.get("role") != "tool":
            continue
        seq = msg.get("_sequence", 0)
        if seq <= checkpoint_seq:
            continue

        tool_call_id = msg.get("tool_call_id", "")
        if not tool_call_id or tool_call_id.startswith("tc_review"):
            continue

        original_content = msg.get("content", "")
        if not original_content:
            continue

        condensed = f"[EXPERIENCE] {experience}"
        msg["content"] = condensed
        affected_count += 1

        # Persist original content to storage (non-blocking)
        step_info = step_lookup.get(tool_call_id)
        if step_info is not None:
            step_id = step_info.get("id", "")
            if step_id:
                await storage.append_step_condensed_content(
                    session_id,
                    run_id,
                    agent_id,
                    step_id,
                    original_content,
                )

    # 2. Remove review_trajectory tool call and result
    _remove_review_tool_call(working)

    logger.info(
        "step_back_executed",
        session_id=session_id,
        affected_count=affected_count,
        checkpoint_seq=checkpoint_seq,
    )

    return StepBackOutcome(
        applied=True,
        affected_count=affected_count,
        messages=working,
        checkpoint_seq=checkpoint_seq,
        experience=experience,
    )


def _remove_review_tool_call(messages: list[dict[str, Any]]) -> None:
    """Remove review_trajectory tool call and its result from the message list."""
    indices_to_remove: list[int] = []
    review_call_ids: set[str] = set()

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            remaining: list[dict[str, Any]] = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                if fn.get("name") == "review_trajectory":
                    review_call_ids.add(tc.get("id", ""))
                else:
                    remaining.append(tc)
            msg["tool_calls"] = remaining
            if not remaining:
                indices_to_remove.append(i)

    for i, msg in enumerate(messages):
        if (
            msg.get("role") == "tool"
            and msg.get("tool_call_id", "") in review_call_ids
        ):
            indices_to_remove.append(i)

    for i in sorted(indices_to_remove, reverse=True):
        messages.pop(i)


__all__ = [
    "StepBackOutcome",
    "execute_step_back",
]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/agent/test_step_back_executor.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Update `agiwo/agent/review/__init__.py` exports**

Add to imports:
```python
from agiwo.agent.review.step_back_executor import (
    StepBackOutcome,
    execute_step_back,
)
```

Update `__all__` to include `"StepBackOutcome"` and `"execute_step_back"`.

- [ ] **Step 6: Commit**

```bash
git add agiwo/agent/review/step_back_executor.py agiwo/agent/review/__init__.py tests/agent/test_step_back_executor.py
git commit -m "feat: add StepBackExecutor with KV-cache-safe content condensation"
```

---

## Phase 4: ReviewBatch — glue between review components

### Task 5: ReviewBatch — per-batch lifecycle (replaces ReviewBatch)

**Files:**
- Modify: `agiwo/agent/review/__init__.py` — add `ReviewBatch` class
- Test: `tests/agent/test_review_batch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_review_batch.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from agiwo.agent.models.review import Milestone, ReviewState
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.run import RunLedger
from agiwo.agent.review import ReviewBatch, StepBackOutcome


class FakeTool:
    def __init__(self, name):
        self.name = name


class FakeToolResult:
    def __init__(self, tool_name, content, is_success=True, tool_call_id="tc_1"):
        self.tool_name = tool_name
        self.content = content
        self.is_success = is_success
        self.tool_call_id = tool_call_id
        self.input_args = {}
        self.content_for_user = None
        self.termination_reason = None
        self.output = {}


class TestReviewBatch:
    def test_enabled_when_configured_and_tools_present(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        assert batch.enabled is True

    def test_disabled_when_flag_off(self):
        config = AgentOptions(enable_goal_directed_review=False)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)
        assert batch.enabled is False

    def test_disabled_when_review_tool_missing(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {"declare_milestones": FakeTool("declare_milestones")}
        batch = ReviewBatch(config, ledger, tools_map)
        assert batch.enabled is False

    def test_disabled_when_milestones_tool_missing(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {"review_trajectory": FakeTool("review_trajectory")}
        batch = ReviewBatch(config, ledger, tools_map)
        assert batch.enabled is False

    def test_process_result_captures_review_feedback(self):
        config = AgentOptions(enable_goal_directed_review=True)
        ledger = RunLedger()
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)

        result = FakeToolResult("review_trajectory", "JWT was a dead end", is_success=True, tool_call_id="tc_review")
        content = batch.process_result(result)
        assert content == "JWT was a dead end"
        assert batch._feedback == "JWT was a dead end"

    def test_process_result_injects_review_when_triggered(self):
        config = AgentOptions(enable_goal_directed_review=True, review_step_interval=1)
        ledger = RunLedger()
        ledger.review.last_review_seq = 0
        ledger.review.milestones = [
            Milestone(id="a", description="Find the bug", status="active")
        ]
        tools_map = {
            "review_trajectory": FakeTool("review_trajectory"),
            "declare_milestones": FakeTool("declare_milestones"),
        }
        batch = ReviewBatch(config, ledger, tools_map)

        result = FakeToolResult("search", "Found 5000 tokens worth of results", tool_call_id="tc_search")
        content = batch.process_result(result, current_seq=2)
        assert "<system-review>" in content
        assert "Find the bug" in content

    def test_register_step_stores_entry(self):
        config = AgentOptions()
        ledger = RunLedger()
        tools_map = {}
        batch = ReviewBatch(config, ledger, tools_map)
        batch.register_step("tc_1", "step_1", 5)
        assert batch._step_lookup["tc_1"] == {"id": "step_1", "sequence": 5}

    @pytest.mark.asyncio
    async def test_finalize_returns_not_applied_when_no_feedback(self):
        config = AgentOptions()
        ledger = RunLedger()
        tools_map = {}
        batch = ReviewBatch(config, ledger, tools_map)
        outcome = await batch.finalize()
        assert outcome.applied is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/agent/test_review_batch.py -v
```

Expected: ImportError for `ReviewBatch`

- [ ] **Step 3: Add `ReviewBatch` class to `agiwo/agent/review/__init__.py`**

Read the current `__init__.py`, then add `ReviewBatch` class alongside the existing imports:

```python
from typing import Any

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.review import ReviewState, Milestone
from agiwo.agent.review.goal_manager import get_active_milestone
from agiwo.agent.models.run import RunLedger
from agiwo.agent.review.review_enforcer import (
    ReviewTrigger,
    check_review_trigger,
    inject_system_review,
)
from agiwo.agent.review.step_back_executor import (
    StepBackOutcome,
    execute_step_back,
)
from agiwo.tool.base import BaseTool, ToolResult


class ReviewBatch:
    """Per-batch review lifecycle object.

    Replaces ``ReviewBatch``.  Caller interacts through three methods:

    * ``process_result()``  — returns final content (may inject <system-review>)
    * ``register_step()``   — registers a committed step for later lookup
    * ``finalize()``        — returns ``StepBackOutcome``; caller applies
      via targeted content updates (NO rebuild_messages)
    """

    def __init__(
        self,
        config: AgentOptions,
        ledger: RunLedger,
        tools_map: dict[str, BaseTool],
    ) -> None:
        self._config = config
        self._ledger = ledger
        self._enabled = (
            config.enable_goal_directed_review
            and "review_trajectory" in tools_map
            and "declare_milestones" in tools_map
        )
        self._feedback: str | None = None
        self._aligned: bool | None = None
        self._step_lookup: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def process_result(
        self,
        result: ToolResult,
        *,
        current_seq: int = 0,
    ) -> str:
        """Process a tool result. Returns the (possibly transformed) content."""
        content = result.content or ""
        if not self._enabled:
            return content

        # Capture review_trajectory feedback
        if result.tool_name == "review_trajectory" and result.is_success:
            self._feedback = content
            return content

        # Capture declare_milestones — set review_pending flag
        if result.tool_name == "declare_milestones" and result.is_success:
            self._ledger.review.is_review_pending = True

        # Track errors
        is_error = not result.is_success
        if is_error:
            self._ledger.review.consecutive_errors += 1
        else:
            self._ledger.review.consecutive_errors = 0

        # Check triggers
        trigger = check_review_trigger(
            state=self._ledger.review,
            enabled=True,
            is_error=is_error,
            step_interval=self._config.review_step_interval,
            error_threshold=2,
            tool_name=result.tool_name,
            current_seq=current_seq,
        )

        if trigger is not ReviewTrigger.NONE:
            milestone = get_active_milestone(self._ledger.review)
            step_count = current_seq - self._ledger.review.last_review_seq
            content = inject_system_review(content, milestone, step_count)

        return content

    def register_step(
        self, tool_call_id: str, step_id: str, sequence: int
    ) -> None:
        """Register a committed step for later step-back lookup."""
        self._step_lookup[tool_call_id] = {
            "id": step_id,
            "sequence": sequence,
        }

    async def finalize(
        self,
        *,
        storage=None,
        session_id: str = "",
        run_id: str = "",
        agent_id: str = "",
    ) -> StepBackOutcome:
        """Build the step-back outcome if feedback was captured.

        Returns StepBackOutcome(applied=False) when no feedback or not enabled.
        """
        if not self._enabled or self._feedback is None:
            return StepBackOutcome()

        checkpoint_seq = self._ledger.review.last_checkpoint_seq
        messages = list(self._ledger.messages)

        outcome = await execute_step_back(
            messages=messages,
            checkpoint_seq=checkpoint_seq,
            experience=self._feedback,
            step_lookup=self._step_lookup,
            storage=storage,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
        )

        # Reset tracking
        self._ledger.review.last_review_seq = max(
            msg.get("_sequence", 0) for msg in outcome.messages
        )
        self._ledger.review.is_review_pending = False

        return outcome
```

Update `__all__` to include `"ReviewBatch"`.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/agent/test_review_batch.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/review/__init__.py tests/agent/test_review_batch.py
git commit -m "feat: add ReviewBatch per-batch lifecycle class"
```

---

## Phase 5: New Runtime Tools

### Task 6: Add declare_milestones and review_trajectory tools

**Files:**
- Modify: `agiwo/scheduler/runtime_tools.py` — add 2 new tool classes
- Modify: `agiwo/scheduler/engine.py` — register new tools
- Test: `tests/scheduler/test_review_tools.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/scheduler/test_review_tools.py
import pytest
from unittest.mock import MagicMock
from agiwo.scheduler.runtime_tools import (
    DeclareMilestonesTool,
    ReviewTrajectoryTool,
)
from agiwo.tool.base import ToolContext
from agiwo.scheduler.tool_control import SchedulerToolControl


class TestDeclareMilestonesTool:
    def test_name_and_description(self):
        tool = DeclareMilestonesTool(MagicMock(spec=SchedulerToolControl))
        assert tool.name == "declare_milestones"
        assert "milestones" in tool.description.lower()

    def test_parameters_schema(self):
        tool = DeclareMilestonesTool(MagicMock(spec=SchedulerToolControl))
        params = tool.get_parameters()
        assert params["type"] == "object"
        assert "milestones" in params["properties"]
        assert "milestones" in params["required"]

    @pytest.mark.asyncio
    async def test_execute_success(self):
        tool = DeclareMilestonesTool(MagicMock(spec=SchedulerToolControl))
        result = await tool.execute(
            parameters={
                "milestones": [
                    {"id": "a", "description": "Step A"},
                    {"id": "b", "description": "Step B"},
                ]
            },
            context=ToolContext(tool_name="declare_milestones", tool_call_id="tc_1"),
        )
        assert result.is_success
        assert "a" in result.content
        assert "b" in result.content

    @pytest.mark.asyncio
    async def test_execute_empty_milestones(self):
        tool = DeclareMilestonesTool(MagicMock(spec=SchedulerToolControl))
        result = await tool.execute(
            parameters={"milestones": []},
            context=ToolContext(tool_name="declare_milestones", tool_call_id="tc_1"),
        )
        assert not result.is_success


class TestReviewTrajectoryTool:
    def test_name_and_description(self):
        tool = ReviewTrajectoryTool(MagicMock(spec=SchedulerToolControl))
        assert tool.name == "review_trajectory"
        assert "system-review" in tool.description.lower()

    def test_parameters_schema(self):
        tool = ReviewTrajectoryTool(MagicMock(spec=SchedulerToolControl))
        params = tool.get_parameters()
        assert params["type"] == "object"
        assert "aligned" in params["properties"]
        assert "experience" in params["properties"]
        assert "aligned" in params["required"]

    @pytest.mark.asyncio
    async def test_execute_aligned_true(self):
        tool = ReviewTrajectoryTool(MagicMock(spec=SchedulerToolControl))
        result = await tool.execute(
            parameters={"aligned": True},
            context=ToolContext(tool_name="review_trajectory", tool_call_id="tc_r"),
        )
        assert result.is_success

    @pytest.mark.asyncio
    async def test_execute_aligned_false_requires_experience(self):
        tool = ReviewTrajectoryTool(MagicMock(spec=SchedulerToolControl))
        result = await tool.execute(
            parameters={
                "aligned": False,
                "experience": "Tried X, learned Y, will do Z next.",
            },
            context=ToolContext(tool_name="review_trajectory", tool_call_id="tc_r"),
        )
        assert result.is_success
        assert "Tried X" in result.content

    @pytest.mark.asyncio
    async def test_execute_aligned_false_without_experience(self):
        tool = ReviewTrajectoryTool(MagicMock(spec=SchedulerToolControl))
        result = await tool.execute(
            parameters={"aligned": False},
            context=ToolContext(tool_name="review_trajectory", tool_call_id="tc_r"),
        )
        assert not result.is_success
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/scheduler/test_review_tools.py -v
```

Expected: `ImportError` for `DeclareMilestonesTool`

- [ ] **Step 3: Add tools to `agiwo/scheduler/runtime_tools.py`**

Add at the end of the file (before `__all__` if one exists):

```python
class DeclareMilestonesTool(BaseTool):
    """Declare or update the milestones for the current task."""

    name = "declare_milestones"
    description = (
        "Declare or update the milestones for the current task. "
        "Break the user's request into concrete, verifiable sub-goals. "
        "Each milestone should have a clear id and a specific description "
        "of what 'done' looks like. The system uses these milestones to "
        "evaluate whether your work stays on track.\n\n"
        "Example: [{\"id\":\"understand\",\"description\":\"Identify how "
        "auth tokens are validated\"}, {\"id\":\"fix\",\"description\":"
        "\"Apply the fix and verify with tests\"}]"
    )
    concurrency_safe = False

    def __init__(self, port: SchedulerToolControl) -> None:
        self._port = port
        super().__init__()

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "milestones": {
                    "type": "array",
                    "description": (
                        "Ordered list of milestone objects. Each must have: "
                        "id (string, unique identifier) and "
                        "description (string, what 'done' looks like)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["id", "description"],
                    },
                },
            },
            "required": ["milestones"],
        }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        milestones = parameters.get("milestones", [])
        if not milestones:
            return _build_failed_result(
                tool_name=self.name,
                error="milestones must be a non-empty array",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )
        ids = [m["id"] for m in milestones]
        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content=f"Milestones declared: {', '.join(ids)}",
            output={},
            start_time=start_time,
        )


class ReviewTrajectoryTool(BaseTool):
    """Respond to a <system-review> prompt."""

    name = "review_trajectory"
    description = (
        "Respond to a <system-review> prompt by assessing whether "
        "your recent tool calls advance the active milestone.\n\n"
        "Parameters:\n"
        "- aligned (boolean, required): true if trajectory aligns with "
        "milestone, false if drifted.\n"
        "- experience (string, required when aligned=false): A concise "
        "summary covering what was attempted, what was learned, and "
        "how this should inform the next approach.\n\n"
        "This tool call itself will be transparently removed after processing."
    )
    concurrency_safe = False

    def __init__(self, port: SchedulerToolControl) -> None:
        self._port = port
        super().__init__()

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "aligned": {
                    "type": "boolean",
                    "description": (
                        "true if your recent trajectory aligns with the "
                        "active milestone, false if it has drifted."
                    ),
                },
                "experience": {
                    "type": "string",
                    "description": (
                        "Required when aligned=false. A concise summary: "
                        "what was attempted, what was learned, and how "
                        "this should inform the next approach."
                    ),
                },
            },
            "required": ["aligned"],
        }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        aligned = parameters.get("aligned", False)
        experience = parameters.get("experience", "")

        if not aligned and not experience:
            return _build_failed_result(
                tool_name=self.name,
                error="experience is required when aligned=false",
                context=context,
                parameters=parameters,
                start_time=start_time,
            )

        content = (
            f"Trajectory review: aligned={aligned}. {experience}"
            if experience
            else f"Trajectory review: aligned={aligned}."
        )
        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content=content,
            output={},
            start_time=start_time,
        )
```

- [ ] **Step 4: Add imports at top of `agiwo/scheduler/runtime_tools.py`**

Add `import time` at the top if not already present (it already is — used by `ReviewTrajectoryTool`).

- [ ] **Step 5: Register new tools in `agiwo/scheduler/engine.py`**

Read file first. Then:

In imports (around line 42-50), add:
```python
    DeclareMilestonesTool,
    ReviewTrajectoryTool,
```

In `__init__` (around line 96), add to `_scheduling_tools` tuple:
```python
            DeclareMilestonesTool(self._tool_control),
            ReviewTrajectoryTool(self._tool_control),
```

Note: Keep `ReviewTrajectoryTool(self._tool_control)` for now — it will be removed in the cleanup phase.

- [ ] **Step 6: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/scheduler/test_review_tools.py -v
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add agiwo/scheduler/runtime_tools.py agiwo/scheduler/engine.py tests/scheduler/test_review_tools.py
git commit -m "feat: add declare_milestones and review_trajectory runtime tools"
```

---

## Phase 6: System Prompt Alignment & Config Update

### Task 7: Add Goal-Directed Review section to system prompt

**Files:**
- Modify: `agiwo/agent/prompt.py` — add `_render_goal_directed_review` function

- [ ] **Step 1: Add `_render_goal_directed_review` function**

Add this function after `_render_environment` (after line 103):

```python
def _render_goal_directed_review() -> str:
    return """---
## Goal-Directed Review

You are expected to work in a goal-directed manner. The system helps you
stay on track through a review mechanism.

### Milestones
When you receive a task, break it into concrete milestones using the
`declare_milestones` tool. Each milestone should be a verifiable
sub-goal. Keep milestones focused and specific — "understand the code"
is too vague; "identify how auth tokens are validated" is concrete.

### System Reviews
The system will periodically ask you to review your trajectory against
the active milestone. When you see a `<system-review>` tag in a tool
result, you MUST respond with the `review_trajectory` tool:

- If your recent steps advance the milestone: set `aligned=true` and
  briefly note what was accomplished.
- If your recent steps drifted from the milestone: set `aligned=false`
  and provide a concise experience summary of what was learned.
  The system will condense the off-track results so they don't clutter
  the context.

The system review is not optional — it enforces that you periodically
check your own direction. Treat it as a mandatory checkpoint.

### Step-Back
When you indicate misalignment, the system automatically condenses the
off-target tool results into your experience summary. The tool call
history is preserved so future decisions can reference what was tried,
but the verbose outputs are replaced with the lesson learned."""
```

- [ ] **Step 2: Add `_render_goal_directed_review()` call in `build_system_prompt`**

In `build_system_prompt()` (line 230), add `_render_goal_directed_review()` to the `sections` list after `_render_environment(...)`:

```python
    sections = [
        _render_identity(documents),
        _render_soul(workspace, documents),
        base_prompt.strip() if base_prompt else "",
        _render_environment(
            workspace,
            os_info=_get_os_info(),
            language_info=_get_language_info(),
            timezone=str(current_dt.tzinfo),
            current_date=current_dt.strftime("%Y-%m-%d"),
        ),
        _render_goal_directed_review(),          # <-- ADDED
        _render_tools(
            tuple((tool.name, tool.get_short_description()) for tool in (tools or []))
        ),
        f"---\n\n{skills_section}".strip() if skills_section else "",
        _render_user(documents),
    ]
```

- [ ] **Step 3: Add to `__all__` exports in prompt.py**

Keep `_render_goal_directed_review` module-internal; do not add private helper
names to the public `__all__` list.

- [ ] **Step 4: Verify with existing tests**

```bash
.venv/bin/python -m pytest tests/agent/test_prompt.py -v
```

Expected: all existing tests PASS (or note any failures that need adjusting)

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/prompt.py
git commit -m "feat: add Goal-Directed Review section to system prompt"
```

---

### Task 8: Update AgentOptions config fields

**Files:**
- Modify: `agiwo/agent/models/config.py` — replace 4 step-back fields with 2 review fields
- Modify: `console/server/models/agent_config.py` — same
- Test: `tests/agent/test_config.py` (create if not exists, or update existing)

- [ ] **Step 1: Replace fields in `agiwo/agent/models/config.py`**

Read `agiwo/agent/models/config.py` lines 73-76. Replace:

```python
    enable_goal_directed_review: bool = True
    review_step_interval: int = 1024
    review_step_interval: int = 5
    review_step_interval: int = 8192
```

With:

```python
    enable_goal_directed_review: bool = True
    review_step_interval: int = 8
    review_on_error: bool = True
```

Keep `enable_context_rollback: bool = True` (it's unrelated to step-back).

- [ ] **Step 2: Replace fields in `console/server/models/agent_config.py`**

Read lines 32-35. Replace:

```python
    enable_goal_directed_review: bool = True
    review_step_interval: int = Field(default=1024, ge=1)
    review_step_interval: int = Field(default=5, ge=1)
    review_step_interval: int = Field(default=8192, ge=1)
```

With:

```python
    enable_goal_directed_review: bool = True
    review_step_interval: int = Field(default=8, ge=1)
    review_on_error: bool = True
```

- [ ] **Step 3: Run all tests to check for config field usage**

```bash
.venv/bin/python -m pytest tests/ -k "config" -v --tb=short
```

Fix any test failures related to the renamed fields (update tests that reference old field names).

- [ ] **Step 4: Commit**

```bash
git add agiwo/agent/models/config.py console/server/models/agent_config.py
git commit -m "feat: replace legacy review config fields with review config fields"
```

---

## Phase 7: Log, Stream & Runtime Decision Type Updates

### Task 9: Replace StepBackApplied → StepBackApplied in log, stream, runtime_decision models

**Files:**
- Modify: `agiwo/agent/models/log.py` — add `StepBackApplied`, keep old type temporarily
- Modify: `agiwo/agent/models/stream.py` — add `StepBackAppliedEvent`, keep old type temporarily
- Modify: `agiwo/agent/models/runtime_decision.py` — add `StepBackDecisionView`
- Modify: `agiwo/agent/storage/serialization.py` — handle new types
- Modify: `agiwo/agent/trace_writer.py` — handle new span name
- Modify: `agiwo/agent/hooks.py` — add `BEFORE_REVIEW`/`AFTER_STEP_BACK` phases
- Modify: `agiwo/agent/__init__.py` — update exports

Note: Old types (`StepBackApplied`, `StepBackAppliedEvent`, etc.) stay until the cleanup phase.

- [ ] **Step 1: Add `StepBackApplied` to `agiwo/agent/models/log.py`**

Add after the `StepBackApplied` class:

```python
@dataclass(frozen=True, kw_only=True)
class StepBackApplied(RunLogEntry):
    """Log entry recorded when step-back condenses off-track tool results."""

    affected_count: int
    checkpoint_seq: int
    experience: str
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.STEP_BACK_APPLIED
    )
```

Add `"step_back_applied"` to `RunLogEntryKind` enum. Keep the existing `STEP_BACK_APPLIED` for backward compatibility for now.

- [ ] **Step 2: Add `StepBackAppliedEvent` to `agiwo/agent/models/stream.py`**

```python
@dataclass(kw_only=True)
class StepBackAppliedEvent(AgentStreamItemBase):
    affected_count: int
    checkpoint_seq: int
    experience: str
    type: Literal["step_back_applied"] = "step_back_applied"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["affected_count"] = self.affected_count
        payload["checkpoint_seq"] = self.checkpoint_seq
        payload["experience"] = self.experience
        return payload
```

Add to `_stream_item_from_runtime_entry()` and `AgentStreamItem` TypeAlias.

- [ ] **Step 3: Add `StepBackDecisionView` to `agiwo/agent/models/runtime_decision.py`**

```python
@dataclass(frozen=True, slots=True)
class StepBackDecisionView:
    session_id: str
    run_id: str
    agent_id: str
    sequence: int
    created_at: datetime
    affected_count: int
    checkpoint_seq: int
    experience: str
```

Add `latest_step_back: StepBackDecisionView | None = None` to `RuntimeDecisionState`.

- [ ] **Step 4: Add `BEFORE_REVIEW` / `AFTER_STEP_BACK` hook phases to `agiwo/agent/hooks.py`**

In `HookPhase` enum, add:
```python
    BEFORE_REVIEW = "before_review"
    AFTER_STEP_BACK = "after_step_back"
```

In `_DECISION_SUPPORT_PHASES`, add `HookPhase.BEFORE_REVIEW`.
In `_DECISION_SUPPORT_ALLOWLISTS`, add `HookPhase.BEFORE_REVIEW: {"review_advice"}`.

- [ ] **Step 5: Update serialization and tracing**

In `agiwo/agent/storage/serialization.py`, add handling for `StepBackApplied` in `build_runtime_decision_state_from_entries()`.

In `agiwo/agent/trace_writer.py`, add `StepBackApplied` to the type union and create a `"step_back"` span for it.

- [ ] **Step 6: Update `agiwo/agent/__init__.py` exports**

Add imports and `__all__` entries for new types. Keep old exports for backward compatibility.

- [ ] **Step 7: Run all agent tests**

```bash
.venv/bin/python -m pytest tests/agent/ -v --tb=short
```

Fix any failures.

- [ ] **Step 8: Commit**

```bash
git add agiwo/agent/models/log.py agiwo/agent/models/stream.py agiwo/agent/models/runtime_decision.py agiwo/agent/storage/serialization.py agiwo/agent/trace_writer.py agiwo/agent/hooks.py agiwo/agent/__init__.py
git commit -m "feat: add StepBack log/stream/decision types alongside old step-back types"
```

---

## Phase 8: Core Integration — wire ReviewBatch into run_tool_batch

### Task 10: Replace ReviewBatch with ReviewBatch in run_tool_batch.py

**Files:**
- Modify: `agiwo/agent/run_tool_batch.py` — swap `ReviewBatch` for `ReviewBatch`, remove `rebuild_messages` call
- Modify: `agiwo/agent/runtime/state_writer.py` — add `record_step_back_applied` method
- Test: `tests/agent/test_run_tool_batch_review.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/agent/test_run_tool_batch_review.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agiwo.agent.run_tool_batch import execute_tool_batch_cycle


class TestExecuteToolBatchCycleWithReview:
    @pytest.mark.asyncio
    async def test_review_batch_injects_system_review_on_trigger(self):
        """When tool results come back and a review interval is met,
        <system-review> should be injected into the final tool result content."""
        # Setup context, runtime, tool calls, mock execute_tool_batch
        # Verify that ReviewBatch is used, not ReviewBatch
        # Verify no rebuild_messages call
        pass  # Detailed implementation in actual step

    @pytest.mark.asyncio
    async def test_step_back_applies_targeted_content_updates(self):
        """When agent calls review_trajectory(aligned=false), content updates
        are applied as targeted replacements, not rebuild_messages."""
        pass

    @pytest.mark.asyncio
    async def test_review_tool_call_removed_after_processing(self):
        """review_trajectory's own messages are removed after step-back."""
        pass
```

Note: This is a high-level integration test. Write detailed test code in the actual step based on the shape of real context/runtime objects.

- [ ] **Step 2: Modify `agiwo/agent/run_tool_batch.py`**

Read current file (98 lines). The key changes:

**Change import** (line 8):
```python
# OLD:
from agiwo.agent.review import ReviewBatch
# NEW:
from agiwo.agent.review import ReviewBatch
```

**In `execute_tool_batch_cycle`** (line 35): Replace `ReviewBatch` with `ReviewBatch`:
```python
batch = ReviewBatch(context.config, context.ledger, runtime.tools_map)
```

**In process_result call** (line 52): Pass `current_seq`:
```python
content=batch.process_result(result, current_seq=tool_step.sequence),
```

**In `_apply_step-back_outcome` → rename to `_apply_review_outcome`:**

```python
async def _apply_review_outcome(
    context: RunContext,
    batch: ReviewBatch,
    *,
    writer: RunStateWriter,
) -> None:
    outcome = await batch.finalize(
        storage=context.session_runtime.run_log_storage,
        session_id=context.session_id,
        run_id=context.run_id,
        agent_id=context.agent_id,
    )
    if not outcome.applied:
        return

    # KEY CHANGE: targeted content updates instead of rebuild_messages
    for i, msg in enumerate(outcome.messages):
        if i < len(context.ledger.messages):
            context.ledger.messages[i] = msg
        else:
            context.ledger.messages.append(msg)

    # Trim excess if outcome removed review_trajectory messages
    while len(context.ledger.messages) > len(outcome.messages):
        context.ledger.messages.pop()

    # Record step_back event to run log
    step_back_entries = await writer.record_step_back_applied(
        affected_count=outcome.affected_count,
        checkpoint_seq=outcome.checkpoint_seq,
        experience=outcome.experience or "",
    )
    await context.session_runtime.project_run_log_entries(
        step_back_entries,
        run_id=context.run_id,
        agent_id=context.agent_id,
        parent_run_id=context.parent_run_id,
        depth=context.depth,
    )
```

**Update call site** (line 63): Change from `_apply_step-back_outcome` to `_apply_review_outcome`.

- [ ] **Step 3: Add `record_step_back_applied` to `agiwo/agent/runtime/state_writer.py`**

```python
async def record_step_back_applied(
    self,
    *,
    affected_count: int,
    checkpoint_seq: int,
    experience: str,
) -> list[object]:
    return await self.append_entries(
        [
            build_step_back_applied_entry(
                self._state,
                sequence=await self._state.session_runtime.allocate_sequence(),
                affected_count=affected_count,
                checkpoint_seq=checkpoint_seq,
                experience=experience,
            )
        ]
    )
```

Add `build_step_back_applied_entry` function:
```python
def build_step_back_applied_entry(
    state: RunContext,
    *,
    sequence: int,
    affected_count: int,
    checkpoint_seq: int,
    experience: str,
) -> StepBackApplied:
    return StepBackApplied(
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        sequence=sequence,
        affected_count=affected_count,
        checkpoint_seq=checkpoint_seq,
        experience=experience,
    )
```

Update imports accordingly.

- [ ] **Step 4: Run integration tests**

```bash
.venv/bin/python -m pytest tests/agent/test_run_tool_batch_review.py -v
```

- [ ] **Step 5: Run all agent tests to check for regressions**

```bash
.venv/bin/python -m pytest tests/agent/ -v --tb=short
```

Fix any regressions. Tests that directly test `ReviewBatch` will fail — those will be addressed in the cleanup phase.

- [ ] **Step 6: Commit**

```bash
git add agiwo/agent/run_tool_batch.py agiwo/agent/runtime/state_writer.py tests/agent/test_run_tool_batch_review.py
git commit -m "feat: integrate ReviewBatch into run_tool_batch with targeted content updates"
```

---

## Phase 9: Cleanup — remove old step-back code

### Task 11: Delete the step-back package and clean up all references

**Files to delete:**
- `agiwo/agent/review/__init__.py`
- `agiwo/agent/review/triggers.py`
- `agiwo/agent/review/executor.py`
- `agiwo/agent/review/` directory
- `tests/agent/test_step-back.py`

**Files to modify (remove old type references):**
- `agiwo/agent/models/run.py` — remove the legacy review-tracking state and field from `RunLedger`
- `agiwo/agent/models/log.py` — remove `StepBackApplied`
- `agiwo/agent/models/stream.py` — remove `StepBackAppliedEvent`
- `agiwo/agent/models/runtime_decision.py` — remove `StepBackDecisionView`, remove `latest_step_back` from `RuntimeDecisionState`
- `agiwo/agent/hooks.py` — remove `BEFORE_REVIEW`/`AFTER_STEP_BACK` from `HookPhase`
- `agiwo/agent/runtime/state_writer.py` — remove `record_step_back_applied`/`build_step_back_applied_entry`
- `agiwo/agent/storage/serialization.py` — remove `StepBackApplied` handling, update `RuntimeDecisionState` construction
- `agiwo/agent/trace_writer.py` — remove `StepBackApplied` from type unions
- `agiwo/agent/__init__.py` — remove old step-back exports
- `agiwo/scheduler/runtime_tools.py` — remove `ReviewTrajectoryTool`
- `agiwo/scheduler/engine.py` — remove `ReviewTrajectoryTool` import and instantiation
- `console/server/services/runtime/session_view_service.py` — remove `latest_step_back` branch
- `scripts/repo_guard.py` — remove AGW045-049 rules, add new review rules

**Test files to modify:**
- `tests/scheduler/test_runtime_facts.py` — remove step-back references
- `tests/agent/test_runtime_decision_views.py` — update to use `StepBackDecisionView`
- `tests/agent/test_run_log_replay_parity.py` — remove step-back references
- `tests/observability/test_collector.py` — update to use `StepBackApplied`
- `console/tests/test_run_query_service.py` — update to use `StepBackDecisionView`

- [ ] **Step 1: Delete step-back package**

```bash
rm -rf agiwo/agent/review/
rm tests/agent/test_step-back.py
```

- [ ] **Step 2: Clean up `agiwo/agent/models/run.py`**

Remove the legacy review-tracking state class (lines 97-103) and its import. Remove the old review-tracking field from `RunLedger` (line 133).

- [ ] **Step 3: Clean up `agiwo/agent/models/log.py`**

Remove `StepBackApplied` class (lines 177-186). Keep `STEP_BACK_APPLIED` kind for backwards compatibility with existing stored log entries.

- [ ] **Step 4: Clean up `agiwo/agent/models/stream.py`**

Remove `StepBackAppliedEvent` class. Add `StepBackAppliedEvent` to `AgentStreamItem` TypeAlias if not already.

- [ ] **Step 5: Clean up `agiwo/agent/models/runtime_decision.py`**

Remove `StepBackDecisionView`. Remove `latest_step_back` from `RuntimeDecisionState`. Add `latest_step_back`.

- [ ] **Step 6: Clean up `agiwo/agent/hooks.py`**

Remove `BEFORE_REVIEW`/`AFTER_STEP_BACK` from `HookPhase`. Remove from `_DECISION_SUPPORT_PHASES` and `_DECISION_SUPPORT_ALLOWLISTS`. Add `BEFORE_REVIEW`/`AFTER_STEP_BACK` if not already done in Task 9.

- [ ] **Step 7: Clean up `agiwo/agent/runtime/state_writer.py`**

Remove `record_step_back_applied` and `build_step_back_applied_entry` functions.

- [ ] **Step 8: Clean up remaining files**

Update `storage/serialization.py`, `trace_writer.py`, `__init__.py`, `runtime_tools.py`, `engine.py`, console files, and repo_guard.py as described in the file list above.

- [ ] **Step 9: Update all test files**

Update test files listed above to use new types and remove step-back references.

- [ ] **Step 10: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```

Fix all remaining failures until green.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "refactor: remove step-back package, clean up all old type references"
```

---

## Phase 10: Console Update

### Task 12: Update console to show step-back decisions

**Files:**
- Modify: `console/server/services/runtime/session_view_service.py` — replace `latest_step_back` with `latest_step_back`

- [ ] **Step 1: Update `session_view_service.py`**

Read lines 258-279. Replace the `latest_step_back` block:

```python
# OLD:
if runtime_decisions.latest_step_back is not None:
    decision = runtime_decisions.latest_step_back
    decisions.append(
        RuntimeDecisionRecord(
            kind="step-back",
            ...
        )
    )

# NEW:
if runtime_decisions.latest_step_back is not None:
    decision = runtime_decisions.latest_step_back
    decisions.append(
        RuntimeDecisionRecord(
            kind="step_back",
            sequence=decision.sequence,
            run_id=decision.run_id,
            agent_id=decision.agent_id,
            created_at=decision.created_at,
            summary=(
                f"{decision.affected_count} results condensed, "
                f"checkpoint at seq {decision.checkpoint_seq}"
            ),
            details={
                "affected_count": decision.affected_count,
                "checkpoint_seq": decision.checkpoint_seq,
                "experience": decision.experience,
            },
        )
    )
```

- [ ] **Step 2: Update console tests**

Update `console/tests/test_run_query_service.py` to use `StepBackApplied`/`StepBackDecisionView`.

```bash
.venv/bin/python -m pytest console/tests/ -v --tb=short
```

- [ ] **Step 3: Commit**

```bash
git add console/server/services/runtime/session_view_service.py console/tests/test_run_query_service.py
git commit -m "feat: update console to show step_back decisions"
```

---

## Phase 11: Documentation & Final Validation

### Task 13: Update documentation and run full validation

- [ ] **Step 1: Update `docs/guides/context-optimization.md`**

Replace the "Tool Result StepBack" section with "Goal-Directed Review" covering:
- Milestones concept
- System-review mechanism
- Step-back execution
- Configuration fields
- Updated before/after examples
- Updated package structure

- [ ] **Step 2: Archive old design docs**

```bash
mkdir -p docs/plans/archived
mv docs/plans/2026-04-04-context-rollback-and-step-back-design.md docs/plans/archived/
mv docs/plans/2026-04-05-step-back-batch-refactor-design.md docs/plans/archived/
```

- [ ] **Step 3: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Run repo guard**

```bash
.venv/bin/python scripts/repo_guard.py
```

Expected: all rules PASS.

- [ ] **Step 5: Run linting**

```bash
ruff check agiwo/ console/server/ tests/
ruff format --check agiwo/ console/server/ tests/
```

Expected: no issues.

- [ ] **Step 6: Run contract checks**

```bash
lint-imports --config lint/importlinter_agiwo.ini
```

Expected: all contracts KEPT.

- [ ] **Step 7: Final commit**

```bash
git add docs/guides/context-optimization.md docs/plans/archived/
git commit -m "docs: update context optimization guide for goal-directed review"
```
