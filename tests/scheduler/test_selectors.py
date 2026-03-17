"""Tests for scheduler selector helpers."""

from datetime import datetime, timedelta, timezone

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.selectors import (
    select_debounced_event_targets,
    select_ready_waiting_states,
    select_timed_out_states,
    select_unpropagated_children,
)


def _state(id: str, *, status: AgentStateStatus, **kwargs) -> AgentState:
    return AgentState(
        id=id,
        session_id="sess",
        status=status,
        task="task",
        **kwargs,
    )


def test_select_ready_waiting_states() -> None:
    now = datetime.now(timezone.utc)
    ready = _state(
        "ready",
        status=AgentStateStatus.WAITING,
        wake_condition=WakeCondition(
            type=WakeType.TIMER,
            wakeup_at=now - timedelta(seconds=1),
        ),
    )
    blocked = _state(
        "blocked",
        status=AgentStateStatus.WAITING,
        wake_condition=WakeCondition(
            type=WakeType.TIMER,
            wakeup_at=now + timedelta(seconds=1),
        ),
    )
    result = select_ready_waiting_states([ready, blocked], now=now)
    assert [state.id for state in result] == ["ready"]


def test_select_timed_out_states() -> None:
    now = datetime.now(timezone.utc)
    timed_out = _state(
        "timed-out",
        status=AgentStateStatus.WAITING,
        wake_condition=WakeCondition(
            type=WakeType.WAITSET,
            timeout_at=now - timedelta(seconds=1),
        ),
    )
    result = select_timed_out_states([timed_out], now=now)
    assert [state.id for state in result] == ["timed-out"]


def test_select_debounced_event_targets() -> None:
    now = datetime.now(timezone.utc)
    events = [
        PendingEvent(
            id="e1",
            target_agent_id="a",
            session_id="s1",
            event_type=SchedulerEventType.USER_HINT,
            payload={},
            created_at=now - timedelta(seconds=30),
        ),
        PendingEvent(
            id="e2",
            target_agent_id="a",
            session_id="s1",
            event_type=SchedulerEventType.USER_HINT,
            payload={},
            created_at=now - timedelta(seconds=20),
        ),
    ]
    result = select_debounced_event_targets(
        events,
        min_count=2,
        max_wait_seconds=60,
        now=now,
    )
    assert result == [("a", "s1")]


def test_select_unpropagated_children() -> None:
    done = _state(
        "child-1",
        status=AgentStateStatus.COMPLETED,
        parent_id="root",
        signal_propagated=False,
    )
    ignored = _state(
        "child-2",
        status=AgentStateStatus.RUNNING,
        parent_id="root",
        signal_propagated=False,
    )
    result = select_unpropagated_children([done, ignored])
    assert [state.id for state in result] == ["child-1"]
