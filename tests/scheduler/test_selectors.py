"""Tests for scheduler tick planning."""

import asyncio
from datetime import datetime, timedelta, timezone

from agiwo.scheduler.commands import DispatchReason
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler._tick import plan_tick
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerConfig,
    SchedulerEventType,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store.memory import InMemoryAgentStateStorage


def _state(id: str, *, status: AgentStateStatus, **kwargs) -> AgentState:
    return AgentState(
        id=id,
        session_id="sess",
        status=status,
        task="task",
        **kwargs,
    )


def _engine(*, event_debounce_min_count: int = 2) -> Scheduler:
    return Scheduler(
        store=InMemoryAgentStateStorage(),
        config=SchedulerConfig(event_debounce_min_count=event_debounce_min_count),
        semaphore=asyncio.Semaphore(1),
    )


def test_plan_tick_marks_ready_waiting_state() -> None:
    now = datetime.now(timezone.utc)
    engine = _engine()
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

    actions = plan_tick(engine, [ready, blocked], [], now=now)

    assert [(action.state.id, action.reason) for action in actions] == [
        ("ready", DispatchReason.WAKE_READY)
    ]


def test_plan_tick_marks_timed_out_waiting_state() -> None:
    now = datetime.now(timezone.utc)
    engine = _engine()
    timed_out = _state(
        "timed-out",
        status=AgentStateStatus.WAITING,
        wake_condition=WakeCondition(
            type=WakeType.WAITSET,
            timeout_at=now - timedelta(seconds=1),
        ),
    )

    actions = plan_tick(engine, [timed_out], [], now=now)

    assert [(action.state.id, action.reason) for action in actions] == [
        ("timed-out", DispatchReason.WAKE_TIMEOUT)
    ]


def test_plan_tick_debounces_pending_events_for_waiting_state() -> None:
    now = datetime.now(timezone.utc)
    engine = _engine(event_debounce_min_count=2)
    waiting = _state("parent", status=AgentStateStatus.WAITING)
    events = [
        PendingEvent(
            id="e1",
            target_agent_id="parent",
            session_id="sess",
            event_type=SchedulerEventType.USER_HINT,
            payload={"hint": "first"},
            created_at=now - timedelta(seconds=30),
        ),
        PendingEvent(
            id="e2",
            target_agent_id="parent",
            session_id="sess",
            event_type=SchedulerEventType.USER_HINT,
            payload={"hint": "second"},
            created_at=now - timedelta(seconds=20),
        ),
    ]

    actions = plan_tick(engine, [waiting], events, now=now)

    assert len(actions) == 1
    assert actions[0].reason == DispatchReason.WAKE_EVENTS
    assert [event.id for event in actions[0].events] == ["e1", "e2"]


def test_plan_tick_starts_pending_child_and_queued_root() -> None:
    now = datetime.now(timezone.utc)
    engine = _engine()
    pending_child = _state(
        "child-1",
        status=AgentStateStatus.PENDING,
        parent_id="root",
    )
    queued_root = _state(
        "root",
        status=AgentStateStatus.QUEUED,
        is_persistent=True,
        pending_input="first input",
    )
    mailbox_event = PendingEvent(
        id="e1",
        target_agent_id="root",
        session_id="sess",
        event_type=SchedulerEventType.USER_HINT,
        payload={"hint": "second input"},
        created_at=now,
    )

    actions = plan_tick(engine, [pending_child, queued_root], [mailbox_event], now=now)

    assert [(action.state.id, action.reason) for action in actions] == [
        ("child-1", DispatchReason.CHILD_PENDING),
        ("root", DispatchReason.ROOT_QUEUED_INPUT),
    ]
    queued_action = actions[1]
    assert queued_action.input_override is not None
    assert "first input" in queued_action.input_override
    assert "second input" in queued_action.input_override
