"""Tests for scheduler tick planning."""

import asyncio
from datetime import datetime, timedelta, timezone

from agiwo.agent import ChannelContext, ContentPart, ContentType, UserMessage
from agiwo.scheduler.commands import DispatchReason
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler._tick import plan_tick
from agiwo.scheduler.formatting import build_events_message
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerConfig,
    SchedulerEventType,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.runtime_state import build_mailbox_input
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
            payload={"user_input": UserMessage.to_storage_value("first")},
            created_at=now - timedelta(seconds=30),
        ),
        PendingEvent(
            id="e2",
            target_agent_id="parent",
            session_id="sess",
            event_type=SchedulerEventType.USER_HINT,
            payload={"user_input": UserMessage.to_storage_value("second")},
            created_at=now - timedelta(seconds=20),
        ),
    ]

    actions = plan_tick(engine, [waiting], events, now=now)

    assert len(actions) == 1
    assert actions[0].reason == DispatchReason.WAKE_EVENTS
    assert [event.id for event in actions[0].events] == ["e1", "e2"]


def test_plan_tick_non_urgent_single_event_waits_for_debounce() -> None:
    """Baseline: a single non-urgent event inside the debounce window stays queued."""
    now = datetime.now(timezone.utc)
    engine = _engine(event_debounce_min_count=3)
    waiting = _state("parent", status=AgentStateStatus.WAITING)
    events = [
        PendingEvent(
            id="single",
            target_agent_id="parent",
            session_id="sess",
            event_type=SchedulerEventType.USER_HINT,
            payload={"user_input": UserMessage.to_storage_value("hi")},
            created_at=now,
        ),
    ]

    actions = plan_tick(engine, [waiting], events, now=now)

    assert actions == []


def test_plan_tick_urgent_user_hint_bypasses_debounce_for_waiting_state() -> None:
    """A single urgent USER_HINT must wake a WAITING root immediately."""
    now = datetime.now(timezone.utc)
    engine = _engine(event_debounce_min_count=3)
    waiting = _state("parent", status=AgentStateStatus.WAITING)
    events = [
        PendingEvent(
            id="urgent-1",
            target_agent_id="parent",
            session_id="sess",
            event_type=SchedulerEventType.USER_HINT,
            payload={"user_input": UserMessage.to_storage_value("wake up")},
            created_at=now,
            urgent=True,
        ),
    ]

    actions = plan_tick(engine, [waiting], events, now=now)

    assert len(actions) == 1
    assert actions[0].reason == DispatchReason.WAKE_EVENTS
    assert [event.id for event in actions[0].events] == ["urgent-1"]


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
        payload={"user_input": UserMessage.to_storage_value("second input")},
        created_at=now,
    )

    actions = plan_tick(engine, [pending_child, queued_root], [mailbox_event], now=now)

    assert [(action.state.id, action.reason) for action in actions] == [
        ("child-1", DispatchReason.CHILD_PENDING),
        ("root", DispatchReason.ROOT_QUEUED_INPUT),
    ]
    queued_action = actions[1]
    assert queued_action.input_override is not None
    merged_text = UserMessage.from_value(queued_action.input_override).extract_text()
    assert "first input" in merged_text
    assert "second input" in merged_text


def _make_user_hint_event(
    event_id: str,
    user_input,
    *,
    created_at: datetime | None = None,
) -> PendingEvent:
    return PendingEvent(
        id=event_id,
        target_agent_id="root",
        session_id="sess",
        event_type=SchedulerEventType.USER_HINT,
        payload={"user_input": UserMessage.to_storage_value(user_input)},
        created_at=created_at or datetime.now(timezone.utc),
    )


def test_build_mailbox_input_preserves_multimodal_parts_and_channel_context() -> None:
    """USER_HINT attachments and ChannelContext must flow into the next run."""
    hint_message = UserMessage(
        content=[
            ContentPart(type=ContentType.TEXT, text="please review the image"),
            ContentPart(
                type=ContentType.IMAGE,
                url="/tmp/image.png",
                mime_type="image/png",
                metadata={"name": "image.png", "size": 123, "source": "feishu"},
            ),
        ],
        context=ChannelContext(
            source="feishu",
            metadata={"chat_id": "oc-123", "trigger_user": "alice"},
        ),
    )
    pending_input = "pending task"
    events = (_make_user_hint_event("h-1", hint_message),)

    merged = build_mailbox_input(pending_input, events)

    assert isinstance(merged, UserMessage)
    merged_text = merged.extract_text()
    assert "pending task" in merged_text
    assert "please review the image" in merged_text

    # Image part must survive intact.
    image_parts = [p for p in merged.content if p.type == ContentType.IMAGE]
    assert len(image_parts) == 1
    assert image_parts[0].url == "/tmp/image.png"
    assert image_parts[0].metadata["source"] == "feishu"

    # ChannelContext falls back to the hint when the pending input has none.
    assert merged.context is not None
    assert merged.context.source == "feishu"
    assert merged.context.metadata["chat_id"] == "oc-123"


def test_build_events_message_preserves_multimodal_and_channel_context() -> None:
    hint_message = UserMessage(
        content=[
            ContentPart(type=ContentType.TEXT, text="see attachment"),
            ContentPart(
                type=ContentType.FILE,
                url="/tmp/report.pdf",
                mime_type="application/pdf",
                metadata={"name": "report.pdf"},
            ),
        ],
        context=ChannelContext(
            source="feishu",
            metadata={"chat_id": "oc-456"},
        ),
    )
    events = (
        _make_user_hint_event("hint-1", hint_message),
        PendingEvent(
            id="child-1",
            target_agent_id="root",
            session_id="sess",
            event_type=SchedulerEventType.CHILD_COMPLETED,
            payload={"result": "done", "child_agent_id": "worker-1"},
            created_at=datetime.now(timezone.utc),
            source_agent_id="worker-1",
        ),
    )

    merged = build_events_message(events)

    assert isinstance(merged, UserMessage)
    text = merged.extract_text()
    assert "User hint: see attachment" in text
    assert "Child Completed" in text
    assert any(p.type == ContentType.FILE for p in merged.content)
    assert merged.context is not None
    assert merged.context.source == "feishu"
