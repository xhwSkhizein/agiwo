"""Tests for scheduler data models."""

from datetime import datetime, timedelta, timezone

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerConfig,
    SchedulerEventType,
    TaskLimits,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
    to_seconds,
)
from agiwo.scheduler.store.codec import (
    deserialize_wake_condition_for_store,
    serialize_wake_condition_for_store,
)


class TestTimeUnit:
    def test_to_seconds_seconds(self):
        assert to_seconds(30, TimeUnit.SECONDS) == 30

    def test_to_seconds_minutes(self):
        assert to_seconds(2, TimeUnit.MINUTES) == 120

    def test_to_seconds_hours(self):
        assert to_seconds(1.5, TimeUnit.HOURS) == 5400


class TestWakeCondition:
    def test_waitset_all_not_satisfied(self):
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1", "c2", "c3"],
            wait_mode=WaitMode.ALL,
            completed_ids=["c1"],
        )
        assert not wc.is_satisfied(datetime.now(timezone.utc))

    def test_waitset_all_satisfied(self):
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1", "c2", "c3"],
            wait_mode=WaitMode.ALL,
            completed_ids=["c1", "c2", "c3"],
        )
        assert wc.is_satisfied(datetime.now(timezone.utc))

    def test_waitset_any_satisfied(self):
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1", "c2"],
            wait_mode=WaitMode.ANY,
            completed_ids=["c1"],
        )
        assert wc.is_satisfied(datetime.now(timezone.utc))

    def test_waitset_any_not_satisfied(self):
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1", "c2"],
            wait_mode=WaitMode.ANY,
            completed_ids=[],
        )
        assert not wc.is_satisfied(datetime.now(timezone.utc))

    def test_timer_satisfied(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        wc = WakeCondition(
            type=WakeType.TIMER,
            time_value=1,
            time_unit=TimeUnit.MINUTES,
            wakeup_at=past,
        )
        assert wc.is_satisfied(datetime.now(timezone.utc))

    def test_pending_events_not_self_satisfied(self):
        wc = WakeCondition(type=WakeType.PENDING_EVENTS)
        assert not wc.is_satisfied(datetime.now(timezone.utc))

    def test_is_timed_out(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1"],
            timeout_at=past,
        )
        assert wc.is_timed_out(datetime.now(timezone.utc))

    def test_store_codec_round_trip(self):
        timeout = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        wakeup = datetime(2026, 6, 1, 11, 0, 0, tzinfo=timezone.utc)
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1", "c2"],
            wait_mode=WaitMode.ANY,
            completed_ids=["c1"],
            wakeup_at=wakeup,
            timeout_at=timeout,
        )
        payload = serialize_wake_condition_for_store(wc)
        assert payload is not None
        restored = deserialize_wake_condition_for_store(payload)
        assert restored.type == WakeType.WAITSET
        assert restored.wait_for == ("c1", "c2")
        assert restored.wait_mode == WaitMode.ANY
        assert restored.completed_ids == ("c1",)
        assert restored.wakeup_at == wakeup
        assert restored.timeout_at == timeout

    def test_to_seconds_method(self):
        wc = WakeCondition(
            type=WakeType.TIMER,
            time_value=2.0,
            time_unit=TimeUnit.HOURS,
        )
        assert wc.to_seconds() == 7200.0


class TestAgentState:
    def test_create_with_defaults(self):
        state = AgentState(
            id="agent-1",
            session_id="sess-1",
            status=AgentStateStatus.PENDING,
            task="do something",
        )
        assert state.id == "agent-1"
        assert state.status == AgentStateStatus.PENDING
        assert state.pending_input is None
        assert state.is_persistent is False
        assert state.depth == 0
        assert state.wake_count == 0

    def test_status_enum_values(self):
        assert AgentStateStatus.PENDING.value == "pending"
        assert AgentStateStatus.RUNNING.value == "running"
        assert AgentStateStatus.WAITING.value == "waiting"
        assert AgentStateStatus.IDLE.value == "idle"
        assert AgentStateStatus.QUEUED.value == "queued"
        assert AgentStateStatus.COMPLETED.value == "completed"
        assert AgentStateStatus.FAILED.value == "failed"

    def test_root_helpers(self):
        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.IDLE,
            task="task",
            is_persistent=True,
        )
        assert state.is_idle_root()
        assert not state.is_queued_root()
        assert state.can_accept_enqueue_input()

    def test_queued_root_helper(self):
        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.QUEUED,
            task="task",
            pending_input="next",
            is_persistent=True,
        )
        assert state.is_queued_root()
        assert not state.can_accept_enqueue_input()


class TestPendingEvent:
    def test_create_event(self):
        now = datetime.now(timezone.utc)
        event = PendingEvent(
            id="evt-1",
            target_agent_id="parent-agent",
            session_id="sess-1",
            event_type=SchedulerEventType.CHILD_COMPLETED,
            payload={"result": "done"},
            created_at=now,
            source_agent_id="child-agent",
        )
        assert event.id == "evt-1"
        assert event.event_type == SchedulerEventType.CHILD_COMPLETED
        assert event.payload["result"] == "done"
        assert event.source_agent_id == "child-agent"


class TestSchedulerConfig:
    def test_debounce_defaults(self):
        config = SchedulerConfig()
        assert config.event_debounce_min_count == 3
        assert config.event_debounce_max_wait_seconds == 10.0

    def test_debounce_custom(self):
        config = SchedulerConfig(
            event_debounce_min_count=3,
            event_debounce_max_wait_seconds=60.0,
        )
        assert config.event_debounce_min_count == 3
        assert config.event_debounce_max_wait_seconds == 60.0


class TestTaskLimits:
    def test_defaults(self):
        limits = TaskLimits()
        assert limits.max_depth == 5
        assert limits.max_children_per_agent == 10
        assert limits.default_wait_timeout == 600.0
        assert limits.max_wake_count == 20
