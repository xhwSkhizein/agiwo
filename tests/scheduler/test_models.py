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

    def test_waitset_empty_not_satisfied(self):
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=[],
            completed_ids=[],
        )
        assert not wc.is_satisfied(datetime.now(timezone.utc))

    def test_timer_not_satisfied(self):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        wc = WakeCondition(
            type=WakeType.TIMER,
            time_value=1,
            time_unit=TimeUnit.HOURS,
            wakeup_at=future,
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

    def test_periodic_satisfied(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        wc = WakeCondition(
            type=WakeType.PERIODIC,
            time_value=5,
            time_unit=TimeUnit.MINUTES,
            wakeup_at=past,
        )
        assert wc.is_satisfied(datetime.now(timezone.utc))

    def test_task_submitted_satisfied(self):
        wc = WakeCondition(
            type=WakeType.TASK_SUBMITTED,
            submitted_task="new task",
        )
        assert wc.is_satisfied(datetime.now(timezone.utc))

    def test_task_submitted_not_satisfied(self):
        wc = WakeCondition(
            type=WakeType.TASK_SUBMITTED,
        )
        assert not wc.is_satisfied(datetime.now(timezone.utc))

    def test_is_timed_out(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1"],
            timeout_at=past,
        )
        assert wc.is_timed_out(datetime.now(timezone.utc))

    def test_not_timed_out(self):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1"],
            timeout_at=future,
        )
        assert not wc.is_timed_out(datetime.now(timezone.utc))

    def test_store_codec_waitset(self):
        timeout = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1", "c2"],
            wait_mode=WaitMode.ANY,
            completed_ids=["c1"],
            timeout_at=timeout,
        )
        payload = serialize_wake_condition_for_store(wc)
        assert payload is not None
        restored = deserialize_wake_condition_for_store(payload)
        assert restored.type == WakeType.WAITSET
        assert restored.wait_for == ["c1", "c2"]
        assert restored.wait_mode == WaitMode.ANY
        assert restored.completed_ids == ["c1"]
        assert restored.timeout_at == timeout

    def test_store_codec_timer(self):
        wakeup = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        wc = WakeCondition(
            type=WakeType.TIMER,
            time_value=1.0,
            time_unit=TimeUnit.MINUTES,
            wakeup_at=wakeup,
        )
        payload = serialize_wake_condition_for_store(wc)
        assert payload is not None
        restored = deserialize_wake_condition_for_store(payload)
        assert restored.type == WakeType.TIMER
        assert restored.time_value == 1.0
        assert restored.time_unit == TimeUnit.MINUTES
        assert restored.wakeup_at == wakeup

    def test_store_codec_task_submitted(self):
        wc = WakeCondition(
            type=WakeType.TASK_SUBMITTED,
            submitted_task="do stuff",
        )
        payload = serialize_wake_condition_for_store(wc)
        assert payload is not None
        restored = deserialize_wake_condition_for_store(payload)
        assert restored.type == WakeType.TASK_SUBMITTED
        assert restored.submitted_task == "do stuff"

    def test_to_seconds_method(self):
        wc = WakeCondition(
            type=WakeType.TIMER,
            time_value=2.0,
            time_unit=TimeUnit.HOURS,
        )
        assert wc.to_seconds() == 7200.0

    def test_to_seconds_none_when_missing(self):
        wc = WakeCondition(type=WakeType.WAITSET)
        assert wc.to_seconds() is None


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
        assert state.parent_id is None
        assert state.config_overrides == {}
        assert state.wake_condition is None
        assert state.result_summary is None
        assert state.signal_propagated is False
        assert state.is_persistent is False
        assert state.depth == 0
        assert state.wake_count == 0

    def test_status_enum_values(self):
        assert AgentStateStatus.PENDING.value == "pending"
        assert AgentStateStatus.RUNNING.value == "running"
        assert AgentStateStatus.SLEEPING.value == "sleeping"
        assert AgentStateStatus.COMPLETED.value == "completed"
        assert AgentStateStatus.FAILED.value == "failed"


class TestAgentStateNewFields:
    def test_explain_field_default_none(self):
        state = AgentState(
            id="a1",
            session_id="s1",
            status=AgentStateStatus.RUNNING,
            task="task",
        )
        assert state.explain is None
        assert state.last_activity_at is None
        assert state.recent_steps is None

    def test_explain_field_set(self):
        state = AgentState(
            id="a1",
            session_id="s1",
            status=AgentStateStatus.SLEEPING,
            task="task",
            explain="Waiting for news search results",
        )
        assert state.explain == "Waiting for news search results"


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

    def test_scheduler_event_types(self):
        assert SchedulerEventType.CHILD_SLEEP_RESULT.value == "child_sleep_result"
        assert SchedulerEventType.CHILD_COMPLETED.value == "child_completed"
        assert SchedulerEventType.CHILD_FAILED.value == "child_failed"
        assert SchedulerEventType.HEALTH_WARNING.value == "health_warning"
        assert SchedulerEventType.USER_HINT.value == "user_hint"


class TestSchedulerConfig:
    def test_debounce_defaults(self):
        config = SchedulerConfig()
        assert config.event_debounce_min_count == 1
        assert config.event_debounce_max_wait_seconds == 30.0

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
        assert limits.health_check_threshold_seconds == 300.0
