"""Tests for scheduler data models."""

import pytest
from datetime import datetime, timedelta, timezone

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    TaskLimits,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
    to_seconds,
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

    def test_to_dict_and_from_dict_waitset(self):
        timeout = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1", "c2"],
            wait_mode=WaitMode.ANY,
            completed_ids=["c1"],
            timeout_at=timeout,
        )
        d = wc.to_dict()
        restored = WakeCondition.from_dict(d)
        assert restored.type == WakeType.WAITSET
        assert restored.wait_for == ["c1", "c2"]
        assert restored.wait_mode == WaitMode.ANY
        assert restored.completed_ids == ["c1"]
        assert restored.timeout_at == timeout

    def test_to_dict_and_from_dict_timer(self):
        wakeup = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        wc = WakeCondition(
            type=WakeType.TIMER,
            time_value=1.0,
            time_unit=TimeUnit.MINUTES,
            wakeup_at=wakeup,
        )
        d = wc.to_dict()
        restored = WakeCondition.from_dict(d)
        assert restored.type == WakeType.TIMER
        assert restored.time_value == 1.0
        assert restored.time_unit == TimeUnit.MINUTES
        assert restored.wakeup_at == wakeup

    def test_to_dict_and_from_dict_task_submitted(self):
        wc = WakeCondition(
            type=WakeType.TASK_SUBMITTED,
            submitted_task="do stuff",
        )
        d = wc.to_dict()
        restored = WakeCondition.from_dict(d)
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
            agent_id="agent-1",
            parent_agent_id="parent",
            status=AgentStateStatus.PENDING,
            task="do something",
        )
        assert state.id == "agent-1"
        assert state.status == AgentStateStatus.PENDING
        assert state.parent_state_id is None
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


class TestTaskLimits:
    def test_defaults(self):
        limits = TaskLimits()
        assert limits.max_depth == 5
        assert limits.max_children_per_agent == 10
        assert limits.default_wait_timeout == 600.0
        assert limits.max_wake_count == 20
