"""Tests for scheduler data models."""

import pytest
from datetime import datetime, timedelta, timezone

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    TimeUnit,
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
    def test_children_complete_not_satisfied(self):
        wc = WakeCondition(
            type=WakeType.CHILDREN_COMPLETE,
            total_children=3,
            completed_children=1,
        )
        assert not wc.is_satisfied(datetime.now(timezone.utc))

    def test_children_complete_satisfied(self):
        wc = WakeCondition(
            type=WakeType.CHILDREN_COMPLETE,
            total_children=3,
            completed_children=3,
        )
        assert wc.is_satisfied(datetime.now(timezone.utc))

    def test_children_complete_zero_total_not_satisfied(self):
        wc = WakeCondition(
            type=WakeType.CHILDREN_COMPLETE,
            total_children=0,
            completed_children=0,
        )
        assert not wc.is_satisfied(datetime.now(timezone.utc))

    def test_delay_not_satisfied(self):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        wc = WakeCondition(
            type=WakeType.DELAY,
            time_value=1,
            time_unit=TimeUnit.HOURS,
            wakeup_at=future,
        )
        assert not wc.is_satisfied(datetime.now(timezone.utc))

    def test_delay_satisfied(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        wc = WakeCondition(
            type=WakeType.DELAY,
            time_value=1,
            time_unit=TimeUnit.MINUTES,
            wakeup_at=past,
        )
        assert wc.is_satisfied(datetime.now(timezone.utc))

    def test_interval_satisfied(self):
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        wc = WakeCondition(
            type=WakeType.INTERVAL,
            time_value=5,
            time_unit=TimeUnit.MINUTES,
            wakeup_at=past,
        )
        assert wc.is_satisfied(datetime.now(timezone.utc))

    def test_to_dict_and_from_dict_children(self):
        wc = WakeCondition(
            type=WakeType.CHILDREN_COMPLETE,
            total_children=5,
            completed_children=2,
        )
        d = wc.to_dict()
        restored = WakeCondition.from_dict(d)
        assert restored.type == WakeType.CHILDREN_COMPLETE
        assert restored.total_children == 5
        assert restored.completed_children == 2

    def test_to_dict_and_from_dict_delay(self):
        wakeup = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        wc = WakeCondition(
            type=WakeType.DELAY,
            time_value=1.0,
            time_unit=TimeUnit.MINUTES,
            wakeup_at=wakeup,
        )
        d = wc.to_dict()
        restored = WakeCondition.from_dict(d)
        assert restored.type == WakeType.DELAY
        assert restored.time_value == 1.0
        assert restored.time_unit == TimeUnit.MINUTES
        assert restored.wakeup_at == wakeup

    def test_to_seconds_method(self):
        wc = WakeCondition(
            type=WakeType.DELAY,
            time_value=2.0,
            time_unit=TimeUnit.HOURS,
        )
        assert wc.to_seconds() == 7200.0

    def test_to_seconds_none_when_missing(self):
        wc = WakeCondition(type=WakeType.CHILDREN_COMPLETE)
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

    def test_status_enum_values(self):
        assert AgentStateStatus.PENDING.value == "pending"
        assert AgentStateStatus.RUNNING.value == "running"
        assert AgentStateStatus.SLEEPING.value == "sleeping"
        assert AgentStateStatus.COMPLETED.value == "completed"
        assert AgentStateStatus.FAILED.value == "failed"
