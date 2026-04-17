"""Tests for scheduler data models."""

import copy
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

from agiwo.agent import TerminationReason
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    ChildAgentConfigOverrides,
    PendingEvent,
    SchedulerRunResult,
    SchedulerConfig,
    SchedulerEventType,
    TaskLimits,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
    to_seconds,
)
from agiwo.scheduler.formatting import build_fork_task_notice
from agiwo.scheduler.store.codec import (
    deserialize_child_agent_config_overrides,
    deserialize_wake_condition_for_store,
    serialize_child_agent_config_overrides,
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

    def test_with_running_clears_last_run_result(self):
        state = AgentState(
            id="root",
            session_id="sess-1",
            status=AgentStateStatus.IDLE,
            task="task",
            is_persistent=True,
            last_run_result=SchedulerRunResult(
                run_id="run-1",
                termination_reason=TerminationReason.COMPLETED,
                summary="done",
            ),
        )

        updated = state.with_running(task="next task")

        assert updated.status == AgentStateStatus.RUNNING
        assert updated.last_run_result is None

    def test_with_queued_preserves_last_run_result(self):
        result = SchedulerRunResult(
            run_id="run-2",
            termination_reason=TerminationReason.CANCELLED,
            error="cancelled by user",
        )
        state = AgentState(
            id="root",
            session_id="sess-1",
            status=AgentStateStatus.IDLE,
            task="task",
            is_persistent=True,
            last_run_result=result,
        )

        updated = state.with_queued(pending_input="next")

        assert updated.status == AgentStateStatus.QUEUED
        assert updated.last_run_result == result


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
        assert event.urgent is False

    def test_urgent_flag_defaults_false_and_can_be_set(self):
        now = datetime.now(timezone.utc)
        urgent_event = PendingEvent(
            id="evt-u",
            target_agent_id="root",
            session_id="sess",
            event_type=SchedulerEventType.USER_HINT,
            payload={"user_input": "hi"},
            created_at=now,
            urgent=True,
        )
        normal_event = PendingEvent(
            id="evt-n",
            target_agent_id="root",
            session_id="sess",
            event_type=SchedulerEventType.USER_HINT,
            payload={"user_input": "hi"},
            created_at=now,
        )
        assert urgent_event.urgent is True
        assert normal_event.urgent is False


class TestMappingProxyDeepcopy:
    """Regression guard against the old ``copy._deepcopy_dispatch`` hack."""

    def test_agent_state_deepcopy_preserves_frozen_mapping(self):
        state = AgentState(
            id="deep-1",
            session_id="sess",
            status=AgentStateStatus.RUNNING,
            task="task",
            config_overrides={"nested": {"key": "value"}, "list": [1, 2]},
        )
        cloned = copy.deepcopy(state)
        assert cloned is not state
        assert cloned.config_overrides["nested"]["key"] == "value"
        assert tuple(cloned.config_overrides["list"]) == (1, 2)

    def test_pending_event_deepcopy_preserves_payload(self):
        now = datetime.now(timezone.utc)
        event = PendingEvent(
            id="deep-2",
            target_agent_id="root",
            session_id="sess",
            event_type=SchedulerEventType.USER_HINT,
            payload={"user_input": "hi", "nested": {"k": "v"}},
            created_at=now,
        )
        cloned = copy.deepcopy(event)
        assert cloned is not event
        assert cloned.payload["user_input"] == "hi"
        assert cloned.payload["nested"]["k"] == "v"

    def test_mapping_proxy_not_in_global_dispatch(self):
        """The scheduler.models import must not mutate ``copy._deepcopy_dispatch``."""
        # Importing models already triggered any side effects; the dispatch
        # table should not carry a scheduler-specific handler anymore.
        assert MappingProxyType not in copy._deepcopy_dispatch


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


class TestChildAgentConfigOverridesCodec:
    def test_fork_flag_round_trip(self):
        overrides = ChildAgentConfigOverrides(fork=True)
        data = serialize_child_agent_config_overrides(overrides)
        assert data["fork"] is True
        restored = deserialize_child_agent_config_overrides(data)
        assert restored.fork is True

    def test_fork_flag_default_false(self):
        overrides = ChildAgentConfigOverrides()
        data = serialize_child_agent_config_overrides(overrides)
        assert "fork" not in data
        restored = deserialize_child_agent_config_overrides(data)
        assert restored.fork is False

    def test_fork_flag_missing_in_data(self):
        restored = deserialize_child_agent_config_overrides({"instruction": "hi"})
        assert restored.fork is False


class TestBuildForkTaskNotice:
    def test_wraps_task_with_notice(self):
        result = build_fork_task_notice("Analyze data")
        assert "<system-notice>" in result
        assert "forked child agent" in result
        assert "Do NOT use spawn_agent" in result
        assert result.endswith("Analyze data")
