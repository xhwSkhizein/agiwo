"""Tests for AgentStateStorage implementations."""

from datetime import datetime, timedelta, timezone

import pytest

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
    TimeUnit,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store import InMemoryAgentStateStorage, SQLiteAgentStateStorage


@pytest.fixture
def store():
    return InMemoryAgentStateStorage()


def _make_state(
    id: str = "agent-1",
    session_id: str = "sess-1",
    parent_id: str | None = "parent",
    status: AgentStateStatus = AgentStateStatus.PENDING,
    task: str = "do something",
    **kwargs,
) -> AgentState:
    return AgentState(
        id=id,
        session_id=session_id,
        status=status,
        task=task,
        parent_id=parent_id,
        **kwargs,
    )


class TestInMemorySaveAndGet:
    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        state = _make_state()
        await store.save_state(state)
        retrieved = await store.get_state("agent-1")
        assert retrieved is not None
        assert retrieved.id == "agent-1"
        assert retrieved.status == AgentStateStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        result = await store.get_state("nope")
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert(self, store):
        state = _make_state(status=AgentStateStatus.PENDING)
        await store.save_state(state)
        state.status = AgentStateStatus.RUNNING
        await store.save_state(state)
        retrieved = await store.get_state("agent-1")
        assert retrieved is not None
        assert retrieved.status == AgentStateStatus.RUNNING


class TestInMemoryUpdateStatus:
    @pytest.mark.asyncio
    async def test_update_status(self, store):
        await store.save_state(_make_state())
        await store.update_status("agent-1", AgentStateStatus.RUNNING)
        state = await store.get_state("agent-1")
        assert state is not None
        assert state.status == AgentStateStatus.RUNNING

    @pytest.mark.asyncio
    async def test_update_status_with_wake_condition(self, store):
        await store.save_state(_make_state())
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1", "c2", "c3"],
        )
        await store.update_status(
            "agent-1", AgentStateStatus.SLEEPING, wake_condition=wc
        )
        state = await store.get_state("agent-1")
        assert state is not None
        assert state.status == AgentStateStatus.SLEEPING
        assert state.wake_condition is not None
        assert state.wake_condition.wait_for == ["c1", "c2", "c3"]

    @pytest.mark.asyncio
    async def test_update_status_with_result_summary(self, store):
        await store.save_state(_make_state())
        await store.update_status(
            "agent-1",
            AgentStateStatus.COMPLETED,
            result_summary="Task done",
        )
        state = await store.get_state("agent-1")
        assert state is not None
        assert state.result_summary == "Task done"

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, store):
        await store.update_status("nope", AgentStateStatus.RUNNING)


class TestInMemoryQueries:
    @pytest.mark.asyncio
    async def test_get_states_by_parent(self, store):
        await store.save_state(_make_state(id="child-1", parent_id="root"))
        await store.save_state(_make_state(id="child-2", parent_id="root"))
        await store.save_state(_make_state(id="child-3", parent_id="other"))
        children = await store.get_states_by_parent("root")
        assert len(children) == 2

    @pytest.mark.asyncio
    async def test_find_pending(self, store):
        await store.save_state(_make_state(id="a", status=AgentStateStatus.PENDING))
        await store.save_state(_make_state(id="b", status=AgentStateStatus.RUNNING))
        await store.save_state(_make_state(id="c", status=AgentStateStatus.PENDING))
        pending = await store.find_pending()
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_find_wakeable_waitset(self, store):
        state = _make_state(id="sleeper", status=AgentStateStatus.SLEEPING)
        state.wake_condition = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1", "c2"],
            completed_ids=["c1", "c2"],
        )
        await store.save_state(state)
        wakeable = await store.find_wakeable(datetime.now(timezone.utc))
        assert len(wakeable) == 1
        assert wakeable[0].id == "sleeper"

    @pytest.mark.asyncio
    async def test_find_wakeable_timer(self, store):
        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        state = _make_state(id="delayed", status=AgentStateStatus.SLEEPING)
        state.wake_condition = WakeCondition(
            type=WakeType.TIMER,
            time_value=1,
            time_unit=TimeUnit.MINUTES,
            wakeup_at=past,
        )
        await store.save_state(state)
        wakeable = await store.find_wakeable(datetime.now(timezone.utc))
        assert len(wakeable) == 1

    @pytest.mark.asyncio
    async def test_find_wakeable_not_yet(self, store):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        state = _make_state(id="waiting", status=AgentStateStatus.SLEEPING)
        state.wake_condition = WakeCondition(
            type=WakeType.TIMER,
            time_value=1,
            time_unit=TimeUnit.HOURS,
            wakeup_at=future,
        )
        await store.save_state(state)
        wakeable = await store.find_wakeable(datetime.now(timezone.utc))
        assert len(wakeable) == 0

    @pytest.mark.asyncio
    async def test_find_wakeable_task_submitted(self, store):
        state = _make_state(id="persistent", status=AgentStateStatus.SLEEPING)
        state.wake_condition = WakeCondition(
            type=WakeType.TASK_SUBMITTED,
            submitted_task="new work",
        )
        await store.save_state(state)
        wakeable = await store.find_wakeable(datetime.now(timezone.utc))
        assert len(wakeable) == 1


class TestInMemorySignalPropagation:
    @pytest.mark.asyncio
    async def test_find_unpropagated_completed(self, store):
        s1 = _make_state(
            id="done-1",
            status=AgentStateStatus.COMPLETED,
            parent_id="root",
        )
        s2 = _make_state(
            id="done-2",
            status=AgentStateStatus.COMPLETED,
            parent_id="root",
        )
        s2.signal_propagated = True
        s3 = _make_state(
            id="done-3",
            status=AgentStateStatus.COMPLETED,
            parent_id=None,
        )
        await store.save_state(s1)
        await store.save_state(s2)
        await store.save_state(s3)
        unprop = await store.find_unpropagated_completed()
        assert len(unprop) == 1
        assert unprop[0].id == "done-1"

    @pytest.mark.asyncio
    async def test_find_unpropagated_includes_failed(self, store):
        s = _make_state(
            id="failed-1",
            status=AgentStateStatus.FAILED,
            parent_id="root",
        )
        await store.save_state(s)
        unprop = await store.find_unpropagated_completed()
        assert len(unprop) == 1

    @pytest.mark.asyncio
    async def test_mark_child_completed(self, store):
        parent = _make_state(id="root", status=AgentStateStatus.SLEEPING)
        parent.wake_condition = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1", "c2", "c3"],
            completed_ids=[],
        )
        await store.save_state(parent)
        await store.mark_child_completed("root", "c1")
        await store.mark_child_completed("root", "c2")
        await store.mark_child_completed("root", "c1")  # duplicate should be ignored
        state = await store.get_state("root")
        assert state is not None
        assert state.wake_condition is not None
        assert state.wake_condition.completed_ids == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_mark_propagated(self, store):
        s = _make_state(id="child", status=AgentStateStatus.COMPLETED, parent_id="root")
        await store.save_state(s)
        await store.mark_propagated("child")
        state = await store.get_state("child")
        assert state is not None
        assert state.signal_propagated is True


class TestInMemoryTimeout:
    @pytest.mark.asyncio
    async def test_find_timed_out(self, store):
        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        state = _make_state(id="timed-out", status=AgentStateStatus.SLEEPING)
        state.wake_condition = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1"],
            timeout_at=past,
        )
        await store.save_state(state)
        timed_out = await store.find_timed_out(datetime.now(timezone.utc))
        assert len(timed_out) == 1
        assert timed_out[0].id == "timed-out"

    @pytest.mark.asyncio
    async def test_find_timed_out_excludes_not_yet(self, store):
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        state = _make_state(id="not-yet", status=AgentStateStatus.SLEEPING)
        state.wake_condition = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1"],
            timeout_at=future,
        )
        await store.save_state(state)
        timed_out = await store.find_timed_out(datetime.now(timezone.utc))
        assert len(timed_out) == 0


class TestInMemoryWakeCount:
    @pytest.mark.asyncio
    async def test_increment_wake_count(self, store):
        state = _make_state(id="agent-1")
        await store.save_state(state)
        await store.increment_wake_count("agent-1")
        await store.increment_wake_count("agent-1")
        s = await store.get_state("agent-1")
        assert s is not None
        assert s.wake_count == 2


def _make_event(
    id: str = "evt-1",
    target_agent_id: str = "parent",
    session_id: str = "sess-1",
    event_type: SchedulerEventType = SchedulerEventType.CHILD_COMPLETED,
    payload: dict | None = None,
    source_agent_id: str | None = "child",
    created_at: datetime | None = None,
) -> PendingEvent:
    return PendingEvent(
        id=id,
        target_agent_id=target_agent_id,
        session_id=session_id,
        event_type=event_type,
        payload=payload or {"result": "done"},
        source_agent_id=source_agent_id,
        created_at=created_at or datetime.now(timezone.utc),
    )


class TestInMemoryPendingEvents:
    @pytest.mark.asyncio
    async def test_save_and_get_event(self, store):
        event = _make_event()
        await store.save_event(event)
        events = await store.get_pending_events("parent", "sess-1")
        assert len(events) == 1
        assert events[0].id == "evt-1"
        assert events[0].event_type == SchedulerEventType.CHILD_COMPLETED

    @pytest.mark.asyncio
    async def test_get_events_filters_by_target_and_session(self, store):
        await store.save_event(_make_event(id="e1", target_agent_id="a", session_id="s1"))
        await store.save_event(_make_event(id="e2", target_agent_id="b", session_id="s1"))
        await store.save_event(_make_event(id="e3", target_agent_id="a", session_id="s2"))
        events_a_s1 = await store.get_pending_events("a", "s1")
        assert len(events_a_s1) == 1
        assert events_a_s1[0].id == "e1"

    @pytest.mark.asyncio
    async def test_delete_events(self, store):
        await store.save_event(_make_event(id="e1"))
        await store.save_event(_make_event(id="e2"))
        await store.delete_events(["e1"])
        events = await store.get_pending_events("parent", "sess-1")
        assert len(events) == 1
        assert events[0].id == "e2"

    @pytest.mark.asyncio
    async def test_find_agents_with_debounced_events_by_count(self, store):
        # 3 events → should trigger with min_count=3
        await store.save_event(_make_event(id="e1"))
        await store.save_event(_make_event(id="e2"))
        await store.save_event(_make_event(id="e3"))
        now = datetime.now(timezone.utc)
        result = await store.find_agents_with_debounced_events(3, 60.0, now)
        # Returns list of (agent_id, session_id) tuples
        assert ("parent", "sess-1") in result

    @pytest.mark.asyncio
    async def test_find_agents_with_debounced_events_by_time(self, store):
        # 1 event older than 10s → triggers when max_wait=5s
        old_ts = datetime.now(timezone.utc) - timedelta(seconds=10)
        await store.save_event(_make_event(id="e1", created_at=old_ts))
        now = datetime.now(timezone.utc)
        # min_count=99 (won't trigger by count), max_wait=5 (will trigger by time)
        result = await store.find_agents_with_debounced_events(99, 5.0, now)
        # Returns list of (agent_id, session_id) tuples
        assert ("parent", "sess-1") in result

    @pytest.mark.asyncio
    async def test_has_recent_health_warning_false_when_none(self, store):
        now = datetime.now(timezone.utc)
        result = await store.has_recent_health_warning("parent", "child", 300.0, now)
        assert result is False

    @pytest.mark.asyncio
    async def test_has_recent_health_warning_true_when_recent(self, store):
        now = datetime.now(timezone.utc)
        event = _make_event(
            id="hw1",
            event_type=SchedulerEventType.HEALTH_WARNING,
            target_agent_id="parent",
            source_agent_id="child",
            created_at=now - timedelta(seconds=10),
        )
        await store.save_event(event)
        result = await store.has_recent_health_warning("parent", "child", 300.0, now)
        assert result is True

    @pytest.mark.asyncio
    async def test_update_status_with_explain(self, store):
        state = _make_state(id="a1", status=AgentStateStatus.RUNNING)
        await store.save_state(state)
        await store.update_status("a1", AgentStateStatus.SLEEPING, explain="Waiting for user")
        updated = await store.get_state("a1")
        assert updated is not None
        assert updated.explain == "Waiting for user"

    @pytest.mark.asyncio
    async def test_update_status_with_recent_steps(self, store):
        state = _make_state(id="a1", status=AgentStateStatus.RUNNING)
        await store.save_state(state)
        steps = [{"role": "assistant", "timestamp": "2026-01-01T00:00:00", "tool_calls": ["web_search"]}]
        await store.update_status("a1", AgentStateStatus.RUNNING, recent_steps=steps)
        updated = await store.get_state("a1")
        assert updated is not None
        assert updated.recent_steps == steps

    @pytest.mark.asyncio
    async def test_find_running(self, store):
        await store.save_state(_make_state(id="r1", status=AgentStateStatus.RUNNING, parent_id=None))
        await store.save_state(_make_state(id="s1", status=AgentStateStatus.SLEEPING, parent_id=None))
        running = await store.find_running()
        ids = [s.id for s in running]
        assert "r1" in ids
        assert "s1" not in ids


class TestSQLiteAgentStateStorage:
    @pytest.mark.asyncio
    async def test_save_and_get_round_trip(self, tmp_path):
        store = SQLiteAgentStateStorage(str(tmp_path / "scheduler.db"))
        state = _make_state(
            id="sqlite-agent",
            status=AgentStateStatus.SLEEPING,
            explain="waiting for child",
        )
        state.wake_condition = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["child-1"],
            completed_ids=["child-1"],
        )

        await store.save_state(state)
        retrieved = await store.get_state("sqlite-agent")

        assert retrieved is not None
        assert retrieved.id == "sqlite-agent"
        assert retrieved.status == AgentStateStatus.SLEEPING
        assert retrieved.explain == "waiting for child"
        assert retrieved.wake_condition is not None
        assert retrieved.wake_condition.completed_ids == ["child-1"]

        await store.close()
