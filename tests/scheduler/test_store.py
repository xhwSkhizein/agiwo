"""Tests for AgentStateStorage implementations."""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store import InMemoryAgentStateStorage


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
