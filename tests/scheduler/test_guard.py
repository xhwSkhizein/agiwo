"""Tests for TaskGuard."""

import pytest
from datetime import datetime, timedelta, timezone

from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    TaskLimits,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store import InMemoryAgentStateStorage


@pytest.fixture
def store():
    return InMemoryAgentStateStorage()


def _make_state(
    id: str = "agent-1",
    depth: int = 0,
    wake_count: int = 0,
    status: AgentStateStatus = AgentStateStatus.RUNNING,
    parent_state_id: str | None = None,
    **kwargs,
) -> AgentState:
    return AgentState(
        id=id,
        session_id="sess",
        agent_id=id,
        parent_agent_id=id,
        parent_state_id=parent_state_id,
        status=status,
        task="task",
        depth=depth,
        wake_count=wake_count,
        **kwargs,
    )


class TestCheckSpawn:
    @pytest.mark.asyncio
    async def test_allowed(self, store):
        guard = TaskGuard(TaskLimits(max_depth=5, max_children_per_agent=10), store)
        state = _make_state(depth=2)
        await store.save_state(state)
        result = await guard.check_spawn(state)
        assert result is None

    @pytest.mark.asyncio
    async def test_rejected_max_depth(self, store):
        guard = TaskGuard(TaskLimits(max_depth=3), store)
        state = _make_state(depth=3)
        await store.save_state(state)
        result = await guard.check_spawn(state)
        assert result is not None
        assert "depth" in result.lower()

    @pytest.mark.asyncio
    async def test_rejected_max_children(self, store):
        guard = TaskGuard(TaskLimits(max_children_per_agent=2), store)
        parent = _make_state(id="parent", depth=0)
        await store.save_state(parent)
        c1 = _make_state(id="c1", parent_state_id="parent", status=AgentStateStatus.RUNNING)
        c2 = _make_state(id="c2", parent_state_id="parent", status=AgentStateStatus.PENDING)
        await store.save_state(c1)
        await store.save_state(c2)

        result = await guard.check_spawn(parent)
        assert result is not None
        assert "children" in result.lower()

    @pytest.mark.asyncio
    async def test_completed_children_dont_count(self, store):
        guard = TaskGuard(TaskLimits(max_children_per_agent=2), store)
        parent = _make_state(id="parent", depth=0)
        await store.save_state(parent)
        c1 = _make_state(id="c1", parent_state_id="parent", status=AgentStateStatus.COMPLETED)
        c2 = _make_state(id="c2", parent_state_id="parent", status=AgentStateStatus.RUNNING)
        await store.save_state(c1)
        await store.save_state(c2)

        result = await guard.check_spawn(parent)
        assert result is None


class TestCheckWake:
    @pytest.mark.asyncio
    async def test_allowed(self, store):
        guard = TaskGuard(TaskLimits(max_wake_count=20), store)
        state = _make_state(wake_count=5)
        result = await guard.check_wake(state)
        assert result is None

    @pytest.mark.asyncio
    async def test_rejected_max_wake_count(self, store):
        guard = TaskGuard(TaskLimits(max_wake_count=10), store)
        state = _make_state(wake_count=10)
        result = await guard.check_wake(state)
        assert result is not None
        assert "wake count" in result.lower()


class TestFindTimedOut:
    @pytest.mark.asyncio
    async def test_finds_timed_out(self, store):
        guard = TaskGuard(TaskLimits(), store)
        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        state = _make_state(id="timed", status=AgentStateStatus.SLEEPING)
        state.wake_condition = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1"],
            timeout_at=past,
        )
        await store.save_state(state)

        result = await guard.find_timed_out(datetime.now(timezone.utc))
        assert len(result) == 1
        assert result[0].id == "timed"

    @pytest.mark.asyncio
    async def test_excludes_not_timed_out(self, store):
        guard = TaskGuard(TaskLimits(), store)
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        state = _make_state(id="waiting", status=AgentStateStatus.SLEEPING)
        state.wake_condition = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["c1"],
            timeout_at=future,
        )
        await store.save_state(state)

        result = await guard.find_timed_out(datetime.now(timezone.utc))
        assert len(result) == 0
