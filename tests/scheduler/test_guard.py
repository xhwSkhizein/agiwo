"""Tests for TaskGuard."""

import pytest

from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import AgentState, AgentStateStatus, TaskLimits
from agiwo.scheduler.store import InMemoryAgentStateStorage


@pytest.fixture
def store():
    return InMemoryAgentStateStorage()


def _make_state(
    id: str = "agent-1",
    depth: int = 0,
    wake_count: int = 0,
    status: AgentStateStatus = AgentStateStatus.RUNNING,
    parent_id: str | None = None,
    **kwargs,
) -> AgentState:
    return AgentState(
        id=id,
        session_id="sess",
        status=status,
        task="task",
        parent_id=parent_id,
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
        assert await guard.check_spawn(state) is None

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
        await store.save_state(
            _make_state(id="c1", parent_id="parent", status=AgentStateStatus.RUNNING)
        )
        await store.save_state(
            _make_state(id="c2", parent_id="parent", status=AgentStateStatus.PENDING)
        )
        result = await guard.check_spawn(parent)
        assert result is not None
        assert "children" in result.lower()

    @pytest.mark.asyncio
    async def test_idle_children_count_as_active(self, store):
        guard = TaskGuard(TaskLimits(max_children_per_agent=1), store)
        parent = _make_state(id="parent", depth=0)
        await store.save_state(parent)
        await store.save_state(
            _make_state(id="c1", parent_id="parent", status=AgentStateStatus.IDLE)
        )
        result = await guard.check_spawn(parent)
        assert result is not None


class TestCheckWake:
    @pytest.mark.asyncio
    async def test_root_not_limited(self, store):
        guard = TaskGuard(TaskLimits(max_wake_count=1), store)
        state = _make_state(wake_count=100, parent_id=None)
        assert await guard.check_wake(state) is None

    @pytest.mark.asyncio
    async def test_rejected_max_wake_count(self, store):
        guard = TaskGuard(TaskLimits(max_wake_count=10), store)
        state = _make_state(wake_count=10, parent_id="parent-1")
        result = await guard.check_wake(state)
        assert result is not None
        assert "wake count" in result.lower()
