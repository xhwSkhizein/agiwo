"""Tests for scheduling tools."""

import pytest
from datetime import datetime, timedelta, timezone

from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.schema import TerminationReason
from agiwo.agent.stream_channel import StreamChannel
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    TaskLimits,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store import InMemoryAgentStateStorage
from agiwo.scheduler.tools import SpawnAgentTool, SleepAndWaitTool, QuerySpawnedAgentTool


@pytest.fixture
def store():
    return InMemoryAgentStateStorage()


@pytest.fixture
def guard(store):
    return TaskGuard(TaskLimits(), store)


@pytest.fixture
def context():
    return ExecutionContext(
        session_id="sess-1",
        run_id="run-1",
        channel=StreamChannel(),
        agent_id="orch",
        sequence_counter=SessionSequenceCounter(0),
    )


async def _register_parent(store, agent_id="orch", depth=0):
    """Helper: create parent AgentState so spawn/sleep can look it up."""
    parent_state = AgentState(
        id=agent_id,
        session_id="sess-1",
        status=AgentStateStatus.RUNNING,
        task="root task",
        parent_id=None,
        depth=depth,
    )
    await store.save_state(parent_state)
    return parent_state


class TestSpawnAgentTool:
    @pytest.mark.asyncio
    async def test_spawn_creates_pending_state(self, store, guard, context):
        await _register_parent(store)
        tool = SpawnAgentTool(store, guard)
        result = await tool.execute(
            {"task": "Research topic A", "tool_call_id": "tc-1"},
            context,
        )
        assert result.is_success
        assert "Spawned child agent" in result.content

        child_id = result.output["child_id"]
        state = await store.get_state(child_id)
        assert state is not None
        assert state.status == AgentStateStatus.PENDING
        assert state.parent_id == "orch"
        assert state.task == "Research topic A"
        assert state.session_id == "sess-1"
        assert state.depth == 1

    @pytest.mark.asyncio
    async def test_spawn_with_custom_child_id(self, store, guard, context):
        await _register_parent(store)
        tool = SpawnAgentTool(store, guard)
        result = await tool.execute(
            {"task": "Task", "child_id": "my-child", "tool_call_id": "tc-1"},
            context,
        )
        assert result.output["child_id"] == "my-child"
        state = await store.get_state("my-child")
        assert state is not None

    @pytest.mark.asyncio
    async def test_spawn_with_system_prompt_override(self, store, guard, context):
        await _register_parent(store)
        tool = SpawnAgentTool(store, guard)
        result = await tool.execute(
            {
                "task": "Task",
                "system_prompt": "You are a specialist.",
                "tool_call_id": "tc-1",
            },
            context,
        )
        child_id = result.output["child_id"]
        state = await store.get_state(child_id)
        assert state is not None
        assert state.config_overrides["system_prompt"] == "You are a specialist."

    @pytest.mark.asyncio
    async def test_spawn_fails_without_agent_id(self, store, guard):
        ctx = ExecutionContext(
            session_id="sess-1",
            run_id="run-1",
            channel=StreamChannel(),
            agent_id=None,
            sequence_counter=SessionSequenceCounter(0),
        )
        tool = SpawnAgentTool(store, guard)
        result = await tool.execute({"task": "Task", "tool_call_id": "tc-1"}, ctx)
        assert not result.is_success

    @pytest.mark.asyncio
    async def test_spawn_generates_unique_ids(self, store, guard, context):
        await _register_parent(store)
        tool = SpawnAgentTool(store, guard)
        r1 = await tool.execute({"task": "A", "tool_call_id": "tc-1"}, context)
        r2 = await tool.execute({"task": "B", "tool_call_id": "tc-2"}, context)
        assert r1.output["child_id"] != r2.output["child_id"]

    @pytest.mark.asyncio
    async def test_spawn_rejected_max_depth(self, store, context):
        limits = TaskLimits(max_depth=2)
        guard = TaskGuard(limits, store)
        await _register_parent(store, depth=2)
        tool = SpawnAgentTool(store, guard)
        result = await tool.execute(
            {"task": "Deep task", "tool_call_id": "tc-1"},
            context,
        )
        assert not result.is_success
        assert "Spawn rejected" in result.content

    @pytest.mark.asyncio
    async def test_spawn_rejected_max_children(self, store, context):
        limits = TaskLimits(max_children_per_agent=2)
        guard = TaskGuard(limits, store)
        await _register_parent(store)
        tool = SpawnAgentTool(store, guard)
        await tool.execute({"task": "A", "child_id": "c1", "tool_call_id": "tc-1"}, context)
        await tool.execute({"task": "B", "child_id": "c2", "tool_call_id": "tc-2"}, context)
        result = await tool.execute({"task": "C", "child_id": "c3", "tool_call_id": "tc-3"}, context)
        assert not result.is_success
        assert "Spawn rejected" in result.content

    @pytest.mark.asyncio
    async def test_spawn_inherits_depth(self, store, guard, context):
        await _register_parent(store, depth=3)
        tool = SpawnAgentTool(store, guard)
        result = await tool.execute(
            {"task": "Task", "child_id": "deep-child", "tool_call_id": "tc-1"},
            context,
        )
        state = await store.get_state("deep-child")
        assert state is not None
        assert state.depth == 4


class TestSleepAndWaitTool:
    @pytest.mark.asyncio
    async def test_sleep_waitset(self, store, guard, context):
        await _register_parent(store)
        spawn_tool = SpawnAgentTool(store, guard)
        await spawn_tool.execute(
            {"task": "A", "child_id": "child-1", "tool_call_id": "tc-1"}, context
        )
        await spawn_tool.execute(
            {"task": "B", "child_id": "child-2", "tool_call_id": "tc-2"}, context
        )

        sleep_tool = SleepAndWaitTool(store, guard)
        result = await sleep_tool.execute(
            {"wake_type": "waitset", "tool_call_id": "tc-3"},
            context,
        )

        assert result.termination_reason == TerminationReason.SLEEPING
        assert "waitset" in result.content

        state = await store.get_state("orch")
        assert state is not None
        assert state.status == AgentStateStatus.SLEEPING
        assert state.wake_condition is not None
        assert state.wake_condition.type == WakeType.WAITSET
        assert set(state.wake_condition.wait_for) == {"child-1", "child-2"}
        assert state.wake_condition.timeout_at is not None

    @pytest.mark.asyncio
    async def test_sleep_waitset_any_mode(self, store, guard, context):
        await _register_parent(store)
        spawn_tool = SpawnAgentTool(store, guard)
        await spawn_tool.execute(
            {"task": "A", "child_id": "child-1", "tool_call_id": "tc-1"}, context
        )
        await spawn_tool.execute(
            {"task": "B", "child_id": "child-2", "tool_call_id": "tc-2"}, context
        )

        sleep_tool = SleepAndWaitTool(store, guard)
        result = await sleep_tool.execute(
            {"wake_type": "waitset", "wait_mode": "any", "tool_call_id": "tc-3"},
            context,
        )

        state = await store.get_state("orch")
        assert state.wake_condition.wait_mode == WaitMode.ANY

    @pytest.mark.asyncio
    async def test_sleep_waitset_explicit_wait_for(self, store, guard, context):
        await _register_parent(store)
        spawn_tool = SpawnAgentTool(store, guard)
        await spawn_tool.execute(
            {"task": "A", "child_id": "child-1", "tool_call_id": "tc-1"}, context
        )
        await spawn_tool.execute(
            {"task": "B", "child_id": "child-2", "tool_call_id": "tc-2"}, context
        )

        sleep_tool = SleepAndWaitTool(store, guard)
        result = await sleep_tool.execute(
            {"wake_type": "waitset", "wait_for": ["child-1"], "tool_call_id": "tc-3"},
            context,
        )

        state = await store.get_state("orch")
        assert state.wake_condition.wait_for == ["child-1"]

    @pytest.mark.asyncio
    async def test_sleep_timer(self, store, guard, context):
        await _register_parent(store)

        sleep_tool = SleepAndWaitTool(store, guard)
        result = await sleep_tool.execute(
            {
                "wake_type": "timer",
                "delay_seconds": 30,
                "time_unit": "minutes",
                "tool_call_id": "tc-1",
            },
            context,
        )
        assert result.termination_reason == TerminationReason.SLEEPING

        state = await store.get_state("orch")
        assert state is not None
        assert state.wake_condition is not None
        assert state.wake_condition.type == WakeType.TIMER
        assert state.wake_condition.time_value == 30
        assert state.wake_condition.time_unit == TimeUnit.MINUTES
        assert state.wake_condition.wakeup_at is not None

    @pytest.mark.asyncio
    async def test_sleep_timer_requires_seconds(self, store, guard, context):
        await _register_parent(store)

        sleep_tool = SleepAndWaitTool(store, guard)
        result = await sleep_tool.execute(
            {"wake_type": "timer", "tool_call_id": "tc-1"},
            context,
        )
        assert not result.is_success
        assert "delay_seconds is required" in result.content

    @pytest.mark.asyncio
    async def test_sleep_invalid_wake_type(self, store, guard, context):
        sleep_tool = SleepAndWaitTool(store, guard)
        result = await sleep_tool.execute(
            {"wake_type": "invalid", "tool_call_id": "tc-1"},
            context,
        )
        assert not result.is_success

    @pytest.mark.asyncio
    async def test_sleep_waitset_with_custom_timeout(self, store, guard, context):
        await _register_parent(store)
        spawn_tool = SpawnAgentTool(store, guard)
        await spawn_tool.execute(
            {"task": "A", "child_id": "child-1", "tool_call_id": "tc-1"}, context
        )

        sleep_tool = SleepAndWaitTool(store, guard)
        result = await sleep_tool.execute(
            {"wake_type": "waitset", "timeout": 120, "tool_call_id": "tc-2"},
            context,
        )

        state = await store.get_state("orch")
        assert state.wake_condition.timeout_at is not None
        now = datetime.now(timezone.utc)
        diff = (state.wake_condition.timeout_at - now).total_seconds()
        assert 100 < diff < 130


class TestQuerySpawnedAgentTool:
    @pytest.mark.asyncio
    async def test_query_existing_agent(self, store, context):
        state = AgentState(
            id="child-1",
            session_id="sess-1",
            status=AgentStateStatus.COMPLETED,
            task="Do research",
            parent_id="orch",
            result_summary="Research results: ...",
        )
        await store.save_state(state)

        tool = QuerySpawnedAgentTool(store)
        result = await tool.execute(
            {"agent_id": "child-1", "tool_call_id": "tc-1"},
            context,
        )
        assert result.is_success
        assert "child-1" in result.content
        assert "completed" in result.content.lower()
        assert "Research results" in result.content

    @pytest.mark.asyncio
    async def test_query_nonexistent_agent(self, store, context):
        tool = QuerySpawnedAgentTool(store)
        result = await tool.execute(
            {"agent_id": "nope", "tool_call_id": "tc-1"},
            context,
        )
        assert not result.is_success
        assert "not found" in result.content
