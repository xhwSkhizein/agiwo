"""Tests for scheduling tools."""

import pytest
from datetime import datetime, timedelta, timezone

from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.schema import TerminationReason
from agiwo.agent.stream_channel import StreamChannel
from agiwo.scheduler.models import AgentState, AgentStateStatus, TimeUnit, WakeCondition, WakeType
from agiwo.scheduler.store import InMemoryAgentStateStorage
from agiwo.scheduler.tools import SpawnAgentTool, SleepAndWaitTool, QuerySpawnedAgentTool


@pytest.fixture
def store():
    return InMemoryAgentStateStorage()


@pytest.fixture
def context():
    return ExecutionContext(
        session_id="sess-1",
        run_id="run-1",
        channel=StreamChannel(),
        agent_id="orch",
        sequence_counter=SessionSequenceCounter(0),
    )


class TestSpawnAgentTool:
    @pytest.mark.asyncio
    async def test_spawn_creates_pending_state(self, store, context):
        tool = SpawnAgentTool(store)
        result = await tool.execute(
            {"task": "Research topic A", "tool_call_id": "tc-1"},
            context,
        )
        assert result.is_success
        assert "Spawned child agent" in result.content

        # Verify state was created
        child_id = result.output["child_id"]
        state = await store.get_state(child_id)
        assert state is not None
        assert state.status == AgentStateStatus.PENDING
        assert state.parent_agent_id == "orch"
        assert state.parent_state_id == "orch"
        assert state.task == "Research topic A"
        assert state.session_id == "sess-1"

    @pytest.mark.asyncio
    async def test_spawn_with_custom_child_id(self, store, context):
        tool = SpawnAgentTool(store)
        result = await tool.execute(
            {"task": "Task", "child_id": "my-child", "tool_call_id": "tc-1"},
            context,
        )
        assert result.output["child_id"] == "my-child"
        state = await store.get_state("my-child")
        assert state is not None

    @pytest.mark.asyncio
    async def test_spawn_with_system_prompt_override(self, store, context):
        tool = SpawnAgentTool(store)
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
    async def test_spawn_fails_without_agent_id(self, store):
        ctx = ExecutionContext(
            session_id="sess-1",
            run_id="run-1",
            channel=StreamChannel(),
            agent_id=None,
            sequence_counter=SessionSequenceCounter(0),
        )
        tool = SpawnAgentTool(store)
        result = await tool.execute({"task": "Task", "tool_call_id": "tc-1"}, ctx)
        assert not result.is_success

    @pytest.mark.asyncio
    async def test_spawn_generates_unique_ids(self, store, context):
        tool = SpawnAgentTool(store)
        r1 = await tool.execute({"task": "A", "tool_call_id": "tc-1"}, context)
        r2 = await tool.execute({"task": "B", "tool_call_id": "tc-2"}, context)
        assert r1.output["child_id"] != r2.output["child_id"]


class TestSleepAndWaitTool:
    @pytest.mark.asyncio
    async def test_sleep_children_complete(self, store, context):
        # Pre-register the parent state (as Scheduler.submit would)
        parent_state = AgentState(
            id="orch",
            session_id="sess-1",
            agent_id="orch",
            parent_agent_id="orch",
            parent_state_id=None,
            status=AgentStateStatus.RUNNING,
            task="root task",
        )
        await store.save_state(parent_state)

        # Spawn 2 children first
        spawn_tool = SpawnAgentTool(store)
        await spawn_tool.execute(
            {"task": "A", "child_id": "child-1", "tool_call_id": "tc-1"}, context
        )
        await spawn_tool.execute(
            {"task": "B", "child_id": "child-2", "tool_call_id": "tc-2"}, context
        )

        # Now sleep
        sleep_tool = SleepAndWaitTool(store)
        result = await sleep_tool.execute(
            {"wake_type": "children_complete", "tool_call_id": "tc-3"},
            context,
        )

        assert result.termination_reason == TerminationReason.SLEEPING
        assert "children_complete" in result.content

        # Verify parent state is SLEEPING with wake condition
        state = await store.get_state("orch")
        assert state is not None
        assert state.status == AgentStateStatus.SLEEPING
        assert state.wake_condition is not None
        assert state.wake_condition.type == WakeType.CHILDREN_COMPLETE
        assert state.wake_condition.total_children == 2

    @pytest.mark.asyncio
    async def test_sleep_delay(self, store, context):
        parent_state = AgentState(
            id="orch",
            session_id="sess-1",
            agent_id="orch",
            parent_agent_id="orch",
            parent_state_id=None,
            status=AgentStateStatus.RUNNING,
            task="root task",
        )
        await store.save_state(parent_state)

        sleep_tool = SleepAndWaitTool(store)
        result = await sleep_tool.execute(
            {
                "wake_type": "delay",
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
        assert state.wake_condition.type == WakeType.DELAY
        assert state.wake_condition.time_value == 30
        assert state.wake_condition.time_unit == TimeUnit.MINUTES
        assert state.wake_condition.wakeup_at is not None

    @pytest.mark.asyncio
    async def test_sleep_delay_requires_seconds(self, store, context):
        parent_state = AgentState(
            id="orch",
            session_id="sess-1",
            agent_id="orch",
            parent_agent_id="orch",
            parent_state_id=None,
            status=AgentStateStatus.RUNNING,
            task="root task",
        )
        await store.save_state(parent_state)

        sleep_tool = SleepAndWaitTool(store)
        result = await sleep_tool.execute(
            {"wake_type": "delay", "tool_call_id": "tc-1"},
            context,
        )
        assert not result.is_success
        assert "delay_seconds is required" in result.content

    @pytest.mark.asyncio
    async def test_sleep_invalid_wake_type(self, store, context):
        sleep_tool = SleepAndWaitTool(store)
        result = await sleep_tool.execute(
            {"wake_type": "invalid", "tool_call_id": "tc-1"},
            context,
        )
        assert not result.is_success


class TestQuerySpawnedAgentTool:
    @pytest.mark.asyncio
    async def test_query_existing_agent(self, store, context):
        state = AgentState(
            id="child-1",
            session_id="sess-1",
            agent_id="child-1",
            parent_agent_id="orch",
            parent_state_id="orch",
            status=AgentStateStatus.COMPLETED,
            task="Do research",
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
