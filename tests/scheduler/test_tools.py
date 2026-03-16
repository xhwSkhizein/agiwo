"""Tests for scheduling tools."""

import asyncio
import pytest
from datetime import datetime, timezone

from agiwo.agent import TerminationReason
from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.stream_channel import StreamChannel
from agiwo.scheduler.coordinator import SchedulerCoordinator
from agiwo.scheduler.engine import SchedulerEngine
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    SchedulerConfig,
    TaskLimits,
    TimeUnit,
    WaitMode,
    WakeType,
)
from agiwo.scheduler.runner import SchedulerRunner
from agiwo.scheduler.runtime_tools import (
    CancelAgentTool,
    ListAgentsTool,
    QuerySpawnedAgentTool,
    SleepAndWaitTool,
    SpawnAgentTool,
)
from agiwo.scheduler.store import InMemoryAgentStateStorage
from agiwo.tool.builtin.bash_tool.process_tool import (
    BashProcessTool,
    BashProcessToolConfig,
)
from agiwo.tool.builtin.bash_tool.types import ProcessInfo
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def store():
    return InMemoryAgentStateStorage()


@pytest.fixture
def guard(store):
    return TaskGuard(TaskLimits(), store)


@pytest.fixture
def coordinator():
    return SchedulerCoordinator()


@pytest.fixture
def runner(store, coordinator):
    return SchedulerRunner(store, coordinator, asyncio.Semaphore(10))


@pytest.fixture
def control(store, guard, coordinator, runner):
    return SchedulerEngine(
        config=SchedulerConfig(),
        store=store,
        guard=guard,
        coordinator=coordinator,
        runner=runner,
    )


@pytest.fixture
def context():
    return ExecutionContext(
        session_id="sess-1",
        run_id="run-1",
        channel=StreamChannel(),
        agent_id="orch",
        agent_name="orchestrator",
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
    async def test_spawn_creates_pending_state(self, store, control, context):
        await _register_parent(store)
        tool = SpawnAgentTool(control)
        result = await tool.execute_for_agent(
            {"task": "Research topic A", "tool_call_id": "tc-1"},
            context,
        )
        assert result.result.is_success
        assert "Spawned child agent" in result.result.content

        child_id = result.result.output["child_id"]
        state = await store.get_state(child_id)
        assert state is not None
        assert state.status == AgentStateStatus.PENDING
        assert state.parent_id == "orch"
        assert state.task == "Research topic A"
        assert state.session_id == "sess-1"
        assert state.depth == 1

    @pytest.mark.asyncio
    async def test_spawn_with_custom_child_id(self, store, control, context):
        await _register_parent(store)
        tool = SpawnAgentTool(control)
        result = await tool.execute_for_agent(
            {"task": "Task", "child_id": "my-child", "tool_call_id": "tc-1"},
            context,
        )
        assert result.result.output["child_id"] == "my-child"
        state = await store.get_state("my-child")
        assert state is not None

    @pytest.mark.asyncio
    async def test_spawn_with_system_prompt_override(self, store, control, context):
        await _register_parent(store)
        tool = SpawnAgentTool(control)
        result = await tool.execute_for_agent(
            {
                "task": "Task",
                "system_prompt": "You are a specialist.",
                "tool_call_id": "tc-1",
            },
            context,
        )
        child_id = result.result.output["child_id"]
        state = await store.get_state(child_id)
        assert state is not None
        assert state.config_overrides["system_prompt"] == "You are a specialist."

    @pytest.mark.asyncio
    async def test_spawn_fails_without_agent_id(self, store, control):
        ctx = ExecutionContext(
            session_id="sess-1",
            run_id="run-1",
            channel=StreamChannel(),
            agent_id="",
            agent_name="",
            sequence_counter=SessionSequenceCounter(0),
        )
        tool = SpawnAgentTool(control)
        result = await tool.execute_for_agent({"task": "Task", "tool_call_id": "tc-1"}, ctx)
        assert not result.result.is_success

    @pytest.mark.asyncio
    async def test_spawn_generates_unique_ids(self, store, control, context):
        await _register_parent(store)
        tool = SpawnAgentTool(control)
        r1 = await tool.execute_for_agent({"task": "A", "tool_call_id": "tc-1"}, context)
        r2 = await tool.execute_for_agent({"task": "B", "tool_call_id": "tc-2"}, context)
        assert r1.result.output["child_id"] != r2.result.output["child_id"]

    @pytest.mark.asyncio
    async def test_spawn_rejected_max_depth(self, store, coordinator, runner, context):
        limits = TaskLimits(max_depth=2)
        guard = TaskGuard(limits, store)
        control = SchedulerEngine(
            config=SchedulerConfig(),
            store=store,
            guard=guard,
            coordinator=coordinator,
            runner=runner,
        )
        await _register_parent(store, depth=2)
        tool = SpawnAgentTool(control)
        result = await tool.execute_for_agent(
            {"task": "Deep task", "tool_call_id": "tc-1"},
            context,
        )
        assert not result.result.is_success
        assert "Spawn rejected" in result.result.content

    @pytest.mark.asyncio
    async def test_spawn_rejected_max_children(self, store, coordinator, runner, context):
        limits = TaskLimits(max_children_per_agent=2)
        guard = TaskGuard(limits, store)
        control = SchedulerEngine(
            config=SchedulerConfig(),
            store=store,
            guard=guard,
            coordinator=coordinator,
            runner=runner,
        )
        await _register_parent(store)
        tool = SpawnAgentTool(control)
        await tool.execute_for_agent({"task": "A", "child_id": "c1", "tool_call_id": "tc-1"}, context)
        await tool.execute_for_agent({"task": "B", "child_id": "c2", "tool_call_id": "tc-2"}, context)
        result = await tool.execute_for_agent({"task": "C", "child_id": "c3", "tool_call_id": "tc-3"}, context)
        assert not result.result.is_success
        assert "Spawn rejected" in result.result.content

    @pytest.mark.asyncio
    async def test_spawn_inherits_depth(self, store, control, context):
        await _register_parent(store, depth=3)
        tool = SpawnAgentTool(control)
        await tool.execute_for_agent(
            {"task": "Task", "child_id": "deep-child", "tool_call_id": "tc-1"},
            context,
        )
        state = await store.get_state("deep-child")
        assert state is not None
        assert state.depth == 4


class TestSleepAndWaitTool:
    @pytest.mark.asyncio
    async def test_sleep_waitset(self, store, control, context):
        await _register_parent(store)
        spawn_tool = SpawnAgentTool(control)
        await spawn_tool.execute_for_agent(
            {"task": "A", "child_id": "child-1", "tool_call_id": "tc-1"}, context
        )
        await spawn_tool.execute_for_agent(
            {"task": "B", "child_id": "child-2", "tool_call_id": "tc-2"}, context
        )

        sleep_tool = SleepAndWaitTool(control)
        result = await sleep_tool.execute_for_agent(
            {"wake_type": "waitset", "tool_call_id": "tc-3"},
            context,
        )

        assert result.termination_reason == TerminationReason.SLEEPING
        assert "waitset" in result.result.content

        state = await store.get_state("orch")
        assert state is not None
        assert state.status == AgentStateStatus.SLEEPING
        assert state.wake_condition is not None
        assert state.wake_condition.type == WakeType.WAITSET
        assert set(state.wake_condition.wait_for) == {"child-1", "child-2"}
        assert state.wake_condition.timeout_at is not None

    @pytest.mark.asyncio
    async def test_sleep_waitset_any_mode(self, store, control, context):
        await _register_parent(store)
        spawn_tool = SpawnAgentTool(control)
        await spawn_tool.execute_for_agent(
            {"task": "A", "child_id": "child-1", "tool_call_id": "tc-1"}, context
        )
        await spawn_tool.execute_for_agent(
            {"task": "B", "child_id": "child-2", "tool_call_id": "tc-2"}, context
        )

        sleep_tool = SleepAndWaitTool(control)
        await sleep_tool.execute_for_agent(
            {"wake_type": "waitset", "wait_mode": "any", "tool_call_id": "tc-3"},
            context,
        )

        state = await store.get_state("orch")
        assert state.wake_condition.wait_mode == WaitMode.ANY

    @pytest.mark.asyncio
    async def test_sleep_waitset_explicit_wait_for(self, store, control, context):
        await _register_parent(store)
        spawn_tool = SpawnAgentTool(control)
        await spawn_tool.execute_for_agent(
            {"task": "A", "child_id": "child-1", "tool_call_id": "tc-1"}, context
        )
        await spawn_tool.execute_for_agent(
            {"task": "B", "child_id": "child-2", "tool_call_id": "tc-2"}, context
        )

        sleep_tool = SleepAndWaitTool(control)
        await sleep_tool.execute_for_agent(
            {"wake_type": "waitset", "wait_for": ["child-1"], "tool_call_id": "tc-3"},
            context,
        )

        state = await store.get_state("orch")
        assert state.wake_condition.wait_for == ["child-1"]

    @pytest.mark.asyncio
    async def test_sleep_timer(self, store, control, context):
        await _register_parent(store)

        sleep_tool = SleepAndWaitTool(control)
        result = await sleep_tool.execute_for_agent(
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
    async def test_sleep_timer_requires_seconds(self, store, control, context):
        await _register_parent(store)

        sleep_tool = SleepAndWaitTool(control)
        result = await sleep_tool.execute_for_agent(
            {"wake_type": "timer", "tool_call_id": "tc-1"},
            context,
        )
        assert not result.result.is_success
        assert "delay_seconds is required" in result.result.content

    @pytest.mark.asyncio
    async def test_sleep_invalid_wake_type(self, store, control, context):
        sleep_tool = SleepAndWaitTool(control)
        result = await sleep_tool.execute_for_agent(
            {"wake_type": "invalid", "tool_call_id": "tc-1"},
            context,
        )
        assert not result.result.is_success

    @pytest.mark.asyncio
    async def test_sleep_waitset_with_custom_timeout(self, store, control, context):
        await _register_parent(store)
        spawn_tool = SpawnAgentTool(control)
        await spawn_tool.execute_for_agent(
            {"task": "A", "child_id": "child-1", "tool_call_id": "tc-1"}, context
        )

        sleep_tool = SleepAndWaitTool(control)
        await sleep_tool.execute_for_agent(
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
    async def test_query_existing_agent(self, store, control, context):
        state = AgentState(
            id="child-1",
            session_id="sess-1",
            status=AgentStateStatus.COMPLETED,
            task="Do research",
            parent_id="orch",
            result_summary="Research results: ...",
        )
        await store.save_state(state)

        tool = QuerySpawnedAgentTool(control)
        result = await tool.execute_for_agent(
            {"agent_id": "child-1", "tool_call_id": "tc-1"},
            context,
        )
        assert result.result.is_success
        assert "child-1" in result.result.content
        assert "completed" in result.result.content.lower()
        assert "Research results" in result.result.content

    @pytest.mark.asyncio
    async def test_query_nonexistent_agent(self, store, control, context):
        tool = QuerySpawnedAgentTool(control)
        result = await tool.execute_for_agent(
            {"agent_id": "nope", "tool_call_id": "tc-1"},
            context,
        )
        assert not result.result.is_success
        assert "not found" in result.result.content

    @pytest.mark.asyncio
    async def test_query_includes_explain_and_recent_steps(self, store, control, context):
        state = AgentState(
            id="child-2",
            session_id="sess-1",
            status=AgentStateStatus.SLEEPING,
            task="Periodic check",
            parent_id="orch",
            explain="Waiting 8h before next run",
            recent_steps=[{"role": "assistant", "timestamp": "2026-01-01T00:00:00", "tool_calls": ["web_search"]}],
        )
        await store.save_state(state)

        tool = QuerySpawnedAgentTool(control)
        result = await tool.execute_for_agent({"agent_id": "child-2", "tool_call_id": "tc-1"}, context)
        assert result.result.is_success
        assert "Waiting 8h" in result.result.content
        assert "web_search" in result.result.content


class TestSleepAndWaitExplain:
    @pytest.mark.asyncio
    async def test_sleep_with_explain_stored(self, store, control, context):
        await _register_parent(store)
        sleep_tool = SleepAndWaitTool(control)
        result = await sleep_tool.execute_for_agent(
            {
                "wake_type": "timer",
                "delay_seconds": 10,
                "explain": "Waiting for rate limit to reset",
                "tool_call_id": "tc-1",
            },
            context,
        )
        assert result.result.is_success
        state = await store.get_state("orch")
        assert state.explain == "Waiting for rate limit to reset"
        assert "Waiting for rate limit to reset" in result.result.content


class TestCancelAgentTool:
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_agent(self, control, context):
        tool = CancelAgentTool(control)
        result = await tool.execute_for_agent({"agent_id": "ghost", "tool_call_id": "tc-1"}, context)
        assert not result.result.is_success
        assert "not found" in result.result.content

    @pytest.mark.asyncio
    async def test_cancel_non_child_agent(self, store, control, context):
        other_state = AgentState(
            id="other",
            session_id="sess-1",
            status=AgentStateStatus.RUNNING,
            task="other task",
            parent_id="someone-else",
        )
        await store.save_state(other_state)
        tool = CancelAgentTool(control)
        result = await tool.execute_for_agent({"agent_id": "other", "tool_call_id": "tc-1"}, context)
        assert not result.result.is_success
        assert "Permission denied" in result.result.content

    @pytest.mark.asyncio
    async def test_cancel_running_without_force_warns(self, store, control, context):
        child = AgentState(
            id="child-run",
            session_id="sess-1",
            status=AgentStateStatus.RUNNING,
            task="run",
            parent_id="orch",
        )
        await store.save_state(child)
        tool = CancelAgentTool(control)
        result = await tool.execute_for_agent({"agent_id": "child-run", "force": False, "tool_call_id": "tc-1"}, context)
        # Should warn and not cancel
        assert result.result.output.get("requires_force") is True

    @pytest.mark.asyncio
    async def test_cancel_running_without_force_includes_running_bash_processes(self, store, coordinator, control, context):
        class MockBashSandbox:
            async def list_processes_by_agent(self, agent_id: str, state: str = "running"):
                if agent_id != "child-run" or state != "running":
                    return []
                return [
                    ProcessInfo(
                        process_id="proc-1",
                        command="sleep 1",
                        state="running",
                        started_at=datetime.now(timezone.utc).timestamp(),
                        exit_code=None,
                    )
                ]

        child = AgentState(
            id="child-run",
            session_id="sess-1",
            status=AgentStateStatus.RUNNING,
            task="run",
            parent_id="orch",
        )
        await store.save_state(child)

        bash_process_tool = BashProcessTool(
            BashProcessToolConfig(
                sandbox=MockBashSandbox(),  # type: ignore[arg-type]
            )
        )
        mock_agent = MagicMock()
        mock_agent.tools = [bash_process_tool]
        mock_agent.id = "child-run"
        coordinator.register_agent(mock_agent)

        tool = CancelAgentTool(control)
        result = await tool.execute_for_agent(
            {"agent_id": "child-run", "force": False, "tool_call_id": "tc-1"},
            context,
        )

        assert result.result.output.get("requires_force") is True
        assert result.result.output["running_processes"][0]["process_id"] == "proc-1"
        assert "sleep 1" in result.result.content

    @pytest.mark.asyncio
    async def test_cancel_with_force(self, store, control, context):
        child = AgentState(
            id="child-force",
            session_id="sess-1",
            status=AgentStateStatus.RUNNING,
            task="run",
            parent_id="orch",
        )
        await store.save_state(child)
        control._recursive_cancel = AsyncMock()  # type: ignore[method-assign]
        tool = CancelAgentTool(control)
        result = await tool.execute_for_agent({"agent_id": "child-force", "force": True, "tool_call_id": "tc-1"}, context)
        assert result.result.is_success
        control._recursive_cancel.assert_awaited_once_with("child-force", "Cancelled by parent agent")

    @pytest.mark.asyncio
    async def test_cancel_already_completed(self, store, control, context):
        child = AgentState(
            id="child-done",
            session_id="sess-1",
            status=AgentStateStatus.COMPLETED,
            task="run",
            parent_id="orch",
        )
        await store.save_state(child)
        tool = CancelAgentTool(control)
        result = await tool.execute_for_agent({"agent_id": "child-done", "tool_call_id": "tc-1"}, context)
        assert "terminal state" in result.result.content


class TestListAgentsTool:
    @pytest.mark.asyncio
    async def test_list_no_children(self, control, context):
        tool = ListAgentsTool(control)
        result = await tool.execute_for_agent({"tool_call_id": "tc-1"}, context)
        assert result.result.is_success
        assert "No child agents" in result.result.content

    @pytest.mark.asyncio
    async def test_list_children(self, store, control, context):
        c1 = AgentState(
            id="c1", session_id="sess-1", status=AgentStateStatus.RUNNING, task="task A", parent_id="orch"
        )
        c2 = AgentState(
            id="c2", session_id="sess-1", status=AgentStateStatus.SLEEPING, task="task B", parent_id="orch",
            explain="Sleeping until morning", result_summary="partial result"
        )
        await store.save_state(c1)
        await store.save_state(c2)

        tool = ListAgentsTool(control)
        result = await tool.execute_for_agent({"tool_call_id": "tc-1"}, context)
        assert result.result.is_success
        assert "c1" in result.result.content
        assert "c2" in result.result.content
        assert "Sleeping until morning" in result.result.content
        assert "partial result" in result.result.content
        agents = result.result.output["agents"]
        assert len(agents) == 2
