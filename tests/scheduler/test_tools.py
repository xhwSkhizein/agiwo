"""Tests for scheduling tools."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agiwo.agent import RunFinished, RunStarted, TerminationReason
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.scheduler.engine import Scheduler
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
from agiwo.scheduler.runtime_tools import (
    CancelAgentTool,
    ForkChildAgentTool,
    ListAgentsTool,
    QuerySpawnedAgentTool,
    SleepAndWaitTool,
    SpawnChildAgentTool,
)
from agiwo.scheduler.store.memory import InMemoryAgentStateStorage
from agiwo.tool.builtin.bash_tool.process_tool import (
    BashProcessTool,
    BashProcessToolConfig,
)
from agiwo.tool.builtin.bash_tool.types import ProcessInfo
from tests.utils.agent_context import build_tool_context


@pytest.fixture
def store():
    return InMemoryAgentStateStorage()


@pytest.fixture
def guard(store):
    return TaskGuard(TaskLimits(), store)


@pytest.fixture
def engine(store, guard):
    return Scheduler(
        config=SchedulerConfig(),
        store=store,
        guard=guard,
        semaphore=asyncio.Semaphore(10),
    )


@pytest.fixture
def control(engine):
    return engine._tool_control


@pytest.fixture
def context():
    return build_tool_context(
        session_id="sess-1",
        run_id="run-1",
        agent_id="orch",
        agent_name="orchestrator",
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


class TestSpawnChildAgentTool:
    @pytest.mark.asyncio
    async def test_spawn_creates_pending_state(self, store, control, context):
        await _register_parent(store)
        tool = SpawnChildAgentTool(control)
        tool_result = await tool.execute(
            {"task": "Research topic A", "tool_call_id": "tc-1"},
            context,
        )
        assert tool_result.termination_reason is None
        assert tool_result.is_success
        assert "Spawned child agent" in tool_result.content

        child_id = tool_result.output["child_id"]
        state = await store.get_state(child_id)
        assert state is not None
        assert state.status == AgentStateStatus.PENDING
        assert state.parent_id == "orch"
        assert state.task == "Research topic A"
        assert state.session_id == "sess-1"
        assert state.depth == 1

    @pytest.mark.asyncio
    async def test_spawn_persists_allowed_tools_override(self, store, control, context):
        await _register_parent(store)
        tool = SpawnChildAgentTool(control)
        tool_result = await tool.execute(
            {
                "task": "Task",
                "allowed_tools": ["bash", "web_search"],
                "tool_call_id": "tc-1",
            },
            context,
        )

        child_id = tool_result.output["child_id"]
        state = await store.get_state(child_id)
        assert state is not None
        assert state.config_overrides["allowed_tools"] == ("bash", "web_search")

    @pytest.mark.asyncio
    async def test_spawn_fails_without_agent_id(self, store, control):
        ctx = build_tool_context(
            session_id="sess-1",
            run_id="run-1",
            agent_id="",
            agent_name="",
        )
        tool = SpawnChildAgentTool(control)
        tool_result = await tool.execute({"task": "Task", "tool_call_id": "tc-1"}, ctx)
        assert not tool_result.is_success

    @pytest.mark.asyncio
    async def test_spawn_generates_unique_ids(self, store, control, context):
        await _register_parent(store)
        tool = SpawnChildAgentTool(control)
        r1 = await tool.execute({"task": "A", "tool_call_id": "tc-1"}, context)
        r2 = await tool.execute({"task": "B", "tool_call_id": "tc-2"}, context)
        assert r1.output["child_id"] != r2.output["child_id"]

    @pytest.mark.asyncio
    async def test_spawn_rejected_max_depth(self, store, context):
        limits = TaskLimits(max_depth=2)
        guard = TaskGuard(limits, store)
        eng = Scheduler(
            config=SchedulerConfig(),
            store=store,
            guard=guard,
            semaphore=asyncio.Semaphore(10),
        )
        await _register_parent(store, depth=2)
        tool = SpawnChildAgentTool(eng._tool_control)
        tool_result = await tool.execute(
            {"task": "Deep task", "tool_call_id": "tc-1"},
            context,
        )
        assert not tool_result.is_success
        assert "Spawn rejected" in tool_result.content

    @pytest.mark.asyncio
    async def test_spawn_rejected_max_children(self, store, context):
        limits = TaskLimits(max_children_per_agent=2)
        guard = TaskGuard(limits, store)
        eng = Scheduler(
            config=SchedulerConfig(),
            store=store,
            guard=guard,
            semaphore=asyncio.Semaphore(10),
        )
        await _register_parent(store)
        tool = SpawnChildAgentTool(eng._tool_control)
        await tool.execute({"task": "A", "tool_call_id": "tc-1"}, context)
        await tool.execute({"task": "B", "tool_call_id": "tc-2"}, context)
        tool_result = await tool.execute({"task": "C", "tool_call_id": "tc-3"}, context)
        assert not tool_result.is_success
        assert "Spawn rejected" in tool_result.content

    @pytest.mark.asyncio
    async def test_spawn_inherits_depth(self, store, control, context):
        await _register_parent(store, depth=3)
        tool = SpawnChildAgentTool(control)
        result = await tool.execute({"task": "Task", "tool_call_id": "tc-1"}, context)
        state = await store.get_state(result.output["child_id"])
        assert state is not None
        assert state.depth == 4

    @pytest.mark.asyncio
    async def test_spawn_rejects_non_list_allowed_skills(self, store, control, context):
        await _register_parent(store)
        tool = SpawnChildAgentTool(control)

        tool_result = await tool.execute(
            {"task": "Task", "allowed_skills": "audit", "tool_call_id": "tc-1"},
            context,
        )

        assert not tool_result.is_success
        assert "allowed_skills must be a list of strings" in tool_result.content

    @pytest.mark.asyncio
    async def test_spawn_rejects_non_list_allowed_tools(self, store, control, context):
        await _register_parent(store)
        tool = SpawnChildAgentTool(control)

        tool_result = await tool.execute(
            {"task": "Task", "allowed_tools": "bash", "tool_call_id": "tc-1"},
            context,
        )

        assert not tool_result.is_success
        assert "allowed_tools must be a list of strings" in tool_result.content

    @pytest.mark.asyncio
    async def test_spawn_rejects_unknown_exact_allowed_skill(
        self,
        store,
        control,
        context,
    ):
        await _register_parent(store)
        tool = SpawnChildAgentTool(control)

        tool_result = await tool.execute(
            {
                "task": "Task",
                "allowed_skills": ["definitely-not-a-real-skill"],
                "tool_call_id": "tc-1",
            },
            context,
        )

        assert not tool_result.is_success
        assert (
            "Unknown allowed skill(s): definitely-not-a-real-skill"
            in tool_result.content
        )

    @pytest.mark.asyncio
    async def test_spawn_rejects_wildcard_allowed_skills(self, store, control, context):
        await _register_parent(store)
        tool = SpawnChildAgentTool(control)

        tool_result = await tool.execute(
            {"task": "Task", "allowed_skills": ["skill*"], "tool_call_id": "tc-1"},
            context,
        )

        assert not tool_result.is_success
        assert "explicit skill names" in tool_result.content

    @pytest.mark.asyncio
    async def test_gate_denies_spawn_from_child_agent(self, store, control):
        child_context = build_tool_context(
            session_id="sess-1",
            run_id="run-1",
            agent_id="child-1",
            agent_name="child",
            depth=1,
        )
        tool = SpawnChildAgentTool(control)
        decision = await tool.gate({"task": "Sub-task"}, child_context)
        assert decision.action == "deny"
        assert "cannot spawn" in decision.reason.lower()


class TestForkChildAgentTool:
    @pytest.mark.asyncio
    async def test_fork_creates_state_with_fork_flag(self, store, control, context):
        await _register_parent(store)
        tool = ForkChildAgentTool(control)
        tool_result = await tool.execute(
            {"task": "Continue analysis", "tool_call_id": "tc-1"},
            context,
        )

        assert tool_result.is_success
        state = await store.get_state(tool_result.output["child_id"])
        assert state is not None
        assert state.config_overrides.get("fork") is True
        assert state.config_overrides.get("instruction") is None
        assert state.config_overrides.get("allowed_skills") is None
        assert state.config_overrides.get("allowed_tools") is None

    @pytest.mark.asyncio
    async def test_gate_denies_fork_from_child_agent(self, store, control):
        child_context = build_tool_context(
            session_id="sess-1",
            run_id="run-1",
            agent_id="child-1",
            agent_name="child",
            depth=1,
        )
        tool = ForkChildAgentTool(control)
        decision = await tool.gate({"task": "Sub-task"}, child_context)
        assert decision.action == "deny"
        assert "cannot spawn" in decision.reason.lower()


class TestSleepAndWaitTool:
    @pytest.mark.asyncio
    async def test_sleep_waitset(self, store, control, context):
        await _register_parent(store)
        spawn_tool = SpawnChildAgentTool(control)
        child_1 = await spawn_tool.execute(
            {"task": "A", "tool_call_id": "tc-1"}, context
        )
        child_2 = await spawn_tool.execute(
            {"task": "B", "tool_call_id": "tc-2"}, context
        )

        sleep_tool = SleepAndWaitTool(control)
        tool_result = await sleep_tool.execute(
            {"wake_type": "waitset", "tool_call_id": "tc-3"},
            context,
        )

        assert tool_result.termination_reason == TerminationReason.SLEEPING
        assert "waitset" in tool_result.content

        state = await store.get_state("orch")
        assert state is not None
        assert state.status == AgentStateStatus.WAITING
        assert state.wake_condition is not None
        assert state.wake_condition.type == WakeType.WAITSET
        assert set(state.wake_condition.wait_for) == {
            child_1.output["child_id"],
            child_2.output["child_id"],
        }
        assert state.wake_condition.timeout_at is not None

    @pytest.mark.asyncio
    async def test_sleep_waitset_any_mode(self, store, control, context):
        await _register_parent(store)
        spawn_tool = SpawnChildAgentTool(control)
        await spawn_tool.execute({"task": "A", "tool_call_id": "tc-1"}, context)
        await spawn_tool.execute({"task": "B", "tool_call_id": "tc-2"}, context)

        sleep_tool = SleepAndWaitTool(control)
        await sleep_tool.execute(
            {"wake_type": "waitset", "wait_mode": "any", "tool_call_id": "tc-3"},
            context,
        )

        state = await store.get_state("orch")
        assert state.wake_condition.wait_mode == WaitMode.ANY

    @pytest.mark.asyncio
    async def test_sleep_waitset_explicit_wait_for(self, store, control, context):
        await _register_parent(store)
        spawn_tool = SpawnChildAgentTool(control)
        child = await spawn_tool.execute({"task": "A", "tool_call_id": "tc-1"}, context)
        await spawn_tool.execute({"task": "B", "tool_call_id": "tc-2"}, context)

        sleep_tool = SleepAndWaitTool(control)
        await sleep_tool.execute(
            {
                "wake_type": "waitset",
                "wait_for": [child.output["child_id"]],
                "tool_call_id": "tc-3",
            },
            context,
        )

        state = await store.get_state("orch")
        assert state.wake_condition.wait_for == (child.output["child_id"],)

    @pytest.mark.asyncio
    async def test_sleep_timer(self, store, control, context):
        await _register_parent(store)

        sleep_tool = SleepAndWaitTool(control)
        tool_result = await sleep_tool.execute(
            {
                "wake_type": "timer",
                "delay_seconds": 30,
                "time_unit": "minutes",
                "tool_call_id": "tc-1",
            },
            context,
        )
        assert tool_result.termination_reason == TerminationReason.SLEEPING

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
        tool_result = await sleep_tool.execute(
            {"wake_type": "timer", "tool_call_id": "tc-1"},
            context,
        )
        assert not tool_result.is_success
        assert "delay_seconds is required" in tool_result.content

    @pytest.mark.asyncio
    async def test_sleep_invalid_wake_type(self, store, control, context):
        sleep_tool = SleepAndWaitTool(control)
        tool_result = await sleep_tool.execute(
            {"wake_type": "invalid", "tool_call_id": "tc-1"},
            context,
        )
        assert not tool_result.is_success

    @pytest.mark.asyncio
    async def test_sleep_waitset_with_custom_timeout(self, store, control, context):
        await _register_parent(store)
        spawn_tool = SpawnChildAgentTool(control)
        await spawn_tool.execute({"task": "A", "tool_call_id": "tc-1"}, context)

        sleep_tool = SleepAndWaitTool(control)
        await sleep_tool.execute(
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
        tool_result = await tool.execute(
            {"agent_id": "child-1", "tool_call_id": "tc-1"},
            context,
        )
        assert tool_result.is_success
        assert "child-1" in tool_result.content
        assert "completed" in tool_result.content.lower()
        assert "Research results" in tool_result.content

    @pytest.mark.asyncio
    async def test_query_nonexistent_agent(self, store, control, context):
        tool = QuerySpawnedAgentTool(control)
        tool_result = await tool.execute(
            {"agent_id": "nope", "tool_call_id": "tc-1"},
            context,
        )
        assert not tool_result.is_success
        assert "not found" in tool_result.content

    @pytest.mark.asyncio
    async def test_query_includes_explain(self, store, control, context):
        state = AgentState(
            id="child-2",
            session_id="sess-1",
            status=AgentStateStatus.WAITING,
            task="Periodic check",
            parent_id="orch",
            explain="Waiting 8h before next run",
        )
        await store.save_state(state)

        tool = QuerySpawnedAgentTool(control)
        tool_result = await tool.execute(
            {"agent_id": "child-2", "tool_call_id": "tc-1"}, context
        )
        assert tool_result.is_success
        assert "Waiting 8h" in tool_result.content

    @pytest.mark.asyncio
    async def test_query_prefers_runtime_run_view_summary(
        self, store, control, context
    ):
        state = AgentState(
            id="child-3",
            session_id="sess-1",
            status=AgentStateStatus.COMPLETED,
            task="Do research",
            parent_id="orch",
            result_summary="stale summary",
        )
        await store.save_state(state)
        storage = InMemoryRunLogStorage()
        await storage.append_entries(
            [
                RunStarted(
                    sequence=1,
                    session_id="child-3",
                    run_id="run-1",
                    agent_id="child-3",
                    user_input="Do research",
                ),
                RunFinished(
                    sequence=2,
                    session_id="child-3",
                    run_id="run-1",
                    agent_id="child-3",
                    response="fresh runtime summary",
                    termination_reason=TerminationReason.COMPLETED,
                ),
            ]
        )
        control._rt.agents["child-3"] = type(
            "RuntimeAgentStub",
            (),
            {"run_log_storage": storage},
        )()

        tool = QuerySpawnedAgentTool(control)
        tool_result = await tool.execute(
            {"agent_id": "child-3", "tool_call_id": "tc-1"},
            context,
        )

        assert tool_result.is_success
        assert "fresh runtime summary" in tool_result.content


class TestSleepAndWaitExplain:
    @pytest.mark.asyncio
    async def test_sleep_with_explain_stored(self, store, control, context):
        await _register_parent(store)
        sleep_tool = SleepAndWaitTool(control)
        tool_result = await sleep_tool.execute(
            {
                "wake_type": "timer",
                "delay_seconds": 10,
                "explain": "Waiting for rate limit to reset",
                "tool_call_id": "tc-1",
            },
            context,
        )
        assert tool_result.is_success
        state = await store.get_state("orch")
        assert state.explain == "Waiting for rate limit to reset"
        assert "Waiting for rate limit to reset" in tool_result.content


class TestCancelAgentTool:
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_agent(self, control, context):
        tool = CancelAgentTool(control)
        tool_result = await tool.execute(
            {"agent_id": "ghost", "tool_call_id": "tc-1"}, context
        )
        assert not tool_result.is_success
        assert "not found" in tool_result.content

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
        tool_result = await tool.execute(
            {"agent_id": "other", "tool_call_id": "tc-1"}, context
        )
        assert not tool_result.is_success
        assert "Permission denied" in tool_result.content

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
        tool_result = await tool.execute(
            {"agent_id": "child-run", "force": False, "tool_call_id": "tc-1"}, context
        )
        assert tool_result.output.get("requires_force") is True

    @pytest.mark.asyncio
    async def test_cancel_running_without_force_includes_running_bash_processes(
        self, store, control, context
    ):
        class MockBashSandbox:
            async def list_processes_by_agent(
                self, agent_id: str, state: str = "running"
            ):
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
        control._rt.agents["child-run"] = mock_agent

        tool = CancelAgentTool(control)
        tool_result = await tool.execute(
            {"agent_id": "child-run", "force": False, "tool_call_id": "tc-1"},
            context,
        )

        assert tool_result.output.get("requires_force") is True
        assert tool_result.output["running_processes"][0]["process_id"] == "proc-1"
        assert "sleep 1" in tool_result.content

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
        tool = CancelAgentTool(control)
        tool_result = await tool.execute(
            {"agent_id": "child-force", "force": True, "tool_call_id": "tc-1"}, context
        )
        assert tool_result.is_success
        updated = await store.get_state("child-force")
        assert updated is not None
        assert updated.status == AgentStateStatus.FAILED
        assert updated.result_summary == "Cancelled by parent agent"

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
        tool_result = await tool.execute(
            {"agent_id": "child-done", "tool_call_id": "tc-1"}, context
        )
        assert "terminal state" in tool_result.content


class TestListAgentsTool:
    @pytest.mark.asyncio
    async def test_list_no_children(self, control, context):
        tool = ListAgentsTool(control)
        tool_result = await tool.execute({"tool_call_id": "tc-1"}, context)
        assert tool_result.is_success
        assert "No child agents" in tool_result.content

    @pytest.mark.asyncio
    async def test_list_children(self, store, control, context):
        c1 = AgentState(
            id="c1",
            session_id="sess-1",
            status=AgentStateStatus.RUNNING,
            task="task A",
            parent_id="orch",
        )
        c2 = AgentState(
            id="c2",
            session_id="sess-1",
            status=AgentStateStatus.WAITING,
            task="task B",
            parent_id="orch",
            explain="Sleeping until morning",
            result_summary="partial result",
        )
        await store.save_state(c1)
        await store.save_state(c2)

        tool = ListAgentsTool(control)
        tool_result = await tool.execute({"tool_call_id": "tc-1"}, context)
        assert tool_result.is_success
        assert "c1" in tool_result.content
        assert "c2" in tool_result.content
        assert "Sleeping until morning" in tool_result.content
        assert "partial result" in tool_result.content
        agents = tool_result.output["agents"]
        assert len(agents) == 2


class TestSleepAndWaitNoProgress:
    @pytest.mark.asyncio
    async def test_periodic_no_progress_propagated(self, store, control, context):
        await _register_parent(store)
        sleep_tool = SleepAndWaitTool(control)
        tool_result = await sleep_tool.execute(
            {
                "wake_type": "periodic",
                "delay_seconds": 60,
                "no_progress": True,
                "tool_call_id": "tc-1",
            },
            context,
        )
        assert tool_result.termination_reason == TerminationReason.SLEEPING
        state = await store.get_state("orch")
        assert state.no_progress is True

    @pytest.mark.asyncio
    async def test_timer_ignores_no_progress(self, store, control, context):
        await _register_parent(store)
        sleep_tool = SleepAndWaitTool(control)
        await sleep_tool.execute(
            {
                "wake_type": "timer",
                "delay_seconds": 30,
                "no_progress": True,
                "tool_call_id": "tc-1",
            },
            context,
        )
        state = await store.get_state("orch")
        assert state.no_progress is False

    @pytest.mark.asyncio
    async def test_no_progress_default_false(self, store, control, context):
        await _register_parent(store)
        sleep_tool = SleepAndWaitTool(control)
        await sleep_tool.execute(
            {
                "wake_type": "periodic",
                "delay_seconds": 60,
                "tool_call_id": "tc-1",
            },
            context,
        )
        state = await store.get_state("orch")
        assert state.no_progress is False
