"""Integration tests for the Scheduler class."""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agiwo.agent.agent import Agent
from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.schema import RunOutput, TerminationReason
from agiwo.agent.stream_channel import StreamChannel
from agiwo.llm.base import Model, StreamChunk
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    SchedulerConfig,
    TaskLimits,
    WaitMode,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.scheduler import Scheduler
from agiwo.scheduler.store import InMemoryAgentStateStorage
from agiwo.tool.base import BaseTool, ToolResult


def _fast_config(**kwargs) -> SchedulerConfig:
    """SchedulerConfig with short check_interval for tests."""
    return SchedulerConfig(check_interval=0.1, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Mock Model
# ═══════════════════════════════════════════════════════════════════════════


class MockModel(Model):
    """Mock model that returns configurable responses per call."""

    def __init__(self, responses: list[list[StreamChunk]] | None = None) -> None:
        super().__init__(id="mock", name="mock", temperature=0.7)
        self._responses = responses or []
        self._call_count = 0

    async def arun_stream(self, messages, tools=None):
        if self._call_count < len(self._responses):
            chunks = self._responses[self._call_count]
        else:
            chunks = [
                StreamChunk(content="Default response"),
                StreamChunk(finish_reason="stop"),
            ]
        self._call_count += 1
        for chunk in chunks:
            yield chunk


def _simple_completion(text: str = "Done") -> list[StreamChunk]:
    return [StreamChunk(content=text), StreamChunk(finish_reason="stop")]


def _tool_call(tool_name: str, args: str = "{}", tc_id: str = "tc-1") -> list[StreamChunk]:
    return [
        StreamChunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": tc_id,
                    "function": {"name": tool_name, "arguments": args},
                }
            ]
        ),
        StreamChunk(finish_reason="tool_calls"),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSchedulerLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        scheduler = Scheduler(_fast_config())
        await scheduler.start()
        assert scheduler._running is True
        await scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with Scheduler(_fast_config()) as scheduler:
            assert scheduler._running is True
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_default_store_is_in_memory(self):
        scheduler = Scheduler()
        assert isinstance(scheduler._store, InMemoryAgentStateStorage)
        await scheduler.stop()


class TestSchedulerPrepareAgent:
    @pytest.mark.asyncio
    async def test_prepare_injects_tools(self):
        scheduler = Scheduler()
        model = MockModel()
        agent = Agent(name="test", description="test", model=model, id="test", tools=[])

        scheduler._prepare_agent(agent)

        tool_names = {t.get_name() for t in agent.tools}
        assert "spawn_agent" in tool_names
        assert "sleep_and_wait" in tool_names
        assert "query_spawned_agent" in tool_names
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_prepare_is_idempotent(self):
        scheduler = Scheduler()
        model = MockModel()
        agent = Agent(name="test", description="test", model=model, id="test", tools=[])

        scheduler._prepare_agent(agent)
        count_after_first = len(agent.tools)
        scheduler._prepare_agent(agent)
        assert len(agent.tools) == count_after_first
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_prepare_registers_agent(self):
        scheduler = Scheduler()
        model = MockModel()
        agent = Agent(name="test", description="test", model=model, id="test", tools=[])

        scheduler._prepare_agent(agent)
        assert "test" in scheduler._agents
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_prepare_enables_termination_summary(self):
        from agiwo.agent.options import AgentOptions
        scheduler = Scheduler()
        model = MockModel()
        opts = AgentOptions(enable_termination_summary=False)
        agent = Agent(name="test", description="test", model=model, id="test", tools=[], options=opts)
        scheduler._prepare_agent(agent)
        assert agent.options.enable_termination_summary is True
        await scheduler.stop()


class TestSchedulerSubmit:
    @pytest.mark.asyncio
    async def test_submit_creates_state(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Hello")])
            agent = Agent(name="test", description="test", model=model, id="test", tools=[])

            state_id = await scheduler.submit(agent, "Hello")
            assert state_id == "test"

            state = await scheduler._store.get_state("test")
            assert state is not None
            assert state.task == "Hello"
            assert state.depth == 0

            await asyncio.sleep(0.5)

    @pytest.mark.asyncio
    async def test_submit_persistent(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Hello")])
            agent = Agent(name="persist", description="test", model=model, id="persist", tools=[])

            state_id = await scheduler.submit(agent, "Hello", persistent=True)
            state = await scheduler._store.get_state(state_id)
            assert state is not None
            assert state.is_persistent is True

            await asyncio.sleep(0.5)

    @pytest.mark.asyncio
    async def test_submit_rejects_concurrent(self):
        scheduler = Scheduler(_fast_config())
        await scheduler.start()

        state = AgentState(
            id="test",
            session_id="sess",
            status=AgentStateStatus.RUNNING,
            task="busy",
        )
        await scheduler._store.save_state(state)

        model = MockModel()
        agent = Agent(name="test", description="test", model=model, id="test", tools=[])

        with pytest.raises(RuntimeError, match="already active"):
            await scheduler.submit(agent, "Another task")

        await scheduler.stop()


class TestSchedulerSimpleCompletion:
    @pytest.mark.asyncio
    async def test_run_simple_agent(self):
        """Agent completes without sleeping — simplest case."""
        model = MockModel([_simple_completion("The answer is 42")])
        agent = Agent(name="simple", description="test", model=model, id="simple", tools=[])

        async with Scheduler(_fast_config()) as scheduler:
            result = await scheduler.run(agent, "What is the answer?", timeout=5)

        assert result.termination_reason == TerminationReason.COMPLETED
        assert result.response == "The answer is 42"


class TestSchedulerCreateChildAgent:
    @pytest.mark.asyncio
    async def test_create_child_copies_config(self):
        scheduler = Scheduler()
        model = MockModel()
        parent = Agent(
            name="parent",
            description="parent agent",
            model=model,
            id="parent",
            tools=[],
            system_prompt="Be helpful",
        )
        scheduler._prepare_agent(parent)

        state = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.PENDING,
            task="sub-task",
            parent_id="parent",
        )
        child = scheduler._create_child_agent(state)

        assert child.id == "child-1"
        assert child.name == "parent"
        assert child.model is parent.model
        assert "Be helpful" in child.system_prompt
        child_tool_names = {t.get_name() for t in child.tools}
        assert "spawn_agent" not in child_tool_names
        assert "sleep_and_wait" in child_tool_names
        assert "query_spawned_agent" in child_tool_names
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_create_child_with_overrides(self):
        scheduler = Scheduler()
        model = MockModel()
        parent = Agent(
            name="parent",
            description="parent agent",
            model=model,
            id="parent",
            tools=[],
            system_prompt="Default prompt",
        )
        scheduler._prepare_agent(parent)

        state = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.PENDING,
            task="sub-task",
            parent_id="parent",
            config_overrides={"instruction": "Focus on specialized area."},
        )
        child = scheduler._create_child_agent(state)

        # Child inherits parent's base system_prompt (before environment section)
        # but has its own workspace paths due to different agent_id
        assert "Default prompt" in child.system_prompt
        assert "child-1" in child.system_prompt  # Child's own ID in workspace paths
        # Instruction is stored for runtime injection via <system-instruction> tag
        assert state.config_overrides["instruction"] == "Focus on specialized area."
        await scheduler.stop()


class TestSchedulerWakeMessage:
    @pytest.mark.asyncio
    async def test_waitset_message(self):
        scheduler = Scheduler()
        store = scheduler._store

        child_state = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.COMPLETED,
            task="A",
            parent_id="orch",
            result_summary="Result A",
        )
        await store.save_state(child_state)

        state = AgentState(
            id="orch",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="root",
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=["child-1"],
                completed_ids=["child-1"],
            ),
        )
        msg = await scheduler._build_wake_message(state)
        assert "Child agents completed" in msg
        assert "Result A" in msg
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_timer_message(self):
        scheduler = Scheduler()
        state = AgentState(
            id="orch",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="root",
            wake_condition=WakeCondition(type=WakeType.TIMER),
        )
        msg = await scheduler._build_wake_message(state)
        assert "delay" in msg.lower()
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_periodic_message(self):
        scheduler = Scheduler()
        state = AgentState(
            id="orch",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="root",
            wake_condition=WakeCondition(type=WakeType.PERIODIC),
        )
        msg = await scheduler._build_wake_message(state)
        assert "periodic" in msg.lower()
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_task_submitted_message(self):
        scheduler = Scheduler()
        state = AgentState(
            id="orch",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="root",
            wake_condition=WakeCondition(
                type=WakeType.TASK_SUBMITTED,
                submitted_task="Do X",
            ),
        )
        msg = await scheduler._build_wake_message(state)
        assert "Do X" in msg
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_no_condition_message(self):
        scheduler = Scheduler()
        state = AgentState(
            id="orch",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="root",
        )
        msg = await scheduler._build_wake_message(state)
        assert "woken up" in msg.lower()
        await scheduler.stop()


class TestSchedulerSignalPropagation:
    @pytest.mark.asyncio
    async def test_propagate_signals(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        parent = AgentState(
            id="parent",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="root",
            parent_id=None,
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=["child-1", "child-2"],
                completed_ids=[],
            ),
        )
        await store.save_state(parent)

        child1 = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.COMPLETED,
            task="A",
            parent_id="parent",
            result_summary="Done A",
        )
        await store.save_state(child1)

        await scheduler._propagate_signals()

        parent_state = await store.get_state("parent")
        assert parent_state is not None
        assert parent_state.wake_condition is not None
        assert "child-1" in parent_state.wake_condition.completed_ids

        child1_state = await store.get_state("child-1")
        assert child1_state is not None
        assert child1_state.signal_propagated is True

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_propagate_failed_child(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        parent = AgentState(
            id="parent",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="root",
            parent_id=None,
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=["child-1"],
                completed_ids=[],
            ),
        )
        await store.save_state(parent)

        child1 = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.FAILED,
            task="A",
            parent_id="parent",
            result_summary="Error",
        )
        await store.save_state(child1)

        await scheduler._propagate_signals()

        parent_state = await store.get_state("parent")
        assert "child-1" in parent_state.wake_condition.completed_ids
        await scheduler.stop()


class TestSchedulerSubmitTask:
    @pytest.mark.asyncio
    async def test_submit_task_to_persistent(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="initial",
            parent_id=None,
            is_persistent=True,
            wake_condition=WakeCondition(type=WakeType.TASK_SUBMITTED),
        )
        await store.save_state(state)

        await scheduler.submit_task("root", "New work")

        updated = await store.get_state("root")
        assert updated.wake_condition.type == WakeType.TASK_SUBMITTED
        assert updated.wake_condition.submitted_task == "New work"
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_submit_task_rejects_non_persistent(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="initial",
            parent_id=None,
            is_persistent=False,
        )
        await store.save_state(state)

        with pytest.raises(RuntimeError, match="not persistent"):
            await scheduler.submit_task("root", "New work")
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_submit_task_rejects_non_sleeping(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.RUNNING,
            task="initial",
            parent_id=None,
            is_persistent=True,
        )
        await store.save_state(state)

        with pytest.raises(RuntimeError, match="not SLEEPING"):
            await scheduler.submit_task("root", "New work")
        await scheduler.stop()


class TestSchedulerCancel:
    @pytest.mark.asyncio
    async def test_recursive_cancel(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        root = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.RUNNING,
            task="root",
            parent_id=None,
        )
        child = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.RUNNING,
            task="child",
            parent_id="root",
        )
        grandchild = AgentState(
            id="gc-1",
            session_id="sess",
            status=AgentStateStatus.PENDING,
            task="grandchild",
            parent_id="child-1",
        )
        await store.save_state(root)
        await store.save_state(child)
        await store.save_state(grandchild)

        result = await scheduler.cancel("root", "test cancel")
        assert result is True

        for sid in ["root", "child-1", "gc-1"]:
            s = await store.get_state(sid)
            assert s.status == AgentStateStatus.FAILED
            assert s.result_summary == "test cancel"

        await scheduler.stop()


class TestSchedulerShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_sleeping_agent(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="root",
            parent_id=None,
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=["child-1"],
            ),
        )
        await store.save_state(state)

        result = await scheduler.shutdown("root")
        assert result is True

        updated = await store.get_state("root")
        assert updated.wake_condition.type == WakeType.TASK_SUBMITTED
        assert "shutdown" in updated.wake_condition.submitted_task.lower()
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_shutdown_pending_agent(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.PENDING,
            task="root",
            parent_id=None,
        )
        await store.save_state(state)

        result = await scheduler.shutdown("root")
        assert result is True

        updated = await store.get_state("root")
        assert updated.status == AgentStateStatus.FAILED
        assert "Shutdown" in updated.result_summary
        await scheduler.stop()
