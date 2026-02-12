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
from agiwo.scheduler.models import AgentState, AgentStateStatus, SchedulerConfig, WakeCondition, WakeType
from agiwo.scheduler.scheduler import Scheduler
from agiwo.scheduler.store import InMemoryAgentStateStorage
from agiwo.tool.base import BaseTool, ToolResult


def _fast_config() -> SchedulerConfig:
    """SchedulerConfig with short check_interval for tests."""
    return SchedulerConfig(check_interval=0.1)


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
        agent = Agent(id="test", description="test", model=model, tools=[])

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
        agent = Agent(id="test", description="test", model=model, tools=[])

        scheduler._prepare_agent(agent)
        count_after_first = len(agent.tools)
        scheduler._prepare_agent(agent)
        assert len(agent.tools) == count_after_first
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_prepare_registers_agent(self):
        scheduler = Scheduler()
        model = MockModel()
        agent = Agent(id="test", description="test", model=model, tools=[])

        scheduler._prepare_agent(agent)
        assert "test" in scheduler._agents
        await scheduler.stop()


class TestSchedulerSubmit:
    @pytest.mark.asyncio
    async def test_submit_creates_state(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Hello")])
            agent = Agent(id="test", description="test", model=model, tools=[])

            state_id = await scheduler.submit(agent, "Hello")
            assert state_id == "test"

            state = await scheduler._store.get_state("test")
            assert state is not None
            assert state.task == "Hello"

            # Wait for the task to complete
            await asyncio.sleep(0.5)

    @pytest.mark.asyncio
    async def test_submit_rejects_concurrent(self):
        scheduler = Scheduler(_fast_config())
        await scheduler.start()

        # Manually create an active state
        state = AgentState(
            id="test",
            session_id="sess",
            agent_id="test",
            parent_agent_id="test",
            status=AgentStateStatus.RUNNING,
            task="busy",
        )
        await scheduler._store.save_state(state)

        model = MockModel()
        agent = Agent(id="test", description="test", model=model, tools=[])

        with pytest.raises(RuntimeError, match="already active"):
            await scheduler.submit(agent, "Another task")

        await scheduler.stop()


class TestSchedulerSimpleCompletion:
    @pytest.mark.asyncio
    async def test_run_simple_agent(self):
        """Agent completes without sleeping — simplest case."""
        model = MockModel([_simple_completion("The answer is 42")])
        agent = Agent(id="simple", description="test", model=model, tools=[])

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
            id="parent",
            description="parent agent",
            model=model,
            tools=[],
            system_prompt="Be helpful",
        )
        scheduler._prepare_agent(parent)

        state = AgentState(
            id="child-1",
            session_id="sess",
            agent_id="child-1",
            parent_agent_id="parent",
            parent_state_id="parent",
            status=AgentStateStatus.PENDING,
            task="sub-task",
        )
        child = scheduler._create_child_agent(state)

        assert child.id == "child-1"
        assert child.model is parent.model
        assert "Be helpful" in child.system_prompt
        # Child should have scheduling tools (inherited from parent)
        child_tool_names = {t.get_name() for t in child.tools}
        assert "spawn_agent" in child_tool_names
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_create_child_with_overrides(self):
        scheduler = Scheduler()
        model = MockModel()
        parent = Agent(
            id="parent",
            description="parent agent",
            model=model,
            tools=[],
            system_prompt="Default prompt",
        )
        scheduler._prepare_agent(parent)

        state = AgentState(
            id="child-1",
            session_id="sess",
            agent_id="child-1",
            parent_agent_id="parent",
            parent_state_id="parent",
            status=AgentStateStatus.PENDING,
            task="sub-task",
            config_overrides={"system_prompt": "Specialized prompt"},
        )
        child = scheduler._create_child_agent(state)

        assert child.system_prompt != parent.system_prompt
        assert "Specialized" in child.system_prompt
        await scheduler.stop()


class TestSchedulerWakeMessage:
    def test_children_complete_message(self):
        scheduler = Scheduler()
        state = AgentState(
            id="orch",
            session_id="sess",
            agent_id="orch",
            parent_agent_id="orch",
            status=AgentStateStatus.SLEEPING,
            task="root",
            wake_condition=WakeCondition(
                type=WakeType.CHILDREN_COMPLETE,
                total_children=3,
                completed_children=3,
            ),
        )
        msg = scheduler._build_wake_message(state)
        assert "3 child agents" in msg
        assert "query_spawned_agent" in msg

    def test_delay_message(self):
        scheduler = Scheduler()
        state = AgentState(
            id="orch",
            session_id="sess",
            agent_id="orch",
            parent_agent_id="orch",
            status=AgentStateStatus.SLEEPING,
            task="root",
            wake_condition=WakeCondition(type=WakeType.DELAY),
        )
        msg = scheduler._build_wake_message(state)
        assert "delay" in msg.lower()

    def test_interval_message(self):
        scheduler = Scheduler()
        state = AgentState(
            id="orch",
            session_id="sess",
            agent_id="orch",
            parent_agent_id="orch",
            status=AgentStateStatus.SLEEPING,
            task="root",
            wake_condition=WakeCondition(type=WakeType.INTERVAL),
        )
        msg = scheduler._build_wake_message(state)
        assert "interval" in msg.lower()

    def test_no_condition_message(self):
        scheduler = Scheduler()
        state = AgentState(
            id="orch",
            session_id="sess",
            agent_id="orch",
            parent_agent_id="orch",
            status=AgentStateStatus.SLEEPING,
            task="root",
        )
        msg = scheduler._build_wake_message(state)
        assert "woken up" in msg.lower()


class TestSchedulerSignalPropagation:
    @pytest.mark.asyncio
    async def test_propagate_signals(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        # Parent sleeping, waiting for children
        parent = AgentState(
            id="parent",
            session_id="sess",
            agent_id="parent",
            parent_agent_id="parent",
            parent_state_id=None,
            status=AgentStateStatus.SLEEPING,
            task="root",
            wake_condition=WakeCondition(
                type=WakeType.CHILDREN_COMPLETE,
                total_children=2,
                completed_children=0,
            ),
        )
        await store.save_state(parent)

        # Child 1 completed but not propagated
        child1 = AgentState(
            id="child-1",
            session_id="sess",
            agent_id="child-1",
            parent_agent_id="parent",
            parent_state_id="parent",
            status=AgentStateStatus.COMPLETED,
            task="A",
            result_summary="Done A",
        )
        await store.save_state(child1)

        await scheduler._propagate_signals()

        # Parent should have 1 completed child
        parent_state = await store.get_state("parent")
        assert parent_state is not None
        assert parent_state.wake_condition is not None
        assert parent_state.wake_condition.completed_children == 1

        # Child 1 should be marked as propagated
        child1_state = await store.get_state("child-1")
        assert child1_state is not None
        assert child1_state.signal_propagated is True

        await scheduler.stop()
