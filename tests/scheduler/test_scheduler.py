"""Integration tests for the Scheduler class."""

import asyncio
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agiwo.agent import Agent
from agiwo.agent import AgentConfig, AgentOptions
from agiwo.agent import AgentHooks
from agiwo.agent import AgentStreamItem, RunCompletedEvent, TerminationReason
from agiwo.utils.abort_signal import AbortSignal
from agiwo.llm.base import Model, StreamChunk
from agiwo.scheduler._tick import propagate_signals
from agiwo.scheduler.commands import DispatchAction, DispatchReason
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerConfig,
    SchedulerEventType,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.formatting import build_wake_message
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.store.memory import InMemoryAgentStateStorage


_TEST_CLEANUP_WAIT_FIRST = 0.1
_TEST_CLEANUP_WAIT_SECOND = 0.05
_TEST_SETTLE_WAIT = 0.1
_TEST_RUN_TIMEOUT = 2


def _cleanup_agiwo_workspace(agent_names: list[str]) -> None:
    """Move agent workspace folders from .agiwo/ to trash directory.

    Per user rules: never use rm, always mv to trash.
    Note: config_root is relative path ".agiwo/", so workspaces are created
    in project directory, not home directory.
    """
    # Get project root (parent of tests/ directory = tests/scheduler/ -> tests/ -> project_root)
    project_dir = Path(__file__).parent.parent.parent
    agiwo_dir = project_dir / ".agiwo"
    trash_dir = project_dir / "trash" / "agiwo_test_cleanup"

    for name in agent_names:
        workspace = agiwo_dir / name
        if workspace.exists():
            # Create trash subdirectory if needed
            trash_dir.mkdir(parents=True, exist_ok=True)
            # Move to trash with timestamp to avoid conflicts
            dest = trash_dir / f"{name}_{time.time_ns()}"
            shutil.move(str(workspace), str(dest))


@pytest.fixture(autouse=True)
async def cleanup_agiwo_folders():
    """Automatically cleanup .agiwo folders after each test."""
    yield
    # Wait for any async Agent initialization to complete
    await asyncio.sleep(_TEST_CLEANUP_WAIT_FIRST)
    # List of agent names used in tests that create workspaces
    agent_names = ["test", "persist", "simple", "parent", "child-1"]
    _cleanup_agiwo_workspace(agent_names)
    # Second cleanup after short delay for any stragglers
    await asyncio.sleep(_TEST_CLEANUP_WAIT_SECOND)
    _cleanup_agiwo_workspace(agent_names)


def _fast_config(**kwargs) -> SchedulerConfig:
    """SchedulerConfig with short check_interval for tests."""
    return SchedulerConfig(check_interval=0.1, **kwargs)


class _FakeEncoding:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


async def _noop_memory_retrieve(*_args, **_kwargs) -> list:
    return []


@pytest.fixture(autouse=True)
def _patch_tiktoken_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agiwo.llm.usage_resolver._resolve_encoding",
        lambda _model_name: _FakeEncoding(),
    )


def _make_agent(
    *,
    name: str,
    model: Model,
    id: str,
    tools: list | None = None,
    system_prompt: str = "",
    options: AgentOptions | None = None,
) -> Agent:
    return Agent(
        AgentConfig(
            name=name,
            description="test" if name != "parent" else "parent agent",
            system_prompt=system_prompt,
            options=options or AgentOptions(),
        ),
        model=model,
        tools=tools,
        hooks=AgentHooks(on_memory_retrieve=_noop_memory_retrieve),
        id=id,
    )


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


class EchoMessagesModel(Model):
    """Model that echoes the raw prompt payload for assertion-friendly tests."""

    def __init__(self, *, delay_seconds: float = 0.0) -> None:
        super().__init__(id="echo", name="echo", temperature=0.0)
        self.delay_seconds = delay_seconds
        self.calls: list[str] = []

    async def arun_stream(self, messages, tools=None):
        del tools
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        payload = str(messages)
        self.calls.append(payload)
        yield StreamChunk(content=payload)
        yield StreamChunk(finish_reason="stop")


def _simple_completion(text: str = "Done") -> list[StreamChunk]:
    return [StreamChunk(content=text), StreamChunk(finish_reason="stop")]


def _tool_call(
    tool_name: str, args: str = "{}", tc_id: str = "tc-1"
) -> list[StreamChunk]:
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
    async def test_submit_prepares_runtime_clone(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Hello")])
            agent = _make_agent(name="test", model=model, id="test", tools=[])

            state_id = await scheduler.submit(agent, "Hello")
            registered = scheduler.get_registered_agent(state_id)
            assert registered is not None
            assert registered is not agent

            tool_names = {t.get_name() for t in registered.tools}
            assert "spawn_agent" in tool_names
            assert "sleep_and_wait" in tool_names
            assert "query_spawned_agent" in tool_names

    @pytest.mark.asyncio
    async def test_submit_does_not_mutate_original_agent(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Hello")])
            opts = AgentOptions(enable_termination_summary=False)
            agent = _make_agent(
                name="test", model=model, id="test", tools=[], options=opts
            )

            await scheduler.submit(agent, "Hello")

            tool_names = {t.get_name() for t in agent.tools}
            assert "spawn_agent" not in tool_names
            assert "sleep_and_wait" not in tool_names
            assert "query_spawned_agent" not in tool_names
            assert agent.options.enable_termination_summary is False

    @pytest.mark.asyncio
    async def test_submit_registers_agent(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Hello")])
            agent = _make_agent(name="test", model=model, id="test", tools=[])

            await scheduler.submit(agent, "Hello")
            assert scheduler.get_registered_agent("test") is not None


class TestSchedulerSubmit:
    @pytest.mark.asyncio
    async def test_submit_creates_state(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Hello")])
            agent = _make_agent(name="test", model=model, id="test", tools=[])

            state_id = await scheduler.submit(agent, "Hello")
            assert state_id == "test"

            state = await scheduler._store.get_state("test")
            assert state is not None
            assert state.task == "Hello"
            assert state.depth == 0

            await asyncio.sleep(_TEST_SETTLE_WAIT)

    @pytest.mark.asyncio
    async def test_submit_persistent(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Hello")])
            agent = _make_agent(name="persist", model=model, id="persist", tools=[])

            state_id = await scheduler.submit(agent, "Hello", persistent=True)
            state = await scheduler._store.get_state(state_id)
            assert state is not None
            assert state.is_persistent is True

            await asyncio.sleep(_TEST_SETTLE_WAIT)

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
        agent = _make_agent(name="test", model=model, id="test", tools=[])

        with pytest.raises(RuntimeError, match="already active"):
            await scheduler.submit(agent, "Another task")

        await scheduler.stop()


class TestSchedulerSimpleCompletion:
    @pytest.mark.asyncio
    async def test_run_simple_agent(self):
        """Agent completes without sleeping — simplest case."""
        model = MockModel([_simple_completion("The answer is 42")])
        agent = _make_agent(name="simple", model=model, id="simple", tools=[])

        async with Scheduler(_fast_config()) as scheduler:
            result = await scheduler.run(
                agent, "What is the answer?", timeout=_TEST_RUN_TIMEOUT
            )

        assert result.termination_reason == TerminationReason.COMPLETED
        assert result.response == "The answer is 42"


class TestSchedulerCreateChildAgent:
    @pytest.mark.asyncio
    async def test_create_child_copies_config(self):
        scheduler = Scheduler()
        model = MockModel()
        parent = _make_agent(
            name="parent",
            model=model,
            id="parent",
            tools=[],
            system_prompt="Be helpful",
        )
        state_id = await scheduler.submit(parent, "root task")
        assert state_id == "parent"

        state = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.PENDING,
            task="sub-task",
            parent_id="parent",
        )
        child_agent = await scheduler._runner.create_child_agent(state)

        assert child_agent.id == "child-1"
        assert child_agent.name == "parent"
        assert child_agent.model is parent.model
        assert child_agent.hooks is not parent.hooks
        assert child_agent.run_step_storage is not parent.run_step_storage
        assert child_agent.session_storage is not parent.session_storage
        assert child_agent.options.enable_termination_summary is True
        # Check child inherited parent's system prompt (get_effective_system_prompt triggers build)
        child_prompt = await child_agent.get_effective_system_prompt()
        assert "Be helpful" in child_prompt
        child_tool_names = {t.get_name() for t in child_agent.tools}
        assert "spawn_agent" not in child_tool_names
        assert "sleep_and_wait" in child_tool_names
        assert "query_spawned_agent" in child_tool_names
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_create_child_with_overrides(self):
        scheduler = Scheduler()
        model = MockModel()
        parent = _make_agent(
            name="parent",
            model=model,
            id="parent",
            tools=[],
            system_prompt="Default prompt",
        )
        state_id = await scheduler.submit(parent, "root task")
        assert state_id == "parent"

        state = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.PENDING,
            task="sub-task",
            parent_id="parent",
            config_overrides={"instruction": "Focus on specialized area."},
        )
        child_agent = await scheduler._runner.create_child_agent(state)

        # Check child inherited parent's fully built system_prompt
        child_prompt = await child_agent.get_effective_system_prompt()
        assert "Default prompt" in child_prompt
        # Instruction is stored for runtime injection via <system-instruction> tag
        assert state.config_overrides["instruction"] == "Focus on specialized area."
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_create_child_with_system_prompt_override(self):
        scheduler = Scheduler()
        model = MockModel()
        parent = _make_agent(
            name="parent",
            model=model,
            id="parent",
            tools=[],
            system_prompt="Default prompt",
        )
        state_id = await scheduler.submit(parent, "root task")
        assert state_id == "parent"

        state = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.PENDING,
            task="sub-task",
            parent_id="parent",
            config_overrides={"system_prompt": "You are a specialist."},
        )
        child_agent = await scheduler._runner.create_child_agent(state)

        child_prompt = await child_agent.get_effective_system_prompt()
        assert "You are a specialist." in child_prompt
        assert "Default prompt" not in child_prompt
        await scheduler.stop()


class TestSchedulerWakeMessage:
    @pytest.mark.asyncio
    async def test_waitset_message(self):
        wc = WakeCondition(
            type=WakeType.WAITSET,
            wait_for=["child-1"],
            completed_ids=["child-1"],
        )
        msg = build_wake_message(
            wc,
            succeeded={"child-1": "Result A"},
            failed={},
        )
        assert "Child agents completed" in msg
        assert "Result A" in msg

    @pytest.mark.asyncio
    async def test_timer_message(self):
        wc = WakeCondition(type=WakeType.TIMER)
        msg = build_wake_message(wc, succeeded={}, failed={})
        assert "delay" in msg.lower()

    @pytest.mark.asyncio
    async def test_periodic_message(self):
        wc = WakeCondition(type=WakeType.PERIODIC)
        msg = build_wake_message(wc, succeeded={}, failed={})
        assert "periodic" in msg.lower()

    @pytest.mark.asyncio
    async def test_no_condition_message(self):
        msg = build_wake_message(None, succeeded={}, failed={})
        assert "woken up" in msg.lower()


class TestSchedulerRunnerCleanup:
    @pytest.mark.asyncio
    async def test_timeout_wake_cleans_abort_signal(self):
        scheduler = Scheduler()
        model = MockModel([_simple_completion("Recovered after timeout")])
        agent = _make_agent(name="root", model=model, id="root", tools=[])
        prepared_agent = await agent.create_child_agent(
            child_id=agent.id,
            system_prompt_override=agent.config.system_prompt,
            exclude_tool_names={tool.get_name() for tool in agent.tools},
            extra_tools=list(scheduler._scheduling_tools),
        )
        scheduler._rt.agents[prepared_agent.id] = prepared_agent
        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.WAITING,
            task="root",
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=[],
            ),
        )
        scheduler._rt.abort_signals["root"] = AbortSignal()
        await scheduler._runner.run(
            DispatchAction(state=state, reason=DispatchReason.WAKE_TIMEOUT)
        )

        assert scheduler._rt.abort_signals.get("root") is None
        await scheduler.stop()


class TestSchedulerTimeoutDispatch:
    @pytest.mark.asyncio
    async def test_enforce_timeouts_deduplicates_dispatched_state(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store
        state = AgentState(
            id="sleepy",
            session_id="sess",
            status=AgentStateStatus.WAITING,
            task="wait",
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                timeout_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            ),
        )
        await store.save_state(state)

        started = asyncio.Event()
        release = asyncio.Event()
        call_count = 0

        async def fake_wake_for_timeout(target: AgentState) -> None:
            nonlocal call_count
            assert target.id == "sleepy"
            call_count += 1
            started.set()
            await release.wait()

        async def fake_run(action: DispatchAction) -> None:
            if action.reason != DispatchReason.WAKE_TIMEOUT:
                return
            await fake_wake_for_timeout(action.state)

        scheduler._runner.run = fake_run  # type: ignore[method-assign]

        await scheduler.tick()
        await started.wait()

        assert call_count == 1
        assert "sleepy" in scheduler._rt.dispatched

        await scheduler.tick()
        await asyncio.sleep(0)

        assert call_count == 1

        release.set()
        await asyncio.sleep(_TEST_SETTLE_WAIT)
        scheduler._rt.dispatched.discard("sleepy")
        await scheduler.stop()


class TestSchedulerSignalPropagation:
    @pytest.mark.asyncio
    async def test_propagate_signals(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        parent = AgentState(
            id="parent",
            session_id="sess",
            status=AgentStateStatus.WAITING,
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

        await propagate_signals(scheduler)

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
            status=AgentStateStatus.WAITING,
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

        await propagate_signals(scheduler)

        parent_state = await store.get_state("parent")
        assert "child-1" in parent_state.wake_condition.completed_ids
        await scheduler.stop()


class TestSchedulerEnqueueInput:
    @pytest.mark.asyncio
    async def test_enqueue_input_to_persistent(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.IDLE,
            task="initial",
            parent_id=None,
            is_persistent=True,
        )
        await store.save_state(state)

        await scheduler.enqueue_input("root", "New work")

        updated = await store.get_state("root")
        assert updated.status == AgentStateStatus.QUEUED
        assert updated.pending_input == "New work"
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_enqueue_input_rejects_non_persistent(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.IDLE,
            task="initial",
            parent_id=None,
            is_persistent=False,
        )
        await store.save_state(state)

        with pytest.raises(RuntimeError, match="not persistent"):
            await scheduler.enqueue_input("root", "New work")
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_enqueue_input_rejects_running(self):
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

        with pytest.raises(RuntimeError, match="IDLE or FAILED"):
            await scheduler.enqueue_input("root", "New work")
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_enqueue_input_rejects_completed(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.COMPLETED,
            task="initial",
            parent_id=None,
            is_persistent=True,
        )
        await store.save_state(state)

        with pytest.raises(RuntimeError, match="IDLE or FAILED"):
            await scheduler.enqueue_input("root", "Resume work")
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_enqueue_input_accepts_failed(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.FAILED,
            task="initial",
            parent_id=None,
            is_persistent=True,
        )
        await store.save_state(state)

        await scheduler.enqueue_input("root", "Retry work")
        updated = await store.get_state("root")
        assert updated is not None
        assert updated.status == AgentStateStatus.QUEUED
        assert updated.pending_input == "Retry work"
        await scheduler.stop()


class TestSchedulerQueuedMailbox:
    @pytest.mark.asyncio
    async def test_queued_root_preserves_mailbox_input_for_next_run(self):
        scheduler = Scheduler(_fast_config(event_debounce_min_count=1))
        model = EchoMessagesModel()
        agent = _make_agent(name="root", model=model, id="root", tools=[])
        prepared_agent = await agent.create_child_agent(
            child_id=agent.id,
            system_prompt_override=agent.config.system_prompt,
            exclude_tool_names={tool.get_name() for tool in agent.tools},
            extra_tools=list(scheduler._scheduling_tools),
        )
        scheduler._rt.agents[prepared_agent.id] = prepared_agent
        await scheduler._store.save_state(
            AgentState(
                id="root",
                session_id="sess",
                status=AgentStateStatus.QUEUED,
                task="initial",
                pending_input="first input",
                is_persistent=True,
            )
        )

        steered = await scheduler.steer("root", "second input")
        assert steered is True

        await scheduler.tick()
        result = await scheduler.wait_for("root", timeout=_TEST_RUN_TIMEOUT)

        assert result.response is not None
        assert "first input" in result.response
        assert "second input" in result.response
        await scheduler.stop()


class TestSchedulerStream:
    @pytest.mark.asyncio
    async def test_stream_new_root_run(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Streamed answer")])
            agent = _make_agent(
                name="stream-root", model=model, id="stream-root", tools=[]
            )

            items = [
                item
                async for item in scheduler.stream(
                    "What happened?",
                    agent=agent,
                    session_id="sess-stream",
                    timeout=_TEST_RUN_TIMEOUT,
                )
            ]

        assert items
        assert isinstance(items[-1], RunCompletedEvent)
        assert items[-1].response == "Streamed answer"

    @pytest.mark.asyncio
    async def test_stream_enqueues_input_for_persistent_root(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel(
                [
                    _simple_completion("First answer"),
                    _simple_completion("Second answer"),
                ]
            )
            agent = _make_agent(
                name="stream-persist", model=model, id="stream-persist", tools=[]
            )

            state_id = await scheduler.submit(
                agent,
                "first",
                session_id="sess-persist",
                persistent=True,
            )
            first = await scheduler.wait_for(state_id, timeout=_TEST_RUN_TIMEOUT)
            assert first.response == "First answer"

            items = [
                item
                async for item in scheduler.stream(
                    "second",
                    agent=agent,
                    state_id=state_id,
                    timeout=_TEST_RUN_TIMEOUT,
                )
            ]

        assert items
        assert isinstance(items[-1], RunCompletedEvent)
        assert items[-1].response == "Second answer"

    @pytest.mark.asyncio
    async def test_stream_rejects_second_subscriber_for_same_root(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = EchoMessagesModel(delay_seconds=0.2)
            agent = _make_agent(
                name="stream-single", model=model, id="stream-single", tools=[]
            )

            state_id = await scheduler.submit(
                agent,
                "first",
                session_id="sess-stream-single",
                persistent=True,
            )
            await scheduler.wait_for(state_id, timeout=_TEST_RUN_TIMEOUT)

            async def consume_stream(user_input: str) -> list[AgentStreamItem]:
                return [
                    item
                    async for item in scheduler.stream(
                        user_input,
                        agent=agent,
                        state_id=state_id,
                        timeout=_TEST_RUN_TIMEOUT,
                    )
                ]

            first_consumer = asyncio.create_task(consume_stream("second"))
            await asyncio.sleep(0.05)

            with pytest.raises(RuntimeError, match="subscriber"):
                await consume_stream("third")

            await first_consumer


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


class TestSchedulerDebounce:
    @pytest.mark.asyncio
    async def test_process_pending_events_wakes_waiting_agent(self):
        """A WAITING parent agent with pending events should be woken."""
        scheduler = Scheduler(_fast_config(event_debounce_min_count=1))
        store = scheduler._store
        model = MockModel([_simple_completion("done")])
        prepared_agent = await _make_agent(
            name="parent", model=model, id="parent-dbounce", tools=[]
        ).create_child_agent(
            child_id="parent-dbounce",
            system_prompt_override="",
            exclude_tool_names=set(),
            extra_tools=list(scheduler._scheduling_tools),
        )
        scheduler._rt.agents[prepared_agent.id] = prepared_agent

        parent = AgentState(
            id="parent-dbounce",
            session_id="sess",
            status=AgentStateStatus.WAITING,
            task="parent",
            parent_id=None,
            wake_count=0,
        )
        await store.save_state(parent)

        event = PendingEvent(
            id="evt-1",
            target_agent_id="parent-dbounce",
            session_id="sess",
            event_type=SchedulerEventType.CHILD_COMPLETED,
            payload={"result": "done", "child_agent_id": "child-x"},
            source_agent_id="child-x",
            created_at=datetime.now(timezone.utc),
        )
        await store.save_event(event)

        # _process_pending_events should detect and dispatch
        await scheduler.tick()

        await asyncio.sleep(_TEST_SETTLE_WAIT)

        # Event should be deleted once the dispatched run has started.
        remaining = await store.list_events(
            target_agent_id="parent-dbounce",
            session_id="sess",
        )
        assert len(remaining) == 0

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_process_pending_events_skips_non_waiting(self):
        """Non-WAITING agents should NOT be woken via debounce."""
        scheduler = Scheduler(_fast_config(event_debounce_min_count=1))
        store = scheduler._store

        running_agent = AgentState(
            id="agent-running",
            session_id="sess",
            status=AgentStateStatus.RUNNING,
            task="task",
            parent_id=None,
        )
        await store.save_state(running_agent)

        event = PendingEvent(
            id="evt-2",
            target_agent_id="agent-running",
            session_id="sess",
            event_type=SchedulerEventType.USER_HINT,
            payload={"hint": "check this"},
            source_agent_id=None,
            created_at=datetime.now(timezone.utc),
        )
        await store.save_event(event)

        before_dispatched = len(scheduler._rt.dispatched)
        await scheduler.tick()
        after_dispatched = len(scheduler._rt.dispatched)

        # No new dispatch should have happened
        assert after_dispatched == before_dispatched
        await scheduler.stop()


class TestSchedulerShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_running_persistent_root_requeues_summary(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = EchoMessagesModel(delay_seconds=0.2)
            agent = _make_agent(name="root", model=model, id="root", tools=[])

            state_id = await scheduler.submit(
                agent,
                "initial",
                session_id="sess",
                persistent=True,
            )

            await asyncio.sleep(0.05)
            result = await scheduler.shutdown(state_id)
            assert result is True

            await asyncio.sleep(0.3)
            updated = await scheduler.get_state(state_id)
            assert updated is not None
            assert updated.status in (
                AgentStateStatus.QUEUED,
                AgentStateStatus.RUNNING,
            )
            shutdown_text = (
                updated.pending_input if updated.pending_input else updated.task
            )
            assert isinstance(shutdown_text, str)
            assert "shutdown" in shutdown_text.lower()

    @pytest.mark.asyncio
    async def test_shutdown_waiting_root(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.WAITING,
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
        assert updated.status == AgentStateStatus.QUEUED
        assert updated.pending_input is not None
        assert "shutdown" in updated.pending_input.lower()
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


class TestSchedulerWaiters:
    @pytest.mark.asyncio
    async def test_wait_for_supports_multiple_concurrent_waiters(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = EchoMessagesModel(delay_seconds=0.1)
            agent = _make_agent(name="multiwait", model=model, id="multiwait", tools=[])

            state_id = await scheduler.submit(
                agent,
                "hello",
                session_id="sess-multiwait",
            )

            first_task = asyncio.create_task(
                scheduler.wait_for(state_id, timeout=_TEST_RUN_TIMEOUT)
            )
            second_task = asyncio.create_task(
                scheduler.wait_for(state_id, timeout=_TEST_RUN_TIMEOUT)
            )

            first, second = await asyncio.gather(first_task, second_task)

        assert first.termination_reason == TerminationReason.COMPLETED
        assert second.termination_reason == TerminationReason.COMPLETED
        assert first.response == second.response
