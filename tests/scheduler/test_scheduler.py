"""Integration tests for the Scheduler class."""

import asyncio
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agiwo.agent import Agent, AgentConfig, AgentHooks, TerminationReason, extract_text
from agiwo.agent.options import AgentOptions
from agiwo.agent.scheduler_port import adapt_scheduler_agent
from agiwo.utils.abort_signal import AbortSignal
from agiwo.llm.base import Model, StreamChunk
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerConfig,
    SchedulerEventType,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.scheduler import Scheduler
from agiwo.scheduler.store import InMemoryAgentStateStorage


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
        agent = _make_agent(name="test", model=model, id="test", tools=[])

        scheduler._engine.prepare_agent(agent)

        tool_names = {t.get_name() for t in agent.tools}
        assert "spawn_agent" in tool_names
        assert "sleep_and_wait" in tool_names
        assert "query_spawned_agent" in tool_names
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_prepare_is_idempotent(self):
        scheduler = Scheduler()
        model = MockModel()
        agent = _make_agent(name="test", model=model, id="test", tools=[])

        agent_port = adapt_scheduler_agent(agent)
        scheduler._engine.prepare_agent(agent_port)
        count_after_first = len(agent.tools)
        scheduler._engine.prepare_agent(agent_port)
        assert len(agent.tools) == count_after_first
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_prepare_registers_agent(self):
        scheduler = Scheduler()
        model = MockModel()
        agent = _make_agent(name="test", model=model, id="test", tools=[])

        agent_port = adapt_scheduler_agent(agent)
        scheduler._engine.prepare_agent(agent_port)
        assert scheduler.get_registered_agent("test") is agent
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_prepare_enables_termination_summary(self):
        scheduler = Scheduler()
        model = MockModel()
        opts = AgentOptions(enable_termination_summary=False)
        agent = _make_agent(name="test", model=model, id="test", tools=[], options=opts)
        agent_port = adapt_scheduler_agent(agent)
        scheduler._engine.prepare_agent(agent_port)
        assert agent.options.enable_termination_summary is True
        await scheduler.stop()


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
        parent_port = adapt_scheduler_agent(parent)
        scheduler._engine.prepare_agent(parent_port)

        state = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.PENDING,
            task="sub-task",
            parent_id="parent",
        )
        child = await scheduler._runner.create_child_agent(state)

        assert child.id == "child-1"
        assert child.name == "parent"
        assert child.model is parent.model
        assert child.hooks is not parent.hooks
        assert child.run_step_storage is not parent.run_step_storage
        assert child.session_storage is not parent.session_storage
        assert child._runtime_state.prompt_runtime is not parent._runtime_state.prompt_runtime
        # Check child inherited parent's system prompt (get_effective_system_prompt triggers build)
        child_prompt = await child.get_effective_system_prompt()
        assert "Be helpful" in child_prompt
        child_tool_names = {t.get_name() for t in child.tools}
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
        parent_port = adapt_scheduler_agent(parent)
        scheduler._engine.prepare_agent(parent_port)

        state = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.PENDING,
            task="sub-task",
            parent_id="parent",
            config_overrides={"instruction": "Focus on specialized area."},
        )
        child = await scheduler._runner.create_child_agent(state)

        # Check child inherited parent's fully built system_prompt
        child_prompt = await child.get_effective_system_prompt()
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
        parent_port = adapt_scheduler_agent(parent)
        scheduler._engine.prepare_agent(parent_port)

        state = AgentState(
            id="child-1",
            session_id="sess",
            status=AgentStateStatus.PENDING,
            task="sub-task",
            parent_id="parent",
            config_overrides={"system_prompt": "You are a specialist."},
        )
        child = await scheduler._runner.create_child_agent(state)

        child_prompt = await child.get_effective_system_prompt()
        assert "You are a specialist." in child_prompt
        assert "Default prompt" not in child_prompt
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
        msg = await scheduler._runner.build_wake_message(state)
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
        msg = await scheduler._runner.build_wake_message(state)
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
        msg = await scheduler._runner.build_wake_message(state)
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
        msg = await scheduler._runner.build_wake_message(state)
        assert "Do X" in extract_text(msg)
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
        msg = await scheduler._runner.build_wake_message(state)
        assert "woken up" in msg.lower()
        await scheduler.stop()


class TestSchedulerRunnerCleanup:
    @pytest.mark.asyncio
    async def test_timeout_wake_cleans_abort_signal(self):
        scheduler = Scheduler()
        model = MockModel([_simple_completion("Recovered after timeout")])
        agent = _make_agent(name="root", model=model, id="root", tools=[])
        scheduler._engine.prepare_agent(agent)
        state = AgentState(
            id="root",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="root",
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=[],
            ),
        )
        scheduler._coordinator.set_abort_signal("root", AbortSignal())
        await scheduler._runner.wake_for_timeout(state)

        assert scheduler._coordinator.get_abort_signal("root") is None
        await scheduler.stop()


class TestSchedulerTimeoutDispatch:
    @pytest.mark.asyncio
    async def test_enforce_timeouts_deduplicates_dispatched_state(self):
        scheduler = Scheduler(_fast_config())
        store = scheduler._store
        state = AgentState(
            id="sleepy",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
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
            call_count += 1
            started.set()
            await release.wait()

        scheduler._runner.wake_for_timeout = fake_wake_for_timeout  # type: ignore[method-assign]

        await scheduler._engine._enforce_timeouts()
        await started.wait()

        assert call_count == 1
        assert "sleepy" in scheduler._coordinator.dispatched_state_ids

        await scheduler._engine._enforce_timeouts()
        await asyncio.sleep(0)

        assert call_count == 1

        release.set()
        await asyncio.sleep(_TEST_SETTLE_WAIT)
        scheduler._coordinator.release_state_dispatch("sleepy")
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

        await scheduler._engine._propagate_signals()

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

        await scheduler._engine._propagate_signals()

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
    async def test_submit_task_rejects_running(self):
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

        with pytest.raises(RuntimeError, match="SLEEPING, COMPLETED, or FAILED"):
            await scheduler.submit_task("root", "New work")
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_submit_task_accepts_completed(self):
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

        await scheduler.submit_task("root", "Resume work")
        updated = await store.get_state("root")
        assert updated is not None
        assert updated.status == AgentStateStatus.SLEEPING
        assert updated.wake_condition is not None
        assert updated.wake_condition.submitted_task == "Resume work"
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_submit_task_accepts_failed(self):
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

        await scheduler.submit_task("root", "Retry work")
        updated = await store.get_state("root")
        assert updated is not None
        assert updated.status == AgentStateStatus.SLEEPING
        assert updated.wake_condition is not None
        assert updated.wake_condition.submitted_task == "Retry work"
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


class TestSchedulerHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_emits_warning_for_stuck_agent(self):
        """Stuck RUNNING child agent should generate a HEALTH_WARNING pending event for parent."""
        scheduler = Scheduler(_fast_config())
        store = scheduler._store
        await scheduler.start()

        now = datetime.now(timezone.utc)
        # Agent with last_activity_at 600s ago (> threshold of 300s)
        parent = AgentState(
            id="parent-1",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
            task="parent",
            parent_id=None,
        )
        child = AgentState(
            id="child-stuck",
            session_id="sess",
            status=AgentStateStatus.RUNNING,
            task="stuck child",
            parent_id="parent-1",
            last_activity_at=now - timedelta(seconds=600),
        )
        await store.save_state(parent)
        await store.save_state(child)

        await scheduler._engine._check_health()

        events = await store.get_pending_events("parent-1", "sess")
        assert len(events) == 1
        assert events[0].event_type == SchedulerEventType.HEALTH_WARNING
        assert events[0].source_agent_id == "child-stuck"
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_health_check_no_warning_for_healthy_agent(self):
        """Agent with recent activity should NOT generate health warning."""
        scheduler = Scheduler(_fast_config())
        store = scheduler._store
        await scheduler.start()

        now = datetime.now(timezone.utc)
        parent = AgentState(
            id="parent-2", session_id="sess", status=AgentStateStatus.SLEEPING, task="p", parent_id=None,
        )
        child = AgentState(
            id="child-healthy", session_id="sess", status=AgentStateStatus.RUNNING, task="c",
            parent_id="parent-2", last_activity_at=now - timedelta(seconds=10),
        )
        await store.save_state(parent)
        await store.save_state(child)

        await scheduler._engine._check_health()

        events = await store.get_pending_events("parent-2", "sess")
        assert len(events) == 0
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_health_check_deduplicates_warnings(self):
        """Second health check should not emit duplicate warning if one already exists."""
        scheduler = Scheduler(_fast_config())
        store = scheduler._store
        await scheduler.start()

        now = datetime.now(timezone.utc)
        parent = AgentState(
            id="parent-3", session_id="sess", status=AgentStateStatus.SLEEPING, task="p", parent_id=None,
        )
        child = AgentState(
            id="child-dup", session_id="sess", status=AgentStateStatus.RUNNING, task="c",
            parent_id="parent-3", last_activity_at=now - timedelta(seconds=600),
        )
        await store.save_state(parent)
        await store.save_state(child)

        await scheduler._engine._check_health()
        await scheduler._engine._check_health()  # Second check — should not duplicate

        events = await store.get_pending_events("parent-3", "sess")
        assert len(events) == 1
        await scheduler.stop()


class TestSchedulerDebounce:
    @pytest.mark.asyncio
    async def test_process_pending_events_wakes_sleeping_agent(self):
        """A SLEEPING parent agent with pending events should be woken."""
        scheduler = Scheduler(_fast_config(event_debounce_min_count=1))
        store = scheduler._store

        parent = AgentState(
            id="parent-dbounce",
            session_id="sess",
            status=AgentStateStatus.SLEEPING,
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
        await scheduler._engine._process_pending_events()

        # Event should be deleted after dispatch
        remaining = await store.get_pending_events("parent-dbounce", "sess")
        assert len(remaining) == 0

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_process_pending_events_skips_non_sleeping(self):
        """Non-SLEEPING agents should NOT be woken via debounce."""
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

        before_dispatched = len(scheduler._coordinator.dispatched_state_ids)
        await scheduler._engine._process_pending_events()
        after_dispatched = len(scheduler._coordinator.dispatched_state_ids)

        # No new dispatch should have happened
        assert after_dispatched == before_dispatched
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
