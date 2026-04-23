"""Integration tests for the Scheduler class."""

import asyncio
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agiwo.agent import Agent
from agiwo.agent import AgentConfig, AgentOptions
from agiwo.agent import RunCompletedEvent, TerminationReason
from agiwo.agent import HookPhase, HookRegistry, transform
from agiwo.agent import ChannelContext, ContentPart, ContentType
from agiwo.agent import build_committed_step_entry
from agiwo.agent import RunFinished, RunStarted
from agiwo.agent.models.step import MessageRole, StepView, UserMessage
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.utils.abort_signal import AbortSignal
from agiwo.llm.base import Model, StreamChunk
from agiwo.scheduler._tick import propagate_signals
from agiwo.scheduler.commands import DispatchAction, DispatchReason, RouteStreamMode
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerRunResult,
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


async def _noop_memory_retrieve(payload: dict[str, object]) -> dict[str, object]:
    payload = dict(payload)
    payload["memories"] = []
    return payload


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
    hooks = HookRegistry(
        [
            transform(
                HookPhase.ASSEMBLE_CONTEXT,
                "noop_memory_retrieve",
                _noop_memory_retrieve,
            )
        ]
    )
    return Agent(
        AgentConfig(
            name=name,
            description="test" if name != "parent" else "parent agent",
            system_prompt=system_prompt,
            options=options or AgentOptions(),
        ),
        model=model,
        tools=tools,
        hooks=hooks,
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

            result = await scheduler.route_root_input(
                "Hello", agent=agent, stream_mode=RouteStreamMode.UNTIL_SETTLED
            )
            state_id = result.state_id
            registered = scheduler.get_registered_agent(state_id)
            assert registered is not None
            assert registered is not agent

            tool_names = {t.name for t in registered.tools}
            assert "spawn_child_agent" in tool_names
            assert "fork_child_agent" in tool_names
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

            await scheduler.route_root_input(
                "Hello", agent=agent, stream_mode=RouteStreamMode.UNTIL_SETTLED
            )

            tool_names = {t.name for t in agent.tools}
            assert "spawn_child_agent" not in tool_names
            assert "fork_child_agent" not in tool_names
            assert "sleep_and_wait" not in tool_names
            assert "query_spawned_agent" not in tool_names
            assert agent.options.enable_termination_summary is False

    @pytest.mark.asyncio
    async def test_submit_registers_agent(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Hello")])
            agent = _make_agent(name="test", model=model, id="test", tools=[])

            await scheduler.route_root_input(
                "Hello", agent=agent, stream_mode=RouteStreamMode.UNTIL_SETTLED
            )
            assert scheduler.get_registered_agent("test") is not None


class TestSchedulerSubmit:
    @pytest.mark.asyncio
    async def test_submit_creates_state(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Hello")])
            agent = _make_agent(name="test", model=model, id="test", tools=[])

            result = await scheduler.route_root_input(
                "Hello", agent=agent, stream_mode=RouteStreamMode.UNTIL_SETTLED
            )
            state_id = result.state_id
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

            result = await scheduler.route_root_input(
                "Hello",
                agent=agent,
                persistent=True,
                stream_mode=RouteStreamMode.UNTIL_SETTLED,
            )
            state_id = result.state_id
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

        with pytest.raises(RuntimeError, match="Failed to steer"):
            await scheduler.route_root_input(
                "Another task", agent=agent, stream_mode=RouteStreamMode.UNTIL_SETTLED
            )

        await scheduler.stop()


class TestSchedulerSimpleCompletion:
    @pytest.mark.asyncio
    async def test_run_simple_agent(self):
        """Agent completes without sleeping — simplest case."""
        model = MockModel([_simple_completion("The answer is 42")])
        agent = _make_agent(name="simple", model=model, id="simple", tools=[])

        async with Scheduler(_fast_config()) as scheduler:
            route_result = await scheduler.route_root_input(
                "What is the answer?",
                agent=agent,
                stream_mode=RouteStreamMode.RUN_END,
                timeout=_TEST_RUN_TIMEOUT,
            )
            # Consume stream to get result
            result = None
            async for item in route_result.stream:
                if isinstance(item, RunCompletedEvent):
                    result = item
                    break

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
        result = await scheduler.route_root_input(
            "root task", agent=parent, stream_mode=RouteStreamMode.UNTIL_SETTLED
        )
        state_id = result.state_id
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
        assert child_agent.run_log_storage is not parent.run_log_storage
        assert child_agent.options.enable_termination_summary is True
        # Check child inherited parent's system prompt (get_effective_system_prompt triggers build)
        child_prompt = await child_agent.get_effective_system_prompt()
        assert "Be helpful" in child_prompt
        child_tool_names = {t.name for t in child_agent.tools}
        assert "spawn_child_agent" not in child_tool_names
        assert "fork_child_agent" not in child_tool_names
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
        result = await scheduler.route_root_input(
            "root task", agent=parent, stream_mode=RouteStreamMode.UNTIL_SETTLED
        )
        state_id = result.state_id
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
        result = await scheduler.route_root_input(
            "root task", agent=parent, stream_mode=RouteStreamMode.UNTIL_SETTLED
        )
        state_id = result.state_id
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

    @pytest.mark.asyncio
    async def test_fork_child_inherits_all(self):  # noqa: C901, PLR0915
        """Fork mode: child inherits parent's full tool set and session steps."""
        scheduler = Scheduler()
        model = MockModel()
        parent_system_prompt = "You are a helpful assistant for testing fork."
        parent = _make_agent(
            name="parent-fork",
            model=model,
            id="parent-fork",
            tools=[],
            system_prompt=parent_system_prompt,
        )
        result = await scheduler.route_root_input(
            "root task", agent=parent, stream_mode=RouteStreamMode.UNTIL_SETTLED
        )
        state_id = result.state_id
        assert state_id == "parent-fork"

        # Get the runtime clone from scheduler (this is the actual agent that will spawn child)
        parent_runtime = scheduler.get_registered_agent("parent-fork")
        assert parent_runtime is not None

        # Update parent state in store to ensure session_id matches runtime
        parent_state = await scheduler._store.get_state("parent-fork")
        assert parent_state is not None
        # Root agent session_id is auto-generated UUID
        parent_session_id = parent_state.resolve_runtime_session_id()
        next_sequence = await parent_runtime.run_log_storage.get_max_sequence(
            parent_session_id
        )

        # Add some history steps to parent agent's session
        step1 = StepView(
            session_id=parent_session_id,
            run_id="run-1",
            sequence=next_sequence + 1,
            role=MessageRole.USER,
            agent_id="parent-fork",
            content="Initial user message",
            user_input=UserMessage.from_value("Initial user message"),
        )
        step2 = StepView(
            session_id=parent_session_id,
            run_id="run-1",
            sequence=next_sequence + 2,
            role=MessageRole.ASSISTANT,
            agent_id="parent-fork",
            content="Assistant response",
        )
        await parent_runtime.run_log_storage.append_entries(
            [
                build_committed_step_entry(step1),
                build_committed_step_entry(step2),
            ]
        )

        # Create child state with fork=True
        child_state = AgentState(
            id="fork-child-1",
            session_id="sess-fork",
            status=AgentStateStatus.PENDING,
            task="fork task",
            parent_id="parent-fork",
            config_overrides={"fork": True},  # Enable fork mode
        )
        await scheduler._store.save_state(child_state)

        # Create child agent via runner (this triggers _copy_session_steps_for_fork)
        child_agent = await scheduler._runner.create_child_agent(child_state)

        # ═══════════════════════════════════════════════════════════════════
        # Verify system prompt matches parent (critical for KV cache reuse)
        # ═══════════════════════════════════════════════════════════════════
        child_prompt = await child_agent.get_effective_system_prompt()
        parent_prompt = await parent_runtime.get_effective_system_prompt()

        mismatches = []
        if parent_system_prompt not in child_prompt:
            mismatches.append(
                f"SYSTEM PROMPT MISMATCH:\n"
                f"  Parent: {parent_prompt[:100]}...\n"
                f"  Child:  {child_prompt[:100]}..."
            )

        # ═══════════════════════════════════════════════════════════════════
        # Verify fork child inherits both child-spawn tools for cache-stable tool layout.
        # ═══════════════════════════════════════════════════════════════════
        parent_tool_names = {t.name for t in parent_runtime.tools}
        child_tool_names = {t.name for t in child_agent.tools}

        parent_has_spawn = "spawn_child_agent" in parent_tool_names
        child_has_spawn = "spawn_child_agent" in child_tool_names
        parent_has_fork = "fork_child_agent" in parent_tool_names
        child_has_fork = "fork_child_agent" in child_tool_names

        if parent_has_spawn and not child_has_spawn:
            mismatches.append(
                f"TOOLS MISMATCH: Parent has spawn_child_agent but child does not.\n"
                f"  Parent tools: {parent_tool_names}\n"
                f"  Child tools:  {child_tool_names}"
            )

        if parent_has_fork and not child_has_fork:
            mismatches.append(
                f"TOOLS MISMATCH: Parent has fork_child_agent but child does not.\n"
                f"  Parent tools: {parent_tool_names}\n"
                f"  Child tools:  {child_tool_names}"
            )

        # ═══════════════════════════════════════════════════════════════════
        # Verify session steps were copied from parent to child
        # ═══════════════════════════════════════════════════════════════════
        child_session_id = child_state.resolve_runtime_session_id()
        child_steps = await child_agent.run_log_storage.list_step_views(
            session_id=child_session_id,
            agent_id="fork-child-1",
        )

        parent_steps = await parent_runtime.run_log_storage.list_step_views(
            session_id=parent_session_id,
            agent_id="parent-fork",
        )

        if len(child_steps) != len(parent_steps):
            mismatches.append(
                f"STEPS COUNT MISMATCH:\n"
                f"  Parent steps: {len(parent_steps)}\n"
                f"  Child steps:  {len(child_steps)}"
            )
        elif len(parent_steps) > 0:
            # Verify content matches
            for i, (p_step, c_step) in enumerate(zip(parent_steps, child_steps)):
                if p_step.content != c_step.content:
                    mismatches.append(
                        f"STEP {i} CONTENT MISMATCH:\n"
                        f"  Parent: {p_step.content}\n"
                        f"  Child:  {c_step.content}"
                    )
                # Verify child step has its own session_id and agent_id
                if c_step.session_id != child_session_id:
                    mismatches.append(
                        f"STEP {i} SESSION_ID MISMATCH:\n"
                        f"  Expected: {child_session_id}\n"
                        f"  Got:      {c_step.session_id}"
                    )
                if c_step.agent_id != "fork-child-1":
                    mismatches.append(
                        f"STEP {i} AGENT_ID MISMATCH:\n"
                        f"  Expected: fork-child-1\n"
                        f"  Got:      {c_step.agent_id}"
                    )

        # ═══════════════════════════════════════════════════════════════════
        # Print all mismatches and fail if any
        # ═══════════════════════════════════════════════════════════════════
        if mismatches:
            print("\n" + "=" * 70)
            print("FORK CONSISTENCY CHECK FAILED")
            print("=" * 70)
            for m in mismatches:
                print(f"\n{m}")
            print("\n" + "=" * 70)
            raise AssertionError(
                "Fork consistency check failed. See printed mismatches above."
            )

        # Basic assertions for clarity
        assert child_has_spawn, "Fork child should inherit spawn_child_agent tool"
        assert child_has_fork, "Fork child should inherit fork_child_agent tool"
        assert len(child_steps) == 2, f"Expected 2 steps, got {len(child_steps)}"
        assert "sleep_and_wait" in child_tool_names
        assert "query_spawned_agent" in child_tool_names

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
            child_allowed_tools=[],
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

        await propagate_signals(scheduler._ctx)

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

        await propagate_signals(scheduler._ctx)

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
    async def test_wait_for_uses_last_run_result_termination_reason(self):
        scheduler = Scheduler(_fast_config())
        await scheduler._store.save_state(
            AgentState(
                id="root",
                session_id="sess",
                status=AgentStateStatus.FAILED,
                task="initial",
                last_run_result=SchedulerRunResult(
                    run_id="run-1",
                    termination_reason=TerminationReason.CANCELLED,
                    error="cancelled by user",
                ),
            )
        )

        result = await scheduler.wait_for("root", timeout=_TEST_RUN_TIMEOUT)

        assert result.termination_reason == TerminationReason.CANCELLED
        assert result.error == "cancelled by user"
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_wait_for_prefers_runtime_run_view_when_runtime_agent_exists(self):
        scheduler = Scheduler(_fast_config())
        storage = InMemoryRunLogStorage()

        class RuntimeAgentStub:
            def __init__(self, run_log_storage):
                self.run_log_storage = run_log_storage

            async def close(self):
                return None

        scheduler._rt.agents["root"] = RuntimeAgentStub(storage)
        await storage.append_entries(
            [
                RunStarted(
                    sequence=1,
                    session_id="sess",
                    run_id="run-1",
                    agent_id="root",
                    user_input="initial",
                ),
                RunFinished(
                    sequence=2,
                    session_id="sess",
                    run_id="run-1",
                    agent_id="root",
                    response="fresh response",
                    termination_reason=TerminationReason.COMPLETED,
                ),
            ]
        )
        await scheduler._store.save_state(
            AgentState(
                id="root",
                session_id="sess",
                status=AgentStateStatus.IDLE,
                task="initial",
                is_persistent=True,
                result_summary="stale summary",
                last_run_result=SchedulerRunResult(
                    run_id="run-old",
                    termination_reason=TerminationReason.COMPLETED,
                    summary="stale summary",
                ),
            )
        )

        result = await scheduler.wait_for("root", timeout=_TEST_RUN_TIMEOUT)

        assert result.run_id == "run-1"
        assert result.response == "fresh response"
        assert result.termination_reason == TerminationReason.COMPLETED
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_queued_root_preserves_mailbox_input_for_next_run(self):
        scheduler = Scheduler(_fast_config(event_debounce_min_count=1))
        model = EchoMessagesModel()
        agent = _make_agent(name="root", model=model, id="root", tools=[])
        prepared_agent = await agent.create_child_agent(
            child_id=agent.id,
            system_prompt_override=agent.config.system_prompt,
            child_allowed_tools=[],
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
    async def test_route_root_input_stream_survives_fast_completion_before_iteration(
        self,
    ):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Fast answer")])
            agent = _make_agent(name="stream-race", model=model, id="stream-race")

            result = await scheduler.route_root_input(
                "What happened?",
                agent=agent,
                session_id="sess-stream-race",
                timeout=0.1,
                persistent=True,
                stream_mode=RouteStreamMode.RUN_END,
            )

            await asyncio.sleep(0.05)
            items = [item async for item in result.stream]

        assert items
        assert items[0].type == "run_started"
        assert isinstance(items[-1], RunCompletedEvent)
        assert items[-1].response == "Fast answer"

    @pytest.mark.asyncio
    async def test_stream_new_root_run(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("Streamed answer")])
            agent = _make_agent(
                name="stream-root", model=model, id="stream-root", tools=[]
            )

            route_result = await scheduler.route_root_input(
                "What happened?",
                agent=agent,
                session_id="sess-stream",
                timeout=_TEST_RUN_TIMEOUT,
                stream_mode=RouteStreamMode.RUN_END,
            )
            items = [item async for item in route_result.stream]

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

            state_id = await scheduler._submit(
                agent,
                "first",
                session_id="sess-persist",
            )
            first = await scheduler.wait_for(state_id, timeout=_TEST_RUN_TIMEOUT)
            assert first.response == "First answer"

            route_result = await scheduler.route_root_input(
                "second",
                agent=agent,
                state_id=state_id,
                timeout=_TEST_RUN_TIMEOUT,
                stream_mode=RouteStreamMode.RUN_END,
            )
            items = [item async for item in route_result.stream]

        assert items
        assert isinstance(items[-1], RunCompletedEvent)
        assert items[-1].response == "Second answer"

    @pytest.mark.asyncio
    async def test_stream_rejects_second_subscriber_for_same_root(self):
        pytest.skip("Not applicable with route_root_input API")


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
            assert s.last_run_result is not None
            assert s.last_run_result.termination_reason == TerminationReason.CANCELLED
            assert s.last_run_result.error == "test cancel"

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_cancel_releases_wait_for_immediately(self):
        """wait_for() must return promptly with CANCELLED once cancel completes."""
        scheduler = Scheduler(_fast_config())
        store = scheduler._store

        root = AgentState(
            id="wait-cancel",
            session_id="sess",
            status=AgentStateStatus.RUNNING,
            task="root",
            parent_id=None,
        )
        await store.save_state(root)

        await scheduler.cancel("wait-cancel", "stop it")

        output = await scheduler.wait_for("wait-cancel", timeout=2.0)
        assert output.termination_reason == TerminationReason.CANCELLED
        assert output.error == "stop it"

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
            child_allowed_tools=[],
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
            payload={"user_input": UserMessage.to_storage_value("check this")},
            source_agent_id=None,
            created_at=datetime.now(timezone.utc),
        )
        await store.save_event(event)

        before_dispatched = len(scheduler._rt.dispatched)
        await scheduler.tick()
        after_dispatched = len(scheduler._rt.dispatched)

        # No new dispatch should have happened
        assert after_dispatched == before_dispatched

    @pytest.mark.asyncio
    async def test_wait_ready_acknowledges_consumed_child_events(self):
        """Waitset wake should clear child lifecycle events once consumed."""
        scheduler = Scheduler(_fast_config(event_debounce_min_count=1))
        store = scheduler._store
        model = MockModel([_simple_completion("woke")])
        prepared_agent = await _make_agent(
            name="parent", model=model, id="parent-wake-ready", tools=[]
        ).create_child_agent(
            child_id="parent-wake-ready",
            system_prompt_override="",
            child_allowed_tools=[],
            extra_tools=list(scheduler._scheduling_tools),
        )
        scheduler._rt.agents[prepared_agent.id] = prepared_agent

        parent = AgentState(
            id="parent-wake-ready",
            session_id="sess",
            status=AgentStateStatus.WAITING,
            task="parent",
            parent_id=None,
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=("child-x",),
                completed_ids=("child-x",),
            ),
        )
        await store.save_state(parent)
        await store.save_event(
            PendingEvent(
                id="evt-ready",
                target_agent_id="parent-wake-ready",
                session_id="sess",
                event_type=SchedulerEventType.CHILD_COMPLETED,
                payload={"result": "done", "child_agent_id": "child-x"},
                source_agent_id="child-x",
                created_at=datetime.now(timezone.utc),
            )
        )

        await scheduler.tick()
        await asyncio.sleep(_TEST_SETTLE_WAIT)

        remaining = await store.list_events(
            target_agent_id="parent-wake-ready",
            session_id="sess",
        )
        assert remaining == []

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_wait_timeout_acknowledges_only_waitset_child_events(self):
        """Timeout wake should clear only child lifecycle events for the active waitset."""
        scheduler = Scheduler(_fast_config(event_debounce_min_count=1))
        store = scheduler._store
        model = MockModel([_simple_completion("timed out")])
        prepared_agent = await _make_agent(
            name="parent", model=model, id="parent-timeout", tools=[]
        ).create_child_agent(
            child_id="parent-timeout",
            system_prompt_override="",
            child_allowed_tools=[],
            extra_tools=list(scheduler._scheduling_tools),
        )
        scheduler._rt.agents[prepared_agent.id] = prepared_agent

        parent = AgentState(
            id="parent-timeout",
            session_id="sess",
            status=AgentStateStatus.WAITING,
            task="parent",
            parent_id=None,
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=("child-x",),
                timeout_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            ),
        )
        await store.save_state(parent)
        await store.save_event(
            PendingEvent(
                id="evt-timeout",
                target_agent_id="parent-timeout",
                session_id="sess",
                event_type=SchedulerEventType.CHILD_COMPLETED,
                payload={"result": "done", "child_agent_id": "child-x"},
                source_agent_id="child-x",
                created_at=datetime.now(timezone.utc),
            )
        )
        await store.save_event(
            PendingEvent(
                id="evt-other-child",
                target_agent_id="parent-timeout",
                session_id="sess",
                event_type=SchedulerEventType.CHILD_COMPLETED,
                payload={"result": "other", "child_agent_id": "child-y"},
                source_agent_id="child-y",
                created_at=datetime.now(timezone.utc),
            )
        )
        await store.save_event(
            PendingEvent(
                id="evt-user-hint",
                target_agent_id="parent-timeout",
                session_id="sess",
                event_type=SchedulerEventType.USER_HINT,
                payload={"user_input": UserMessage.to_storage_value("still here")},
                source_agent_id=None,
                created_at=datetime.now(timezone.utc),
            )
        )

        await scheduler.tick()
        await asyncio.sleep(_TEST_SETTLE_WAIT)

        remaining = await store.list_events(
            target_agent_id="parent-timeout",
            session_id="sess",
        )
        assert [event.id for event in remaining] == ["evt-other-child", "evt-user-hint"]

        await scheduler.stop()


class TestSchedulerShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_running_persistent_root_requeues_summary(self):
        async with Scheduler(_fast_config()) as scheduler:
            model = EchoMessagesModel(delay_seconds=0.2)
            agent = _make_agent(name="root", model=model, id="root", tools=[])

            result = await scheduler.route_root_input(
                "initial",
                agent=agent,
                session_id="sess",
                stream_mode=RouteStreamMode.UNTIL_SETTLED,
            )
            state_id = result.state_id

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

            result = await scheduler.route_root_input(
                "hello",
                agent=agent,
                session_id="sess-multiwait",
                stream_mode=RouteStreamMode.UNTIL_SETTLED,
            )
            state_id = result.state_id

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


class TestSchedulerSteer:
    @pytest.mark.asyncio
    async def test_steer_urgent_writes_urgent_pending_event(self):
        """steer(..., urgent=True) on WAITING state must set PendingEvent.urgent."""
        scheduler = Scheduler(_fast_config())
        await scheduler._store.save_state(
            AgentState(
                id="waiting-root",
                session_id="sess",
                status=AgentStateStatus.WAITING,
                task="root",
                parent_id=None,
            )
        )

        ok = await scheduler.steer("waiting-root", "please continue", urgent=True)
        assert ok is True

        events = await scheduler._store.list_events(
            target_agent_id="waiting-root",
            session_id="sess",
        )
        assert len(events) == 1
        event = events[0]
        assert event.event_type == SchedulerEventType.USER_HINT
        assert event.urgent is True
        # Payload uses the new structured ``user_input`` key only.
        assert "hint" not in event.payload
        decoded = UserMessage.from_storage_value(event.payload["user_input"])
        assert UserMessage.from_value(decoded).extract_text() == "please continue"

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_steer_defaults_to_non_urgent(self):
        scheduler = Scheduler(_fast_config())
        await scheduler._store.save_state(
            AgentState(
                id="waiting-root",
                session_id="sess",
                status=AgentStateStatus.WAITING,
                task="root",
                parent_id=None,
            )
        )

        await scheduler.steer("waiting-root", "just a hint")

        events = await scheduler._store.list_events(
            target_agent_id="waiting-root",
            session_id="sess",
        )
        assert len(events) == 1
        assert events[0].urgent is False

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_steer_preserves_structured_user_input(self):
        """Multimodal UserMessage must round-trip through the PendingEvent payload."""
        scheduler = Scheduler(_fast_config())
        await scheduler._store.save_state(
            AgentState(
                id="waiting-root",
                session_id="sess",
                status=AgentStateStatus.WAITING,
                task="root",
                parent_id=None,
            )
        )

        user_input = UserMessage(
            content=[
                ContentPart(type=ContentType.TEXT, text="look at this image"),
                ContentPart(
                    type=ContentType.IMAGE,
                    url="/tmp/image.png",
                    mime_type="image/png",
                    metadata={"name": "image.png"},
                ),
            ],
            context=ChannelContext(source="feishu", metadata={"chat_id": "oc-1"}),
        )

        await scheduler.steer("waiting-root", user_input, urgent=True)

        events = await scheduler._store.list_events(
            target_agent_id="waiting-root",
            session_id="sess",
        )
        assert len(events) == 1
        payload_user_input = events[0].payload["user_input"]
        decoded = UserMessage.from_value(
            UserMessage.from_storage_value(payload_user_input)
        )
        # Text, image, and ChannelContext must all survive persistence.
        assert decoded.extract_text() == "look at this image"
        image_parts = [p for p in decoded.content if p.type == ContentType.IMAGE]
        assert len(image_parts) == 1
        assert image_parts[0].url == "/tmp/image.png"
        assert decoded.context is not None
        assert decoded.context.source == "feishu"
        assert decoded.context.metadata["chat_id"] == "oc-1"

        await scheduler.stop()


class TestSchedulerRuntimeAgentReuse:
    @pytest.mark.asyncio
    async def test_persistent_root_reuses_runtime_agent_across_turns(self):
        """Persistent roots must preserve their run_log_storage across turns.

        This guards against the earlier bug where ``submit`` / ``enqueue_input``
        cloned the ``Agent`` on every call and, under the default in-memory
        storage backend, silently lost prior step history.
        """
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel(
                [
                    _simple_completion("first response"),
                    _simple_completion("second response"),
                ]
            )
            canonical = _make_agent(
                name="persist",
                model=model,
                id="persist",
                tools=[],
            )

            result = await scheduler.route_root_input(
                "first turn",
                agent=canonical,
                session_id="sess-persist",
                stream_mode=RouteStreamMode.UNTIL_SETTLED,
            )
            state_id = result.state_id
            first_output = await scheduler.wait_for(state_id, timeout=_TEST_RUN_TIMEOUT)
            assert first_output.termination_reason == TerminationReason.COMPLETED
            runtime_after_first = scheduler.get_registered_agent(state_id)
            assert runtime_after_first is not None

            storage_after_first = runtime_after_first.run_log_storage
            steps_after_first = await storage_after_first.list_step_views(
                session_id="sess-persist",
                agent_id=state_id,
            )

            await scheduler.enqueue_input(
                state_id,
                "second turn",
                agent=canonical,
            )
            second_output = await scheduler.wait_for(
                state_id, timeout=_TEST_RUN_TIMEOUT
            )
            assert second_output.termination_reason == TerminationReason.COMPLETED

            runtime_after_second = scheduler.get_registered_agent(state_id)
            # Strict reuse: same Python runtime agent, same storage instance.
            assert runtime_after_second is runtime_after_first
            assert runtime_after_second.run_log_storage is storage_after_first

            steps_after_second = await storage_after_first.list_step_views(
                session_id="sess-persist",
                agent_id=state_id,
            )
            assert len(steps_after_second) > len(steps_after_first)

    @pytest.mark.asyncio
    async def test_non_persistent_root_closed_after_terminal(self):
        """Non-persistent roots must be popped + closed after reaching terminal."""
        async with Scheduler(_fast_config()) as scheduler:
            model = MockModel([_simple_completion("ok")])
            agent = _make_agent(name="simple", model=model, id="simple", tools=[])

            state_id = await scheduler._submit(agent, "hello")
            output = await scheduler.wait_for(state_id, timeout=_TEST_RUN_TIMEOUT)
            assert output.termination_reason == TerminationReason.COMPLETED

            # Allow _cleanup_after_run to run.
            await asyncio.sleep(_TEST_SETTLE_WAIT)

            assert scheduler.get_registered_agent(state_id) is None
            assert state_id not in scheduler._rt.agents
            assert state_id not in scheduler._rt.canonical_agents
            assert state_id not in scheduler._rt.execution_handles

    @pytest.mark.asyncio
    async def test_rebind_replaces_runtime_agent(self):
        """Passing a different canonical Agent to rebind must refresh the runtime."""
        async with Scheduler(_fast_config()) as scheduler:
            model_one = MockModel([_simple_completion("first")])
            first_canonical = _make_agent(
                name="persist", model=model_one, id="persist", tools=[]
            )
            result = await scheduler.route_root_input(
                "first",
                agent=first_canonical,
                session_id="sess-rebind",
                stream_mode=RouteStreamMode.UNTIL_SETTLED,
            )
            state_id = result.state_id

            await scheduler.wait_for(state_id, timeout=_TEST_RUN_TIMEOUT)

            first_runtime = scheduler.get_registered_agent(state_id)

            model_two = MockModel([_simple_completion("second")])
            second_canonical = _make_agent(
                name="persist", model=model_two, id="persist", tools=[]
            )
            rebound = await scheduler.rebind_agent(state_id, second_canonical)
            assert rebound is True

            new_runtime = scheduler.get_registered_agent(state_id)
            assert new_runtime is not None
            assert new_runtime is not first_runtime
