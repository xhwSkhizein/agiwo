"""Runtime contract tests for the refactored agent execution pipeline."""

import asyncio
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from agiwo.agent import Agent, AgentConfig, AgentHooks
from agiwo.agent.inner.run_recorder import RunRecorder
from agiwo.agent.inner.run_state import RunState
from agiwo.agent.options import AgentOptions, TraceStorageConfig
from agiwo.agent.runtime import MessageRole, Run, RunStatus, StepMetrics
from agiwo.agent.storage.base import InMemoryRunStepStorage, RunStepStorage
from agiwo.agent.storage.mongo import MongoRunStepStorage
from agiwo.agent.storage.sqlite import SQLiteRunStepStorage
from agiwo.llm.base import Model, StreamChunk
from tests.utils.agent_context import build_agent_context


class FixedResponseModel(Model):
    """Simple streaming model used to validate Agent runner contracts."""

    def __init__(self, response: str = "final answer") -> None:
        super().__init__(id="fixed-model", name="fixed-model", temperature=0.0)
        self._response = response

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(content=self._response)
        yield StreamChunk(
            usage={
                "input_tokens": 5,
                "output_tokens": 3,
                "total_tokens": 8,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
            },
            finish_reason="stop",
        )


class BlockingResponseModel(Model):
    """Streaming model that waits on a gate so handle contracts can be inspected."""

    def __init__(self, gate: asyncio.Event) -> None:
        super().__init__(id="blocking-model", name="blocking-model", temperature=0.0)
        self._gate = gate

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        await self._gate.wait()
        yield StreamChunk(content="released")
        yield StreamChunk(finish_reason="stop")


class CancellableStreamingModel(Model):
    """Streaming model that stays live until the run is aborted."""

    def __init__(self) -> None:
        super().__init__(id="cancellable-model", name="cancellable-model", temperature=0.0)

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        while True:
            await asyncio.sleep(0.01)
            yield StreamChunk(content="tick")


def _storage_kinds() -> list[str]:
    kinds = ["memory", "sqlite"]
    if os.environ.get("AGIWO_TEST_MONGO_URI"):
        kinds.append("mongodb")
    return kinds


@asynccontextmanager
async def _run_step_storage(kind: str) -> AsyncIterator[RunStepStorage]:
    if kind == "memory":
        yield InMemoryRunStepStorage()
        return

    if kind == "sqlite":
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = SQLiteRunStepStorage(db_path=os.path.join(tmpdir, "steps.db"))
            try:
                yield storage
            finally:
                await storage.close()
        return

    mongo_uri = os.environ.get("AGIWO_TEST_MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("AGIWO_TEST_MONGO_URI is required for mongodb storage tests")
    storage = MongoRunStepStorage(
        uri=mongo_uri,
        db_name=f"agiwo_test_{uuid4().hex}",
    )
    try:
        yield storage
    finally:
        if storage.client is not None:
            await storage.client.drop_database(storage.db_name)
        await storage.close()


@pytest.mark.asyncio
async def test_sequence_allocation_is_shared_between_root_and_child_contexts() -> None:
    storage = InMemoryRunStepStorage()
    parent_context = build_agent_context(
        session_id="sequence-session",
        run_id="root-run",
        agent_id="root-agent",
        agent_name="root-agent",
        run_step_storage=storage,
    )
    child_context = parent_context.new_child(
        agent_id="child-agent",
        agent_name="child-agent",
    )

    parent_recorder = RunRecorder(
        context=parent_context,
        hooks=AgentHooks(),
        step_observers=[],
    )
    child_recorder = RunRecorder(
        context=child_context,
        hooks=AgentHooks(),
        step_observers=[],
    )

    root_user = await parent_recorder.create_user_step(user_input="root task")
    child_user = await child_recorder.create_user_step(user_input="child task")
    root_assistant = await parent_recorder.create_assistant_step()

    assert [root_user.sequence, child_user.sequence, root_assistant.sequence] == [1, 2, 3]


@pytest.mark.asyncio
async def test_run_recorder_tracks_state_before_hooks_and_observers() -> None:
    context = build_agent_context(session_id="step-session", run_id="step-run")
    seen: list[tuple[str, int, int, str]] = []

    async def on_step(step) -> None:
        seen.append(("hook", state.steps_count, len(state.messages), step.id))

    async def observer(step) -> None:
        seen.append(("observer", state.steps_count, len(state.messages), step.id))

    state = RunState(
        context=context,
        config=AgentOptions(),
        messages=[],
    )
    recorder = RunRecorder(
        context=context,
        hooks=AgentHooks(on_step=on_step),
        step_observers=[observer],
    ).attach_state(state)

    user_step = await recorder.create_user_step(user_input="hello")
    await recorder.commit_step(user_step, append_message=False)

    assistant_step = await recorder.create_assistant_step()
    assistant_step.content = "done"
    assistant_step.metrics = StepMetrics(total_tokens=3, token_cost=0.25)
    await recorder.commit_step(assistant_step, append_message=True)

    assert state.steps_count == 2
    assert state.token_cost == 0.25
    assert len(state.messages) == 1
    assert state.messages[0]["role"] == MessageRole.ASSISTANT.value
    assert seen == [
        ("hook", 1, 0, user_step.id),
        ("observer", 1, 0, user_step.id),
        ("hook", 2, 1, assistant_step.id),
        ("observer", 2, 1, assistant_step.id),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("storage_kind", _storage_kinds())
async def test_delete_run_only_removes_target_run_steps(storage_kind: str) -> None:
    async with _run_step_storage(storage_kind) as storage:
        session_id = f"delete-session-{storage_kind}"
        run_one = Run(
            id="run-1",
            agent_id="agent-1",
            session_id=session_id,
            user_input="first",
            status=RunStatus.RUNNING,
        )
        run_two = Run(
            id="run-2",
            agent_id="agent-2",
            session_id=session_id,
            user_input="second",
            status=RunStatus.RUNNING,
        )
        await storage.save_run(run_one)
        await storage.save_run(run_two)

        context_one = build_agent_context(
            session_id=session_id,
            run_id="run-1",
            agent_id="agent-1",
            agent_name="agent-1",
            run_step_storage=storage,
        )
        context_two = build_agent_context(
            session_id=session_id,
            run_id="run-2",
            agent_id="agent-2",
            agent_name="agent-2",
            run_step_storage=storage,
        )
        step_one = await RunRecorder(
            context=context_one,
            hooks=AgentHooks(),
            step_observers=[],
        ).create_user_step(user_input="first")
        step_two = await RunRecorder(
            context=context_two,
            hooks=AgentHooks(),
            step_observers=[],
        ).create_user_step(user_input="second")
        await storage.save_step(step_one)
        await storage.save_step(step_two)

        await storage.delete_run("run-1")

        remaining_runs = await storage.list_runs(session_id=session_id)
        remaining_steps = await storage.get_steps(session_id=session_id)

        assert [run.id for run in remaining_runs] == ["run-2"]
        assert [step.run_id for step in remaining_steps] == ["run-2"]
        assert await storage.get_run("run-1") is None


async def _execute_agent(mode: str) -> dict[str, object]:
    steps: list[tuple[str, int]] = []

    async def on_step(step) -> None:
        steps.append((step.role.value, step.sequence))

    async def retrieve_memories(user_input, context) -> list:
        del user_input, context
        return []

    agent = Agent(
        AgentConfig(
            name=f"runner-{mode}",
            description="runner contract test",
            options=AgentOptions(enable_termination_summary=False),
        ),
        model=FixedResponseModel(),
        hooks=AgentHooks(
            on_step=on_step,
            on_memory_retrieve=retrieve_memories,
        ),
    )
    session_id = f"runner-session-{mode}"

    try:
        if mode == "run":
            output = await agent.run("hello", session_id=session_id)
            assert output.response == "final answer"
        else:
            streamed_events = [event async for event in agent.run_stream("hello", session_id=session_id)]
            assert streamed_events

        runs = await agent.run_step_storage.list_runs(session_id=session_id)
        persisted_steps = await agent.run_step_storage.get_steps(session_id=session_id)
    finally:
        await agent.close()

    run = runs[0]
    return {
        "run_status": run.status.value,
        "response": run.response_content,
        "step_roles": [step.role.value for step in persisted_steps],
        "hook_steps": steps,
    }


@pytest.mark.asyncio
async def test_run_and_run_stream_share_storage_and_hook_contracts() -> None:
    run_result = await _execute_agent("run")
    stream_result = await _execute_agent("stream")

    assert run_result == stream_result
    assert run_result["run_status"] == RunStatus.COMPLETED.value
    assert run_result["response"] == "final answer"
    assert run_result["step_roles"] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_start_wait_is_idempotent_and_matches_run() -> None:
    async def retrieve_memories(*_args, **_kwargs) -> list:
        return []

    agent = Agent(
        AgentConfig(
            name="handle-run",
            description="handle contract test",
            options=AgentOptions(enable_termination_summary=False),
        ),
        model=FixedResponseModel(),
        hooks=AgentHooks(on_memory_retrieve=retrieve_memories),
    )
    try:
        run_output = await agent.run("hello", session_id="handle-run-session")
        handle = agent.start("hello", session_id="handle-start-session")
        first = await handle.wait()
        second = await handle.wait()
    finally:
        await agent.close()

    assert first is second
    assert first.response == "final answer"
    assert run_output.response == first.response
    assert run_output.termination_reason == first.termination_reason


@pytest.mark.asyncio
async def test_handle_supports_multiple_live_stream_subscribers() -> None:
    async def retrieve_memories(*_args, **_kwargs) -> list:
        return []

    agent = Agent(
        AgentConfig(
            name="multi-stream",
            description="multi subscriber test",
            options=AgentOptions(enable_termination_summary=False),
        ),
        model=FixedResponseModel(),
        hooks=AgentHooks(on_memory_retrieve=retrieve_memories),
    )
    try:
        handle = agent.start("hello", session_id="multi-stream-session")
        stream_one = handle.stream()
        stream_two = handle.stream()

        async def _collect(stream) -> list[str]:
            return [item.type async for item in stream]

        first_task = asyncio.create_task(_collect(stream_one))
        second_task = asyncio.create_task(_collect(stream_two))
        result = await handle.wait()
        first_items, second_items = await asyncio.gather(first_task, second_task)
    finally:
        await agent.close()

    assert result.response == "final answer"
    assert first_items == second_items
    assert first_items[0] == "run_started"
    assert first_items[-1] == "run_completed"


@pytest.mark.asyncio
async def test_root_handles_keep_steering_queues_isolated() -> None:
    gate = asyncio.Event()

    async def retrieve_memories(*_args, **_kwargs) -> list:
        return []

    agent = Agent(
        AgentConfig(
            name="steer-isolation",
            description="steer isolation test",
            options=AgentOptions(enable_termination_summary=False),
        ),
        model=BlockingResponseModel(gate),
        hooks=AgentHooks(on_memory_retrieve=retrieve_memories),
    )
    try:
        handle_one = agent.start("first", session_id="steer-1")
        handle_two = agent.start("second", session_id="steer-2")
        await asyncio.sleep(0)
        assert await handle_one.steer("only-first") is True
        queued = handle_one._session_runtime.steering_queue.get_nowait()
        assert queued == "only-first"
        assert handle_two._session_runtime.steering_queue.empty()
        gate.set()
        await asyncio.gather(handle_one.wait(), handle_two.wait())
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_handle_surface_does_not_expose_internal_runtime_state() -> None:
    async def retrieve_memories(*_args, **_kwargs) -> list:
        return []

    agent = Agent(
        AgentConfig(
            name="handle-surface",
            description="surface contract test",
            options=AgentOptions(enable_termination_summary=False),
        ),
        model=FixedResponseModel(),
        hooks=AgentHooks(on_memory_retrieve=retrieve_memories),
    )
    try:
        handle = agent.start("hello", session_id="handle-surface-session")
        await handle.wait()
    finally:
        await agent.close()

    assert not hasattr(handle, "context")
    assert not hasattr(handle, "session_runtime")
    assert not hasattr(handle, "abort_signal")
    assert not hasattr(handle, "done")


@pytest.mark.asyncio
async def test_trace_id_is_owned_by_session_runtime_and_visible_in_context() -> None:
    seen_trace_ids: list[str | None] = []

    async def before_run(user_input, context):
        del user_input
        seen_trace_ids.append(context.trace_id)
        return None

    async def retrieve_memories(*_args, **_kwargs) -> list:
        return []

    agent = Agent(
        AgentConfig(
            name="trace-visible",
            description="trace visibility test",
            options=AgentOptions(
                enable_termination_summary=False,
                trace_storage=TraceStorageConfig(storage_type="memory"),
            ),
        ),
        model=FixedResponseModel(),
        hooks=AgentHooks(
            on_before_run=before_run,
            on_memory_retrieve=retrieve_memories,
        ),
    )
    session_id = "trace-visible-session"
    try:
        result = await agent.run("hello", session_id=session_id)
        runs = await agent.run_step_storage.list_runs(session_id=session_id)
    finally:
        await agent.close()

    assert result.response == "final answer"
    assert seen_trace_ids
    assert seen_trace_ids[0] is not None
    assert runs[0].trace_id == seen_trace_ids[0]


@pytest.mark.asyncio
async def test_derive_child_spec_keeps_definition_only() -> None:
    agent = Agent(
        AgentConfig(
            name="child-spec",
            description="child spec contract test",
            options=AgentOptions(enable_termination_summary=False),
        ),
        model=FixedResponseModel(),
    )
    try:
        child_spec = agent.derive_child_spec(
            child_id="child-spec-1",
            instruction="Focus on sub-problem only.",
            exclude_tool_names={"bash"},
            metadata_overrides={"source": "test"},
        )
    finally:
        await agent.close()

    assert child_spec.agent_id == "child-spec-1"
    assert child_spec.metadata_overrides == {"source": "test"}
    assert not hasattr(child_spec, "run_step_storage")
    assert not hasattr(child_spec, "session_storage")
    assert not hasattr(child_spec, "model")
    assert not hasattr(child_spec, "hooks")
    assert not hasattr(child_spec, "tools")
    assert not hasattr(child_spec, "prompt_runtime")


@pytest.mark.asyncio
async def test_agent_close_cancels_active_handles_and_blocks_new_starts() -> None:
    async def retrieve_memories(*_args, **_kwargs) -> list:
        return []

    agent = Agent(
        AgentConfig(
            name="close-owner",
            description="close contract test",
            options=AgentOptions(enable_termination_summary=False),
        ),
        model=CancellableStreamingModel(),
        hooks=AgentHooks(on_memory_retrieve=retrieve_memories),
    )

    handle_one = agent.start("first", session_id="close-1")
    handle_two = agent.start("second", session_id="close-2")
    await agent.close()

    result_one, result_two = await asyncio.gather(handle_one.wait(), handle_two.wait())

    assert result_one.termination_reason.value == "cancelled"
    assert result_two.termination_reason.value == "cancelled"
    with pytest.raises(RuntimeError, match="agent_closed"):
        agent.start("after-close")
