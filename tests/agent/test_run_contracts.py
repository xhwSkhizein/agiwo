import asyncio
import inspect
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

import agiwo.agent.agent as agent_module
from agiwo.agent import Agent
from agiwo.agent import AgentConfig
from agiwo.agent import RunOutput
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.llm.base import Model, StreamChunk


class FixedResponseModel(Model):
    def __init__(
        self,
        response: str = "final answer",
        *,
        start_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__(id="fixed-model", name="fixed-model", temperature=0.0)
        self._response = response
        self._start_event = start_event

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        if self._start_event is not None:
            await self._start_event.wait()
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


class ToolCallDeltaModel(Model):
    def __init__(self) -> None:
        super().__init__(id="tool-call-delta-model", name="tool-call-delta-model")

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "weather_lookup",
                        "arguments": '{"city":"Par',
                    },
                }
            ]
        )
        yield StreamChunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "arguments": 'is"}',
                    },
                }
            ]
        )
        yield StreamChunk(finish_reason="tool_calls")


@pytest.mark.asyncio
async def test_handle_wait_is_idempotent() -> None:
    agent = Agent(
        AgentConfig(name="run-contract", description="run contract test"),
        model=FixedResponseModel(),
    )

    handle = agent.start("hello", session_id="run-contract-session")
    first = await handle.wait()
    second = await handle.wait()

    assert first.response == "final answer"
    assert second.response == first.response
    assert second.run_id == first.run_id


@pytest.mark.asyncio
async def test_handle_supports_multiple_stream_subscribers() -> None:
    start_event = asyncio.Event()
    agent = Agent(
        AgentConfig(name="stream-contract", description="stream contract test"),
        model=FixedResponseModel(start_event=start_event),
    )

    handle = agent.start("hello", session_id="stream-contract-session")
    stream_one = handle.stream()
    stream_two = handle.stream()

    collector_one = asyncio.create_task(_collect_stream(stream_one))
    collector_two = asyncio.create_task(_collect_stream(stream_two))
    await asyncio.sleep(0)
    start_event.set()

    events_one, events_two = await asyncio.gather(collector_one, collector_two)
    result = await handle.wait()

    assert result.response == "final answer"
    assert [event.type for event in events_one] == [event.type for event in events_two]


@pytest.mark.asyncio
async def test_trivial_root_run_keeps_steps_count_stable() -> None:
    agent = Agent(
        AgentConfig(name="metrics-contract", description="metrics contract test"),
        model=FixedResponseModel(),
    )

    result = await agent.run("hello", session_id="metrics-contract-session")

    assert result.response == "final answer"
    assert result.metrics.steps_count == 1


@pytest.mark.asyncio
async def test_stream_assistant_step_accumulates_chat_compatible_tool_call_deltas() -> (
    None
):
    agent = Agent(
        AgentConfig(name="tool-call-contract", description="tool call contract test"),
        model=ToolCallDeltaModel(),
    )

    await agent.run("weather?", session_id="tool-call-contract-session")
    steps = await agent.run_log_storage.list_step_views(
        session_id="tool-call-contract-session",
        agent_id=agent.id,
    )

    assistant_steps = [step for step in steps if step.role.value == "assistant"]
    assert assistant_steps[-1].tool_calls == [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "weather_lookup",
                "arguments": '{"city":"Paris"}',
            },
        }
    ]


async def _collect_stream(stream):
    events = []
    async for event in stream:
        events.append(event)
        if event.type in {"run_completed", "run_failed"}:
            break
    return events


@dataclass
class _DummyEvent:
    type: str


class _StubHandle:
    def __init__(self, *, wait_error: BaseException | None = None) -> None:
        self._wait_error = wait_error
        self.cancel_reason: str | None = None

    def stream(self) -> AsyncIterator[_DummyEvent]:
        async def _stream() -> AsyncIterator[_DummyEvent]:
            yield _DummyEvent(type="step_delta")

        return _stream()

    async def wait(self):
        if self._wait_error is not None:
            raise self._wait_error
        raise AssertionError("wait() should not be called on the completed path")

    def cancel(self, reason: str | None = None) -> None:
        self.cancel_reason = reason


@pytest.mark.asyncio
async def test_run_stream_close_propagates_non_cancelled_cleanup_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent(
        AgentConfig(name="run-stream-contract", description="run stream contract test"),
        model=FixedResponseModel(),
    )
    handle = _StubHandle(wait_error=RuntimeError("boom"))
    monkeypatch.setattr(agent, "start", lambda *_args, **_kwargs: handle)

    stream = agent.run_stream("hello")
    await anext(stream)

    with pytest.raises(RuntimeError, match="boom"):
        await stream.aclose()

    assert handle.cancel_reason == "run_stream consumer closed"


def test_run_child_contract_does_not_expose_child_id_parameter() -> None:
    assert "child_id" not in inspect.signature(Agent.run_child).parameters


@pytest.mark.asyncio
async def test_run_child_uses_agent_instance_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str | None] = {}

    async def fake_execute_run(
        user_input,
        *,
        context,
        model,
        system_prompt,
        tools,
        hooks,
        options,
        abort_signal,
        root_path,
    ) -> RunOutput:
        del (
            user_input,
            model,
            system_prompt,
            tools,
            hooks,
            options,
            abort_signal,
            root_path,
        )
        captured["agent_id"] = context.agent_id
        return RunOutput(response="child ok", run_id=context.run_id)

    monkeypatch.setattr(agent_module, "execute_run", fake_execute_run)

    agent = Agent(
        AgentConfig(name="child-contract", description="child contract test"),
        id="child-template",
        model=FixedResponseModel(),
    )
    session_runtime = SessionRuntime(
        session_id="sess-child",
        run_log_storage=InMemoryRunLogStorage(),
    )

    result = await agent.run_child(
        "hello",
        session_runtime=session_runtime,
        parent_run_id="parent-run",
        parent_depth=0,
        parent_user_id="user-1",
        parent_timeout_at=None,
        parent_metadata={},
    )

    assert result.response == "child ok"
    assert captured["agent_id"] == "child-template"
