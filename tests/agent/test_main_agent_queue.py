"""Regression tests for the unified MainAgent queue drain (Task 11)."""

import asyncio
import json
import time
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import (
    AgentConfig,
    AgentOptions,
    AgentSpec,
    MainAgent,
    MainAgentState,
)
from agiwo.agent.hooks import HookRegistry
from agiwo.llm.base import Model, StreamChunk


class _ScriptedModel(Model):
    """Returns scripted responses; optionally blocks until an event is set."""

    def __init__(
        self,
        responses: list[str | dict],
        *,
        start_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__(id="scripted", name="scripted", temperature=0.0)
        self._responses = iter(responses)
        self._start_event = start_event
        self.calls: list[list[dict]] = []

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del tools
        self.calls.append(messages)
        if self._start_event is not None:
            await self._start_event.wait()
        response = next(self._responses)
        if isinstance(response, dict):
            yield StreamChunk(tool_calls=[response])
        else:
            yield StreamChunk(content=response)
        yield StreamChunk(finish_reason="stop")


def _build_main_agent(*, model: Model) -> MainAgent:
    return MainAgent(
        session_id="session-queue",
        agent_id="agent-queue",
        spec=AgentSpec(
            config=AgentConfig(
                name="main-agent",
                description="queue drain test agent",
                system_prompt="Test prompt",
                options=AgentOptions(max_steps_per_run=10),
            )
        ),
        model=model,
        hooks=HookRegistry(),
    )


def _update_plan_call(call_id: str, status: str) -> dict:
    return {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {
            "name": "update_plan",
            "arguments": json.dumps(
                {"changes": [{"id": "inspect", "description": "d", "status": status}]}
            ),
        },
    }


async def _settle(main: MainAgent, timeout: float = 5.0) -> None:
    """Wait until no run is active and the pending queue is fully drained."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        completion = main._completion_task  # noqa: SLF001 - observe drain lifecycle
        if completion is None:
            if main.peek_pending() is None:
                return
            await asyncio.sleep(0.01)
            continue
        await asyncio.wait({completion})
    raise AssertionError("MainAgent did not settle: pending queue not drained")


async def _step_texts(main: MainAgent) -> list[str]:
    steps = await main.run_log_storage.list_step_views(
        session_id=main.session_id,
        agent_id=main.agent_id,
    )
    return [text for step in steps if (text := step.get_display_text())]


def _model_call_texts(model: "_ScriptedModel") -> list[str]:
    """All message contents the model has seen across calls.

    System-provenance messages (worker reports) do not surface as step
    views, so delivery into the Run context is asserted on model input.
    """
    return [
        str(message.get("content", ""))
        for call in model.calls
        for message in call
        if isinstance(message, dict)
    ]


@pytest.mark.asyncio
async def test_worker_report_behind_user_input_enters_run_context() -> None:
    """USER_INPUT ahead of WORKER_REPORT must not block report delivery."""
    start_event = asyncio.Event()
    model = _ScriptedModel(
        [
            _update_plan_call("plan-1", "pending"),
            _update_plan_call("plan-2", "completed"),
            "final report",
        ],
        start_event=start_event,
    )
    main = _build_main_agent(model=model)

    handle = await main.accept("first message")
    assert handle is not None
    assert main.state is MainAgentState.RUNNING

    await main.accept("mid-run message")
    await main.deliver_worker_report(handle.run_id, "worker finished subtask")

    start_event.set()
    await _settle(main)

    model_texts = _model_call_texts(model)
    assert any(
        "<worker-report>" in text and "worker finished subtask" in text
        for text in model_texts
    )
    assert main.peek_pending() is None
    assert main.state is MainAgentState.IDLE


@pytest.mark.asyncio
async def test_worker_report_after_completion_continues_run() -> None:
    """A report arriving while idle resumes the completed run with the report."""
    model = _ScriptedModel(
        ["first response", "second response", "third response"],
    )
    main = _build_main_agent(model=model)

    handle = await main.accept("first message")
    assert handle is not None
    await main.wait_current_run()
    await _settle(main)

    await main.deliver_worker_report(handle.run_id, "late worker report")
    await _settle(main)

    texts = await _step_texts(main)
    assert "second response" in texts, "idle report must continue the run"
    model_texts = _model_call_texts(model)
    assert any(
        "<worker-report>" in text and "late worker report" in text
        for text in model_texts
    )
    assert main.peek_pending() is None
    assert main.state is MainAgentState.IDLE


@pytest.mark.asyncio
async def test_enqueue_failure_during_run_close_does_not_lose_message() -> None:
    """enqueue_message=False (session closing) leaves the item queued for retry."""

    class _ClosingHandle:
        """Stub handle: active but its SessionRuntime is already closed."""

        run_id = "closing-run"
        is_active = True

        async def enqueue_message(self, message) -> bool:
            del message
            return False

    model = _ScriptedModel(["answered"])
    main = _build_main_agent(model=model)
    main._handle = _ClosingHandle()  # type: ignore[assignment]  # noqa: SLF001

    await main.accept("hello while closing")

    assert main.peek_pending() is not None, "message must stay queued for retry"

    main._handle = None  # noqa: SLF001 - closing run finished
    await main._drain()  # noqa: SLF001
    await _settle(main)

    texts = await _step_texts(main)
    assert "hello while closing" in texts
    assert "answered" in texts
    assert main.peek_pending() is None
    assert main.state is MainAgentState.IDLE
