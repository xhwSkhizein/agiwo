"""Regression tests for MainAgent lifecycle hardening (Task 12)."""

import asyncio
import time
from collections.abc import AsyncIterator

import pytest
from structlog.testing import capture_logs

from agiwo.agent import (
    AgentConfig,
    AgentOptions,
    AgentSpec,
    MainAgent,
    MainAgentState,
    QueueItem,
    QueueItemKind,
)
from agiwo.agent.completion_gates import CompletionGates
from agiwo.agent.hooks import HookRegistry
from agiwo.agent.models.log import RunStarted
from agiwo.llm.base import Model, StreamChunk


class _BlockingModel(Model):
    """Emits one response, optionally waiting for an external event first."""

    def __init__(
        self, response: str = "ok", *, start_event: asyncio.Event | None = None
    ) -> None:
        super().__init__(id="blocking-model", name="blocking-model", temperature=0.0)
        self._response = response
        self._start_event = start_event

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        if self._start_event is not None:
            await self._start_event.wait()
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


def _build_main_agent(*, model: Model) -> MainAgent:
    return MainAgent(
        session_id="session-cancel",
        agent_id="agent-cancel",
        spec=AgentSpec(
            config=AgentConfig(
                name="main-agent",
                description="cancel test agent",
                system_prompt="Test prompt",
                options=AgentOptions(max_steps_per_run=5),
            )
        ),
        model=model,
        hooks=HookRegistry(),
    )


async def _count_run_started(main: MainAgent) -> int:
    entries = await main.run_log_storage.list_entries(session_id=main.session_id)
    return sum(1 for entry in entries if isinstance(entry, RunStarted))


@pytest.mark.asyncio
async def test_cancel_discards_pending_worker_report_without_revival() -> None:
    """cancel() must not let the completion-finally drain start a new Run."""
    start_event = asyncio.Event()
    main = _build_main_agent(model=_BlockingModel(start_event=start_event))

    handle = await main.accept("work")
    assert handle is not None
    # Stage a report without draining, simulating a report that lands while
    # the Run is mid-flight and has not been delivered yet.
    main.enqueue(
        QueueItem(
            kind=QueueItemKind.WORKER_REPORT,
            text="late worker report",
            run_id=handle.run_id,
            created_at=time.time(),
        )
    )

    cancel_task = asyncio.create_task(main.cancel("test cancel"))
    await asyncio.sleep(0.05)
    start_event.set()
    with capture_logs() as logs:
        await cancel_task

    assert main.state is MainAgentState.IDLE
    assert main.peek_pending() is None
    assert await _count_run_started(main) == 1, "cancel must not revive a run"
    assert any(
        entry.get("event") == "worker_report_discarded_on_cancel" for entry in logs
    )

    handle = await main.accept("after cancel")
    assert handle is not None
    output = await handle.wait()
    assert output.error is None
    assert main.state is MainAgentState.IDLE


@pytest.mark.asyncio
async def test_failed_run_is_logged_and_completion_task_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing Run must not escape the completion task unlogged/unretrieved."""

    async def _exploding_evaluate(self, **kwargs):  # noqa: ANN001, ANN003
        del self, kwargs
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(CompletionGates, "evaluate", _exploding_evaluate)
    main = _build_main_agent(model=_BlockingModel())

    loop = asyncio.get_running_loop()
    loop_errors: list[dict] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    try:
        with capture_logs() as logs:
            handle = await main.accept("boom")
            assert handle is not None
            with pytest.raises(RuntimeError, match="gate exploded"):
                await handle.wait()

        assert main.state is MainAgentState.IDLE
        assert any(
            entry.get("event") == "main_agent_run_failed"
            and entry.get("run_id") == handle.run_id
            for entry in logs
        )
    finally:
        loop.set_exception_handler(previous_handler)
    await asyncio.sleep(0)
    assert not loop_errors, loop_errors
