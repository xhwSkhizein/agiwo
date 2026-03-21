import asyncio
import time

import pytest

from agiwo.agent import AgentHooks, AgentOptions, TerminationReason
from agiwo.agent.engine.recorder import RunRecorder
from agiwo.agent.engine.state import RunState
from agiwo.agent.runtime import Run, RunCompletedEvent, RunFailedEvent, RunStatus
from tests.utils.agent_context import build_agent_context


async def _collect_events(context, expected_count: int):
    events = []

    async def _consume():
        async for event in context.session_runtime.subscribe():
            events.append(event)
            if len(events) >= expected_count:
                break

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0)
    return events, task


def _make_run(context) -> Run:
    run = Run(
        id=context.run_id,
        agent_id=context.agent_id,
        session_id=context.session_id,
        user_input="hello",
        status=RunStatus.RUNNING,
    )
    run.metrics.start_at = time.time()
    return run


@pytest.mark.asyncio
async def test_complete_run_publishes_terminal_event_after_run_started() -> None:
    context = build_agent_context(session_id="recorder-complete", run_id="run-1")
    state = RunState(context=context, config=AgentOptions(), messages=[])
    recorder = RunRecorder(
        context=context,
        hooks=AgentHooks(),
        step_observers=[],
        state=state,
    )
    events, task = await _collect_events(context, expected_count=2)

    await recorder.start_run(_make_run(context))
    await recorder.complete_run(state.build_output())
    await asyncio.wait_for(task, timeout=1)

    assert [type(event).__name__ for event in events] == [
        "RunStartedEvent",
        "RunCompletedEvent",
    ]
    assert isinstance(events[-1], RunCompletedEvent)


@pytest.mark.asyncio
async def test_fail_run_publishes_failed_event() -> None:
    context = build_agent_context(session_id="recorder-fail", run_id="run-1")
    state = RunState(context=context, config=AgentOptions(), messages=[])
    recorder = RunRecorder(
        context=context,
        hooks=AgentHooks(),
        step_observers=[],
        state=state,
    )
    events, task = await _collect_events(context, expected_count=2)

    await recorder.start_run(_make_run(context))
    await recorder.fail_run(RuntimeError("boom"))
    await asyncio.wait_for(task, timeout=1)

    assert [type(event).__name__ for event in events] == [
        "RunStartedEvent",
        "RunFailedEvent",
    ]
    assert isinstance(events[-1], RunFailedEvent)


@pytest.mark.asyncio
async def test_complete_run_marks_cancelled_status_from_termination_reason() -> None:
    context = build_agent_context(session_id="recorder-cancel", run_id="run-1")
    state = RunState(context=context, config=AgentOptions(), messages=[])
    recorder = RunRecorder(
        context=context,
        hooks=AgentHooks(),
        step_observers=[],
        state=state,
    )
    run = _make_run(context)

    await recorder.start_run(run)
    state.terminate(TerminationReason.CANCELLED)
    await recorder.complete_run(state.build_output())

    saved = await context.session_runtime.run_step_storage.get_run(context.run_id)
    assert saved is not None
    assert saved.status == RunStatus.CANCELLED
