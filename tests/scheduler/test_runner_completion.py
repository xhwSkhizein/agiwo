from typing import Any

import pytest

from agiwo.agent import RunOutput, TerminationReason
from agiwo.scheduler.commands import DispatchAction, DispatchReason
from agiwo.scheduler.formatting import SHUTDOWN_SUMMARY_TASK
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    SchedulerEventType,
    TimeUnit,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.runner_completion import (
    RunnerCompletionContext,
    RunnerCompletionHandler,
)
from agiwo.utils.abort_signal import AbortSignal


def _build_handler(
    *,
    saved_states: list[AgentState],
    emitted_events: list[tuple[AgentState, SchedulerEventType, dict[str, object]]],
    should_rollback: bool = False,
    shutdown_requested: set[str] | None = None,
    rollback_calls: list[tuple[AgentState, RunOutput]] | None = None,
) -> RunnerCompletionHandler:
    async def save_state(state: AgentState) -> None:
        saved_states.append(state)

    async def emit_event_to_parent(
        state: AgentState,
        event_type: SchedulerEventType,
        payload: dict[str, object],
    ) -> None:
        emitted_events.append((state, event_type, payload))

    def resolve_abort_signal(
        state_id: str,
        abort_signal: AbortSignal | None,
    ) -> AbortSignal | None:
        return abort_signal

    async def rollback_run_steps(state: AgentState, output: RunOutput) -> None:
        if rollback_calls is not None:
            rollback_calls.append((state, output))

    return RunnerCompletionHandler(
        RunnerCompletionContext(
            save_state=save_state,
            emit_event_to_parent=emit_event_to_parent,
            resolve_abort_signal=resolve_abort_signal,
            should_rollback=lambda state: should_rollback,
            rollback_run_steps=rollback_run_steps,
            shutdown_requested=shutdown_requested or set(),
        )
    )


def _base_state(**kwargs: Any) -> AgentState:
    return AgentState(
        id="agent-1",
        session_id="sess-1",
        status=AgentStateStatus.RUNNING,
        task="task",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_failed_child_output_marks_state_failed_and_emits_parent_event() -> None:
    saved_states: list[AgentState] = []
    emitted_events: list[tuple[AgentState, SchedulerEventType, dict[str, object]]] = []
    handler = _build_handler(saved_states=saved_states, emitted_events=emitted_events)

    child_state = _base_state(parent_id="parent-1")
    handled = await handler.handle(
        state=child_state,
        output=RunOutput(
            run_id="run-1",
            response="tool failed",
            error="boom",
            termination_reason=TerminationReason.ERROR,
        ),
        action=DispatchAction(state=child_state, reason=DispatchReason.CHILD_PENDING),
        abort_signal=None,
    )

    assert handled is True
    assert len(saved_states) == 1
    assert saved_states[0].status == AgentStateStatus.FAILED
    assert saved_states[0].result_summary == "boom"
    assert saved_states[0].last_run_result is not None
    assert saved_states[0].last_run_result.run_id == "run-1"
    assert saved_states[0].last_run_result.termination_reason is TerminationReason.ERROR
    assert emitted_events == [
        (
            child_state,
            SchedulerEventType.CHILD_FAILED,
            {"reason": "boom"},
        )
    ]


@pytest.mark.asyncio
async def test_periodic_no_progress_rolls_back_and_preserves_previous_summary() -> None:
    saved_states: list[AgentState] = []
    emitted_events: list[tuple[AgentState, SchedulerEventType, dict[str, object]]] = []
    rollback_calls: list[tuple[AgentState, RunOutput]] = []
    handler = _build_handler(
        saved_states=saved_states,
        emitted_events=emitted_events,
        should_rollback=True,
        rollback_calls=rollback_calls,
    )

    periodic_state = _base_state(
        parent_id="parent-1",
        no_progress=True,
        rollback_count=2,
        result_summary="previous summary",
        explain="still waiting",
        wake_condition=WakeCondition(
            type=WakeType.PERIODIC,
            time_value=30,
            time_unit=TimeUnit.SECONDS,
        ),
    )
    output = RunOutput(
        run_id="run-2",
        response="new summary",
        termination_reason=TerminationReason.COMPLETED,
    )

    handled = await handler.handle(
        state=periodic_state,
        output=output,
        action=DispatchAction(state=periodic_state, reason=DispatchReason.WAKE_READY),
        abort_signal=None,
    )

    assert handled is True
    assert rollback_calls == [(periodic_state, output)]
    assert emitted_events == []
    assert len(saved_states) == 1
    next_state = saved_states[0]
    assert next_state.status == AgentStateStatus.WAITING
    assert next_state.rollback_count == 3
    assert next_state.result_summary == "previous summary"
    assert next_state.wake_condition is not None
    assert next_state.wake_condition.type is WakeType.PERIODIC
    assert next_state.wake_condition.wakeup_at is not None


@pytest.mark.asyncio
async def test_persistent_root_shutdown_requeues_summary_run() -> None:
    saved_states: list[AgentState] = []
    emitted_events: list[tuple[AgentState, SchedulerEventType, dict[str, object]]] = []
    shutdown_requested = {"root-1"}
    handler = _build_handler(
        saved_states=saved_states,
        emitted_events=emitted_events,
        shutdown_requested=shutdown_requested,
    )

    root_state = AgentState(
        id="root-1",
        session_id="sess-1",
        status=AgentStateStatus.RUNNING,
        task="task",
        is_persistent=True,
    )
    handled = await handler.handle(
        state=root_state,
        output=RunOutput(
            run_id="run-3",
            response="wrap up current work",
            termination_reason=TerminationReason.COMPLETED,
        ),
        action=DispatchAction(state=root_state, reason=DispatchReason.ROOT_SUBMIT),
        abort_signal=None,
    )

    assert handled is True
    assert shutdown_requested == set()
    assert emitted_events == []
    assert len(saved_states) == 1
    queued_state = saved_states[0]
    assert queued_state.status == AgentStateStatus.QUEUED
    assert queued_state.pending_input == SHUTDOWN_SUMMARY_TASK
    assert queued_state.result_summary == "wrap up current work"


@pytest.mark.asyncio
async def test_terminal_aborted_state_gets_cancelled_last_run_result() -> None:
    saved_states: list[AgentState] = []
    emitted_events: list[tuple[AgentState, SchedulerEventType, dict[str, object]]] = []
    handler = _build_handler(saved_states=saved_states, emitted_events=emitted_events)

    abort_signal = AbortSignal()
    abort_signal.abort("cancelled by user")
    terminal_state = AgentState(
        id="root-1",
        session_id="sess-1",
        status=AgentStateStatus.FAILED,
        task="task",
    )

    handled = await handler.handle(
        state=terminal_state,
        output=RunOutput(termination_reason=TerminationReason.COMPLETED),
        action=DispatchAction(state=terminal_state, reason=DispatchReason.ROOT_SUBMIT),
        abort_signal=abort_signal,
    )

    assert handled is True
    assert emitted_events == []
    assert len(saved_states) == 1
    assert saved_states[0].last_run_result is not None
    assert (
        saved_states[0].last_run_result.termination_reason
        is TerminationReason.CANCELLED
    )
    assert saved_states[0].last_run_result.error == "cancelled by user"
