"""Scheduler waiting and pending-state resolution helpers."""

import asyncio
import time

from agiwo.agent import RunOutput, RunStatus, TerminationReason
from agiwo.scheduler.models import AgentState, AgentStateStatus
from agiwo.scheduler.runtime_facts import SchedulerRuntimeFacts
from agiwo.scheduler.runtime_state import RuntimeState
from agiwo.scheduler.store.base import AgentStateStorage


async def wait_for_state_result(
    *,
    store: AgentStateStorage,
    rt: RuntimeState,
    runtime_facts: SchedulerRuntimeFacts,
    state_id: str,
    timeout: float | None = None,
) -> RunOutput:
    start = time.monotonic()
    event = asyncio.Event()
    waiters = rt.waiters.setdefault(state_id, set())
    waiters.add(event)

    try:
        while True:
            state = await store.get_state(state_id)
            result = await build_wait_result(runtime_facts, state)
            if result is not None:
                return result

            remaining = remaining_wait_timeout(start, timeout)
            if remaining == 0:
                return RunOutput(termination_reason=TerminationReason.TIMEOUT)

            try:
                await asyncio.wait_for(event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return RunOutput(termination_reason=TerminationReason.TIMEOUT)
            event.clear()
    finally:
        _remove_waiter(rt, state_id, event)


async def resolve_routable_state(
    *,
    store: AgentStateStorage,
    rt: RuntimeState,
    state_id: str,
    deadline: float | None,
) -> AgentState | None:
    state = await store.get_state(state_id)
    if state is None:
        return None
    if state.status != AgentStateStatus.PENDING:
        return state
    return await wait_until_not_pending(
        store=store,
        rt=rt,
        state_id=state.id,
        deadline=deadline,
    )


async def wait_until_not_pending(
    *,
    store: AgentStateStorage,
    rt: RuntimeState,
    state_id: str,
    deadline: float | None,
) -> AgentState | None:
    event = asyncio.Event()
    waiters = rt.waiters.setdefault(state_id, set())
    waiters.add(event)
    try:
        while True:
            state = await store.get_state(state_id)
            if state is None or state.status != AgentStateStatus.PENDING:
                return state
            try:
                await asyncio.wait_for(
                    event.wait(),
                    timeout=deadline_remaining(deadline),
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    f"Timed out waiting for scheduler state '{state_id}' "
                    "to become routable"
                ) from exc
            event.clear()
    finally:
        _remove_waiter(rt, state_id, event)


async def build_wait_result(
    runtime_facts: SchedulerRuntimeFacts,
    state: AgentState | None,
) -> RunOutput | None:
    if state is None or not is_terminal_wait_state(state):
        return None
    latest_run = await runtime_facts.get_latest_run_view(state)
    if latest_run is not None and latest_run.status == RunStatus.COMPLETED:
        return RunOutput(
            session_id=latest_run.session_id,
            run_id=latest_run.run_id,
            response=latest_run.response,
            metrics=latest_run.metrics,
            termination_reason=latest_run.termination_reason
            or TerminationReason.COMPLETED,
        )
    return build_wait_result_from_last_run(state)


def is_terminal_wait_state(state: AgentState) -> bool:
    return state.status in (
        AgentStateStatus.IDLE,
        AgentStateStatus.COMPLETED,
        AgentStateStatus.FAILED,
    )


def build_wait_result_from_last_run(state: AgentState) -> RunOutput | None:
    if state.last_run_result is None:
        return None
    last_run_result = state.last_run_result
    return RunOutput(
        run_id=last_run_result.run_id,
        response=last_run_result.summary if last_run_result.error is None else None,
        error=last_run_result.error,
        termination_reason=last_run_result.termination_reason,
    )


def remaining_wait_timeout(start: float, timeout: float | None) -> float | None:
    if timeout is None:
        return None
    elapsed = time.monotonic() - start
    if elapsed >= timeout:
        return 0
    return timeout - elapsed


def deadline_remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return 0
    return remaining


def _remove_waiter(rt: RuntimeState, state_id: str, event: asyncio.Event) -> None:
    waiters = rt.waiters.get(state_id)
    if waiters is not None:
        waiters.discard(event)
        if not waiters:
            rt.waiters.pop(state_id, None)


__all__ = [
    "build_wait_result",
    "deadline_remaining",
    "resolve_routable_state",
    "wait_for_state_result",
]
