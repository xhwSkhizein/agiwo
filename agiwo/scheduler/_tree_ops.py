"""Subtree cancel/shutdown logic extracted from Scheduler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agiwo.scheduler.formatting import SHUTDOWN_SUMMARY_TASK
from agiwo.scheduler.models import AgentState, AgentStateStatus
from agiwo.scheduler.stream import finish_stream_channel
from agiwo.utils.abort_signal import AbortSignal

if TYPE_CHECKING:
    from agiwo.scheduler.engine import Scheduler


async def cancel_subtree(sched: Scheduler, state_id: str, reason: str) -> None:
    abort_runtime_state(sched, state_id, reason)
    for child in await active_children(sched, state_id):
        await cancel_subtree(sched, child.id, reason)

    state = await sched._store.get_state(state_id)
    if state is None:
        return
    await sched._save_state(state.with_failed(reason))
    if state.is_root:
        await finish_stream_channel(sched._rt.stream_channels, state.id)


async def shutdown_subtree(sched: Scheduler, state_id: str) -> None:
    for child in await active_children(sched, state_id):
        await shutdown_subtree(sched, child.id)

    state = await sched._store.get_state(state_id)
    if state is None:
        return

    abort_runtime_state(sched, state_id, "Shutdown requested")
    if state.status == AgentStateStatus.RUNNING:
        await shutdown_running_state(sched, state)
        return

    await shutdown_passive_state(sched, state)


async def active_children(sched: Scheduler, state_id: str) -> list[AgentState]:
    children = await sched._store.list_states(parent_id=state_id, limit=1000)
    return [child for child in children if child.is_active()]


def abort_runtime_state(sched: Scheduler, state_id: str, reason: str) -> None:
    signal = sched._rt.abort_signals.get(state_id)
    if signal is None:
        signal = AbortSignal()
        sched._rt.abort_signals[state_id] = signal
    if not signal.is_aborted():
        signal.abort(reason)

    handle = sched._rt.execution_handles.get(state_id)
    if handle is not None:
        handle.cancel(reason)


async def shutdown_running_state(sched: Scheduler, state: AgentState) -> None:
    if state.is_root and state.is_persistent:
        sched._rt.shutdown_requested.add(state.id)
        await sched._save_state(
            state.with_queued(pending_input=SHUTDOWN_SUMMARY_TASK)
        )
        return
    await sched._save_state(state.with_failed("Shutdown before completion"))


async def shutdown_passive_state(sched: Scheduler, state: AgentState) -> None:
    if state.is_root and state.status in (
        AgentStateStatus.WAITING,
        AgentStateStatus.IDLE,
    ):
        await sched._save_state(
            state.with_queued(pending_input=SHUTDOWN_SUMMARY_TASK)
        )
        return

    if state.status in (
        AgentStateStatus.WAITING,
        AgentStateStatus.PENDING,
        AgentStateStatus.QUEUED,
    ):
        await sched._save_state(state.with_failed("Shutdown before completion"))
        if state.is_root:
            await finish_stream_channel(sched._rt.stream_channels, state.id)
