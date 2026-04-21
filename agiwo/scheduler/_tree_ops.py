"""Subtree cancel/shutdown logic extracted from Scheduler."""

from agiwo.agent import TerminationReason
from agiwo.scheduler.engine_context import EngineContext
from agiwo.scheduler.formatting import SHUTDOWN_SUMMARY_TASK
from agiwo.scheduler.models import AgentState, AgentStateStatus, SchedulerRunResult
from agiwo.scheduler.stream import finish_stream_channel
from agiwo.utils.abort_signal import AbortSignal


async def cancel_subtree(ctx: EngineContext, state_id: str, reason: str) -> None:
    abort_runtime_state(ctx, state_id, reason)
    for child in await active_children(ctx, state_id):
        await cancel_subtree(ctx, child.id, reason)

    state = await ctx.store.get_state(state_id)
    if state is None:
        return
    cancelled_result = SchedulerRunResult(
        run_id=None,
        termination_reason=TerminationReason.CANCELLED,
        error=reason,
    )
    await ctx.save_state(
        state.with_failed(reason).with_updates(last_run_result=cancelled_result)
    )
    if state.is_root:
        await finish_stream_channel(ctx.rt.stream_channels, state.id)


async def shutdown_subtree(ctx: EngineContext, state_id: str) -> None:
    for child in await active_children(ctx, state_id):
        await shutdown_subtree(ctx, child.id)

    state = await ctx.store.get_state(state_id)
    if state is None:
        return

    abort_runtime_state(ctx, state_id, "Shutdown requested")
    if state.status == AgentStateStatus.RUNNING:
        await shutdown_running_state(ctx, state)
        return

    await shutdown_passive_state(ctx, state)


async def active_children(ctx: EngineContext, state_id: str) -> list[AgentState]:
    # Fetch all children with pagination to avoid missing children beyond page_size
    all_children: list[AgentState] = []
    offset = 0
    while True:
        page = await ctx.store.list_states(
            parent_id=state_id,
            limit=ctx.state_list_page_size,
            offset=offset,
        )
        if not page:
            break
        all_children.extend(page)
        # If we got fewer than page_size, we've reached the end
        if len(page) < ctx.state_list_page_size:
            break
        offset += len(page)
    return [child for child in all_children if child.is_active()]


def abort_runtime_state(ctx: EngineContext, state_id: str, reason: str) -> None:
    signal = ctx.rt.abort_signals.get(state_id)
    if signal is None:
        signal = AbortSignal()
        ctx.rt.abort_signals[state_id] = signal
    if not signal.is_aborted():
        signal.abort(reason)

    handle = ctx.rt.execution_handles.get(state_id)
    if handle is not None:
        handle.cancel(reason)


async def shutdown_running_state(ctx: EngineContext, state: AgentState) -> None:
    if state.is_root and state.is_persistent:
        ctx.rt.shutdown_requested.add(state.id)
        await ctx.save_state(state.with_queued(pending_input=SHUTDOWN_SUMMARY_TASK))
        return
    await _save_shutdown_failed(ctx, state)


async def shutdown_passive_state(ctx: EngineContext, state: AgentState) -> None:
    if state.is_root and state.status in (
        AgentStateStatus.WAITING,
        AgentStateStatus.IDLE,
    ):
        await ctx.save_state(state.with_queued(pending_input=SHUTDOWN_SUMMARY_TASK))
        return

    if state.status in (
        AgentStateStatus.WAITING,
        AgentStateStatus.PENDING,
        AgentStateStatus.QUEUED,
    ):
        await _save_shutdown_failed(ctx, state)
        if state.is_root:
            await finish_stream_channel(ctx.rt.stream_channels, state.id)


async def _save_shutdown_failed(ctx: EngineContext, state: AgentState) -> None:
    reason = "Shutdown before completion"
    shutdown_result = SchedulerRunResult(
        run_id=None,
        termination_reason=TerminationReason.CANCELLED,
        error=reason,
    )
    await ctx.save_state(
        state.with_failed(reason).with_updates(last_run_result=shutdown_result)
    )
