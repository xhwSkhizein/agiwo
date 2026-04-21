"""Tick loop and dispatch logic extracted from Scheduler."""

import asyncio
from datetime import datetime, timezone

from agiwo.scheduler.commands import DispatchAction, DispatchReason
from agiwo.scheduler.engine_context import EngineContext
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
    WakeType,
)
from agiwo.scheduler.runtime_state import (
    build_mailbox_input,
    group_events,
    list_all_states,
    select_debounced_event_targets,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

_ACKABLE_CHILD_EVENT_TYPES = frozenset(
    {
        SchedulerEventType.CHILD_COMPLETED,
        SchedulerEventType.CHILD_FAILED,
        SchedulerEventType.CHILD_SLEEP_RESULT,
    }
)


def _ackable_wait_events(
    state: AgentState,
    event_groups: dict[tuple[str, str], list[PendingEvent]],
) -> tuple[PendingEvent, ...]:
    grouped_events = tuple(event_groups.get((state.id, state.session_id), []))
    if not grouped_events:
        return ()

    wake_condition = state.wake_condition
    if wake_condition is None or wake_condition.type != WakeType.WAITSET:
        return ()

    wait_for_ids = set(wake_condition.wait_for)
    if not wait_for_ids:
        return ()

    ackable: list[PendingEvent] = []
    for event in grouped_events:
        if event.event_type not in _ACKABLE_CHILD_EVENT_TYPES:
            continue
        if event.source_agent_id not in wait_for_ids:
            continue
        ackable.append(event)
    return tuple(ackable)


async def tick(ctx: EngineContext) -> None:
    await propagate_signals(ctx)
    states = await list_all_states(
        ctx.store,
        statuses=(
            AgentStateStatus.PENDING,
            AgentStateStatus.WAITING,
            AgentStateStatus.QUEUED,
        ),
    )
    events = await ctx.store.list_events()
    now = datetime.now(timezone.utc)
    actions = plan_tick(ctx, states, events, now=now)
    for action in actions:
        await dispatch_action(ctx, action)


def plan_tick(
    ctx: EngineContext,
    states: list[AgentState],
    events: list[PendingEvent],
    *,
    now: datetime,
) -> list[DispatchAction]:
    """Pure logic function -- no IO, no awaits.

    **Design decision (sync):** ``plan_tick`` is deliberately synchronous.
    All required data (states, events) must be pre-fetched by the caller
    and passed in.  This gives three benefits:

    1. **Testability** -- correctness can be verified with plain ``pytest``
       assertions, no event loop required.
    2. **Determinism** -- the same inputs always produce the same actions,
       independent of storage latency or concurrency.
    3. **Separation of concerns** -- IO lives in ``tick()``; scheduling
       logic lives here.
    """
    actions: list[DispatchAction] = []
    event_groups = group_events(events)
    debounced_targets = select_debounced_event_targets(
        events,
        min_count=ctx.config.event_debounce_min_count,
        max_wait_seconds=ctx.config.event_debounce_max_wait_seconds,
        now=now,
    )

    for state in states:
        if state.status == AgentStateStatus.PENDING:
            actions.append(
                DispatchAction(state=state, reason=DispatchReason.CHILD_PENDING)
            )
            continue

        if state.is_root and state.status == AgentStateStatus.QUEUED:
            mailbox = tuple(
                event
                for event in event_groups.get((state.id, state.session_id), [])
                if event.event_type == SchedulerEventType.USER_HINT
            )
            actions.append(
                DispatchAction(
                    state=state,
                    reason=DispatchReason.ROOT_QUEUED_INPUT,
                    input_override=build_mailbox_input(state.pending_input, mailbox),
                    events=mailbox,
                )
            )
            continue

        if state.status != AgentStateStatus.WAITING:
            continue

        if state.wake_condition is not None:
            if state.wake_condition.is_timed_out(now):
                actions.append(
                    DispatchAction(
                        state=state,
                        reason=DispatchReason.WAKE_TIMEOUT,
                        events=_ackable_wait_events(state, event_groups),
                    )
                )
                continue

            if state.wake_condition.is_satisfied(now):
                actions.append(
                    DispatchAction(
                        state=state,
                        reason=DispatchReason.WAKE_READY,
                        events=_ackable_wait_events(state, event_groups),
                    )
                )
                continue

        key = (state.id, state.session_id)
        grouped_events = tuple(event_groups.get(key, []))
        has_urgent_event = any(event.urgent for event in grouped_events)
        if grouped_events and (has_urgent_event or key in debounced_targets):
            actions.append(
                DispatchAction(
                    state=state,
                    reason=DispatchReason.WAKE_EVENTS,
                    events=grouped_events,
                )
            )

    return actions


async def dispatch_action(ctx: EngineContext, action: DispatchAction) -> None:
    state = action.state
    if state.id in ctx.rt.dispatched:
        return

    if action.reason in (
        DispatchReason.WAKE_READY,
        DispatchReason.WAKE_EVENTS,
        DispatchReason.WAKE_TIMEOUT,
    ):
        rejection = await ctx.guard.check_wake(state)
        if rejection is not None:
            await ctx.save_state(state.with_failed(f"Wake rejected: {rejection}"))
            return

    ctx.rt.dispatched.add(state.id)
    task = asyncio.create_task(ctx.runner.run(action))
    ctx.track_active_task(task)


async def propagate_signals(ctx: EngineContext) -> None:
    # Fetch all candidates with pagination to avoid missing states beyond page_size
    all_candidates: list[AgentState] = []
    offset = 0
    while True:
        page = await ctx.store.list_states(
            statuses=(AgentStateStatus.COMPLETED, AgentStateStatus.FAILED),
            signal_propagated=False,
            limit=ctx.state_list_page_size,
            offset=offset,
        )
        if not page:
            break
        all_candidates.extend(page)
        # If we got fewer than page_size, we've reached the end
        if len(page) < ctx.state_list_page_size:
            break
        offset += len(page)

    for state in all_candidates:
        if not state.is_child or state.signal_propagated:
            continue

        parent = await ctx.store.get_state(state.parent_id or "")
        if parent is not None and parent.wake_condition is not None:
            completed_ids = list(parent.wake_condition.completed_ids)
            if state.id not in completed_ids:
                completed_ids.append(state.id)
                await ctx.save_state(
                    parent.with_updates(
                        wake_condition=parent.wake_condition.with_completed_ids(
                            completed_ids
                        )
                    )
                )

        await ctx.save_state(state.with_signal_propagated())
