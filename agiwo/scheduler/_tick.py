"""Tick loop and dispatch logic extracted from Scheduler."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from agiwo.scheduler.commands import DispatchAction, DispatchReason
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
)
from agiwo.scheduler.runtime_state import (
    build_mailbox_input,
    group_events,
    list_all_states,
    select_debounced_event_targets,
)
from agiwo.utils.logging import get_logger

if TYPE_CHECKING:
    from agiwo.scheduler.engine import Scheduler

logger = get_logger(__name__)


async def tick(sched: Scheduler) -> None:
    await propagate_signals(sched)
    states = await list_all_states(
        sched._store,
        statuses=(
            AgentStateStatus.PENDING,
            AgentStateStatus.WAITING,
            AgentStateStatus.QUEUED,
        ),
    )
    events = await sched._store.list_events()
    now = datetime.now(timezone.utc)
    actions = plan_tick(sched, states, events, now=now)
    for action in actions:
        await dispatch_action(sched, action)


def plan_tick(
    sched: Scheduler,
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
        min_count=sched._config.event_debounce_min_count,
        max_wait_seconds=sched._config.event_debounce_max_wait_seconds,
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
                    DispatchAction(state=state, reason=DispatchReason.WAKE_TIMEOUT)
                )
                continue

            if state.wake_condition.is_satisfied(now):
                actions.append(
                    DispatchAction(state=state, reason=DispatchReason.WAKE_READY)
                )
                continue

        key = (state.id, state.session_id)
        if key in debounced_targets:
            grouped_events = tuple(event_groups.get(key, []))
            if grouped_events:
                actions.append(
                    DispatchAction(
                        state=state,
                        reason=DispatchReason.WAKE_EVENTS,
                        events=grouped_events,
                    )
                )

    return actions


async def dispatch_action(sched: Scheduler, action: DispatchAction) -> None:
    state = action.state
    if state.id in sched._rt.dispatched:
        return

    if action.reason in (
        DispatchReason.WAKE_READY,
        DispatchReason.WAKE_EVENTS,
        DispatchReason.WAKE_TIMEOUT,
    ):
        rejection = await sched._guard.check_wake(state)
        if rejection is not None:
            await sched._save_state(state.with_failed(f"Wake rejected: {rejection}"))
            return

    sched._rt.dispatched.add(state.id)
    task = asyncio.create_task(sched._runner.run(action))
    sched._track_active_task(task)


async def propagate_signals(sched: Scheduler) -> None:
    candidates = await sched._store.list_states(
        statuses=(AgentStateStatus.COMPLETED, AgentStateStatus.FAILED),
        signal_propagated=False,
        limit=1000,
    )
    for state in candidates:
        if not state.is_child or state.signal_propagated:
            continue

        parent = await sched._store.get_state(state.parent_id or "")
        if parent is not None and parent.wake_condition is not None:
            completed_ids = list(parent.wake_condition.completed_ids)
            if state.id not in completed_ids:
                completed_ids.append(state.id)
                await sched._save_state(
                    parent.with_updates(
                        wake_condition=parent.wake_condition.with_completed_ids(
                            completed_ids
                        )
                    )
                )

        await sched._save_state(state.with_signal_propagated())
