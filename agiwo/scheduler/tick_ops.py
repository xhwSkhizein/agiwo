"""Tick-phase orchestration helpers for the scheduler engine."""

import asyncio
from datetime import datetime, timezone

from agiwo.scheduler.coordinator import SchedulerCoordinator
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    SchedulerConfig,
)
from agiwo.scheduler.runner import SchedulerRunner
from agiwo.scheduler.selectors import (
    select_debounced_event_targets,
    select_ready_waiting_states,
    select_timed_out_states,
    select_unpropagated_children,
)
from agiwo.scheduler.state_ops import SchedulerStateOps
from agiwo.scheduler.store import AgentStateStorage
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SchedulerTickOps:
    def __init__(
        self,
        *,
        config: SchedulerConfig,
        store: AgentStateStorage,
        guard: TaskGuard,
        coordinator: SchedulerCoordinator,
        runner: SchedulerRunner,
        state_ops: SchedulerStateOps,
    ) -> None:
        self._config = config
        self._store = store
        self._guard = guard
        self._coordinator = coordinator
        self._runner = runner
        self._state_ops = state_ops

    async def propagate_signals(self) -> None:
        candidates = await self._store.list_states(
            statuses=(AgentStateStatus.COMPLETED, AgentStateStatus.FAILED),
            signal_propagated=False,
            limit=1000,
        )
        for state in select_unpropagated_children(candidates):
            await self._propagate_child_completion(state)
            state.signal_propagated = True
            await self._store.save_state(state)

    async def enforce_timeouts(self) -> None:
        now = datetime.now(timezone.utc)
        waiting_states = await self._store.list_states(
            statuses=(AgentStateStatus.WAITING,),
            limit=1000,
        )
        for state in select_timed_out_states(waiting_states, now=now):
            self._coordinator.dispatch_state_task(
                state.id,
                lambda state=state: self._runner.wake_for_timeout(state),
            )

    async def start_pending(self) -> None:
        for state in await self._store.list_states(
            statuses=(AgentStateStatus.PENDING,),
            limit=1000,
        ):
            self._coordinator.dispatch_state_task(
                state.id,
                lambda state=state: self._runner.run_pending_agent(state),
            )

    async def start_queued_roots(self) -> None:
        for state in await self._store.list_states(
            statuses=(AgentStateStatus.QUEUED,),
            limit=1000,
        ):
            if not state.is_root:
                continue
            self._coordinator.dispatch_state_task(
                state.id,
                lambda state=state: self._runner.run_queued_root(state),
            )

    async def process_pending_events(self) -> None:
        now = datetime.now(timezone.utc)
        events = await self._store.list_events()
        targets = select_debounced_event_targets(
            events,
            min_count=self._config.event_debounce_min_count,
            max_wait_seconds=self._config.event_debounce_max_wait_seconds,
            now=now,
        )
        for agent_id, session_id in targets:
            state = await self._store.get_state(agent_id)
            if state is None:
                await self._delete_pending_events(agent_id, session_id)
                continue
            if state.status != AgentStateStatus.WAITING:
                await self._delete_pending_events(agent_id, session_id)
                continue
            rejection = await self._guard.check_wake(state)
            if rejection is not None:
                logger.warning(
                    "pending_events_wake_rejected",
                    state_id=agent_id,
                    reason=rejection,
                )
                continue
            await self._dispatch_pending_events_wake(state)

    async def wake_waiting(self) -> None:
        now = datetime.now(timezone.utc)
        waiting_states = await self._store.list_states(
            statuses=(AgentStateStatus.WAITING,),
            limit=1000,
        )
        for state in select_ready_waiting_states(waiting_states, now=now):
            rejection = await self._guard.check_wake(state)
            if rejection is not None:
                logger.warning(
                    "wake_rejected",
                    state_id=state.id,
                    reason=rejection,
                )
                await self._state_ops.mark_failed(
                    state,
                    f"Wake rejected: {rejection}",
                )
                continue
            self._coordinator.dispatch_state_task(
                state.id,
                lambda state=state: self._runner.wake_agent(state),
            )

    async def try_urgent_wake(self, state: AgentState) -> None:
        rejection = await self._guard.check_wake(state)
        if rejection is not None:
            logger.warning("urgent_wake_rejected", state_id=state.id, reason=rejection)
            return
        await self._dispatch_pending_events_wake(state)

    async def _dispatch_pending_events_wake(self, state: AgentState) -> None:
        if not self._coordinator.reserve_state_dispatch(state.id):
            return

        events = await self._store.list_events(
            target_agent_id=state.id,
            session_id=state.session_id,
        )
        if not events:
            self._coordinator.release_state_dispatch(state.id)
            return

        await self._store.delete_events([event.id for event in events])
        self._coordinator.track_active_task(
            asyncio.create_task(self._runner.wake_agent_for_events(state, events))
        )

    async def _delete_pending_events(self, agent_id: str, session_id: str) -> None:
        events = await self._store.list_events(
            target_agent_id=agent_id,
            session_id=session_id,
        )
        if events:
            await self._store.delete_events([event.id for event in events])

    async def _propagate_child_completion(self, state: AgentState) -> None:
        parent = await self._store.get_state(state.parent_id or "")
        if parent is None or parent.wake_condition is None:
            return
        completed_ids = list(parent.wake_condition.completed_ids)
        if state.id not in completed_ids:
            completed_ids.append(state.id)
        parent.wake_condition.completed_ids = completed_ids
        await self._store.save_state(parent)


__all__ = ["SchedulerTickOps"]
