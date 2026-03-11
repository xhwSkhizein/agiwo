"""Scheduler tick orchestration service."""

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from agiwo.scheduler.executor import SchedulerExecutor
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerConfig,
    SchedulerEventType,
)
from agiwo.scheduler.runtime import SchedulerRuntime
from agiwo.scheduler.store import AgentStateStorage
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SchedulerTickEngine:
    """Run scheduler tick stages and dispatch runnable agent tasks."""

    def __init__(
        self,
        *,
        store: AgentStateStorage,
        guard: TaskGuard,
        executor: SchedulerExecutor,
        runtime: SchedulerRuntime,
        config: SchedulerConfig,
    ) -> None:
        self._store = store
        self._guard = guard
        self._executor = executor
        self._runtime = runtime
        self._config = config

    async def tick(self) -> None:
        await self.propagate_signals()
        await self.enforce_timeouts()
        await self.check_health()
        await self.process_pending_events()
        await self.start_pending()
        await self.wake_sleeping()

    async def propagate_signals(self) -> None:
        completed = await self._store.find_unpropagated_completed()
        for state in completed:
            if state.parent_id is not None:
                await self._store.mark_child_completed(state.parent_id, state.id)
                logger.info(
                    "signal_propagated",
                    child_id=state.id,
                    parent_id=state.parent_id,
                    child_status=state.status.value,
                )
            await self._store.mark_propagated(state.id)

    async def enforce_timeouts(self) -> None:
        now = datetime.now(timezone.utc)
        timed_out = await self._guard.find_timed_out(now)
        for state in timed_out:
            self._runtime.dispatch_state_task(
                state,
                lambda: self._executor.wake_for_timeout(state),
            )

    async def start_pending(self) -> None:
        pending = await self._store.find_pending()
        for state in pending:
            self._runtime.dispatch_state_task(
                state,
                lambda: self._executor.run_agent(state),
            )

    async def check_health(self) -> None:
        now = datetime.now(timezone.utc)
        unhealthy = await self._guard.find_unhealthy(now)
        for state in unhealthy:
            if state.parent_id is None:
                continue
            parent_state = await self._store.get_state(state.parent_id)
            if parent_state is None:
                continue

            already_warned = await self._store.has_recent_health_warning(
                target_agent_id=state.parent_id,
                source_agent_id=state.id,
                within_seconds=self._config.task_limits.health_check_threshold_seconds,
                now=now,
            )
            if already_warned:
                continue

            event = PendingEvent(
                id=str(uuid4()),
                target_agent_id=state.parent_id,
                session_id=state.session_id,
                event_type=SchedulerEventType.HEALTH_WARNING,
                payload={
                    "child_agent_id": state.id,
                    "message": (
                        f"Agent '{state.id}' appears stuck — no activity for "
                        f">{self._config.task_limits.health_check_threshold_seconds:.0f}s. "
                        "Consider using cancel_agent to terminate it."
                    ),
                    "last_activity_at": (
                        state.last_activity_at.isoformat() if state.last_activity_at else None
                    ),
                },
                source_agent_id=state.id,
                created_at=now,
            )
            await self._store.save_event(event)
            logger.warning(
                "health_warning_emitted",
                stuck_agent_id=state.id,
                parent_id=state.parent_id,
            )

    async def process_pending_events(self) -> None:
        now = datetime.now(timezone.utc)
        min_count = self._config.event_debounce_min_count
        max_wait = self._config.event_debounce_max_wait_seconds

        agent_session_pairs = await self._store.find_agents_with_debounced_events(
            min_count,
            max_wait,
            now,
        )
        for agent_id, session_id in agent_session_pairs:
            state = await self._store.get_state(agent_id)
            if state is None:
                await self._store.delete_events_by_agent(agent_id)
                continue

            if state.status != AgentStateStatus.SLEEPING:
                events = await self._store.get_pending_events(agent_id, session_id)
                if events:
                    await self._store.delete_events([e.id for e in events])
                continue

            rejection = await self._guard.check_wake(state)
            if rejection is not None:
                logger.warning(
                    "pending_events_wake_rejected",
                    state_id=agent_id,
                    reason=rejection,
                )
                continue

            if not self._runtime.reserve_state_dispatch(state.id):
                continue

            events = await self._store.get_pending_events(agent_id, state.session_id)
            if not events:
                self._runtime.release_state_dispatch(state.id)
                continue

            await self._store.delete_events([e.id for e in events])
            self._runtime.track_active_task(
                asyncio.create_task(self._executor.wake_agent_for_events(state, events))
            )

    async def wake_sleeping(self) -> None:
        now = datetime.now(timezone.utc)
        wakeable = await self._store.find_wakeable(now)
        for state in wakeable:
            rejection = await self._guard.check_wake(state)
            if rejection is not None:
                logger.warning(
                    "wake_rejected",
                    state_id=state.id,
                    reason=rejection,
                )
                await self._store.update_status(
                    state.id,
                    AgentStateStatus.FAILED,
                    result_summary=f"Wake rejected: {rejection}",
                )
                continue

            self._runtime.dispatch_state_task(
                state,
                lambda: self._executor.wake_agent(state),
            )

    async def try_urgent_wake(self, state: AgentState) -> None:
        rejection = await self._guard.check_wake(state)
        if rejection is not None:
            logger.warning("urgent_wake_rejected", state_id=state.id, reason=rejection)
            return

        if not self._runtime.reserve_state_dispatch(state.id):
            return

        events = await self._store.get_pending_events(state.id, state.session_id)
        if not events:
            self._runtime.release_state_dispatch(state.id)
            return

        await self._store.delete_events([e.id for e in events])
        self._runtime.track_active_task(
            asyncio.create_task(self._executor.wake_agent_for_events(state, events))
        )
