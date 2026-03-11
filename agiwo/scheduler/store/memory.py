"""In-memory scheduler state storage."""

import asyncio
from datetime import datetime, timezone

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    WakeCondition,
)
from agiwo.scheduler.store.base import AgentStateStorage
from agiwo.scheduler.store.semantics import (
    apply_status_update,
    append_recent_steps,
    find_debounced_agent_sessions,
    has_recent_health_warning_event,
    is_timed_out_state,
    is_wakeable_state,
)


class InMemoryAgentStateStorage(AgentStateStorage):
    """In-memory implementation for testing."""

    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}
        self._events: dict[str, PendingEvent] = {}
        self._lock = asyncio.Lock()

    async def save_state(self, state: AgentState) -> None:
        async with self._lock:
            self._states[state.id] = state

    async def get_state(self, state_id: str) -> AgentState | None:
        return self._states.get(state_id)

    async def update_status(
        self,
        state_id: str,
        status: AgentStateStatus,
        *,
        wake_condition: WakeCondition | None = ...,
        result_summary: str | None = ...,
        explain: str | None = ...,
        last_activity_at: datetime | None = ...,
        recent_steps: list[dict] | None = ...,
    ) -> None:
        async with self._lock:
            state = self._states.get(state_id)
            if state is None:
                return
            apply_status_update(
                state,
                status=status,
                wake_condition=wake_condition,
                result_summary=result_summary,
                explain=explain,
                last_activity_at=last_activity_at,
                recent_steps=recent_steps,
                now=datetime.now(timezone.utc),
            )

    async def get_states_by_parent(self, parent_id: str) -> list[AgentState]:
        return [s for s in self._states.values() if s.parent_id == parent_id]

    async def find_pending(self) -> list[AgentState]:
        return [
            s for s in self._states.values() if s.status == AgentStateStatus.PENDING
        ]

    async def find_wakeable(self, now: datetime) -> list[AgentState]:
        return [s for s in self._states.values() if is_wakeable_state(s, now)]

    async def find_unpropagated_completed(self) -> list[AgentState]:
        return [
            s
            for s in self._states.values()
            if s.status in (AgentStateStatus.COMPLETED, AgentStateStatus.FAILED)
            and not s.signal_propagated
            and s.parent_id is not None
        ]

    async def mark_child_completed(self, parent_id: str, child_id: str) -> None:
        async with self._lock:
            state = self._states.get(parent_id)
            if state is None or state.wake_condition is None:
                return
            if child_id not in state.wake_condition.completed_ids:
                state.wake_condition.completed_ids.append(child_id)
            state.updated_at = datetime.now(timezone.utc)

    async def mark_propagated(self, state_id: str) -> None:
        async with self._lock:
            state = self._states.get(state_id)
            if state is None:
                return
            state.signal_propagated = True
            state.updated_at = datetime.now(timezone.utc)

    async def find_timed_out(self, now: datetime) -> list[AgentState]:
        return [s for s in self._states.values() if is_timed_out_state(s, now)]

    async def increment_wake_count(self, state_id: str) -> None:
        async with self._lock:
            state = self._states.get(state_id)
            if state is None:
                return
            state.wake_count += 1
            state.updated_at = datetime.now(timezone.utc)

    async def list_all(
        self,
        *,
        status: AgentStateStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentState]:
        states = list(self._states.values())
        if status is not None:
            states = [s for s in states if s.status == status]
        states.sort(key=lambda s: s.updated_at, reverse=True)
        return states[offset : offset + limit]

    async def find_running(self) -> list[AgentState]:
        return [
            s for s in self._states.values() if s.status == AgentStateStatus.RUNNING
        ]

    async def save_event(self, event: PendingEvent) -> None:
        async with self._lock:
            self._events[event.id] = event

    async def get_pending_events(
        self, target_agent_id: str, session_id: str
    ) -> list[PendingEvent]:
        events = [
            e
            for e in self._events.values()
            if e.target_agent_id == target_agent_id and e.session_id == session_id
        ]
        events.sort(key=lambda e: e.created_at)
        return events

    async def delete_events(self, event_ids: list[str]) -> None:
        async with self._lock:
            for eid in event_ids:
                self._events.pop(eid, None)

    async def append_recent_step(self, state_id: str, step: dict) -> None:
        async with self._lock:
            state = self._states.get(state_id)
            if state is None:
                return
            state.recent_steps = append_recent_steps(state.recent_steps, step)
            state.last_activity_at = datetime.now(timezone.utc)
            state.updated_at = datetime.now(timezone.utc)

    async def delete_events_by_agent(self, target_agent_id: str) -> None:
        async with self._lock:
            to_delete = [
                eid
                for eid, event in self._events.items()
                if event.target_agent_id == target_agent_id
            ]
            for eid in to_delete:
                del self._events[eid]

    async def find_agents_with_debounced_events(
        self,
        min_count: int,
        max_wait_seconds: float,
        now: datetime,
    ) -> list[tuple[str, str]]:
        return find_debounced_agent_sessions(
            list(self._events.values()),
            min_count=min_count,
            max_wait_seconds=max_wait_seconds,
            now=now,
        )

    async def has_recent_health_warning(
        self,
        target_agent_id: str,
        source_agent_id: str,
        within_seconds: float,
        now: datetime,
    ) -> bool:
        return has_recent_health_warning_event(
            list(self._events.values()),
            target_agent_id=target_agent_id,
            source_agent_id=source_agent_id,
            within_seconds=within_seconds,
            now=now,
        )
