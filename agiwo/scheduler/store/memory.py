"""In-memory scheduler state storage."""

import asyncio
from copy import deepcopy
from collections.abc import Collection

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    normalize_statuses,
)
from agiwo.scheduler.store.base import AgentStateStorage


class InMemoryAgentStateStorage(AgentStateStorage):
    """In-memory implementation for testing."""

    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}
        self._events: dict[str, PendingEvent] = {}
        self._lock = asyncio.Lock()

    async def save_state(self, state: AgentState) -> None:
        async with self._lock:
            self._states[state.id] = deepcopy(state)

    async def get_state(self, state_id: str) -> AgentState | None:
        state = self._states.get(state_id)
        if state is None:
            return None
        return deepcopy(state)

    async def list_states(
        self,
        *,
        statuses: Collection[AgentStateStatus] | None = None,
        parent_id: str | None = None,
        session_id: str | None = None,
        signal_propagated: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentState]:
        status_filter = normalize_statuses(statuses)
        states = [deepcopy(state) for state in self._states.values()]
        if status_filter is not None:
            states = [state for state in states if state.status in status_filter]
        if parent_id is not None:
            states = [state for state in states if state.parent_id == parent_id]
        if session_id is not None:
            states = [state for state in states if state.session_id == session_id]
        if signal_propagated is not None:
            states = [
                state
                for state in states
                if state.signal_propagated == signal_propagated
            ]
        states.sort(key=lambda state: state.updated_at, reverse=True)
        return states[offset : offset + limit]

    async def save_event(self, event: PendingEvent) -> None:
        async with self._lock:
            self._events[event.id] = deepcopy(event)

    async def list_events(
        self,
        *,
        target_agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[PendingEvent]:
        events = [deepcopy(event) for event in self._events.values()]
        if target_agent_id is not None:
            events = [
                event for event in events if event.target_agent_id == target_agent_id
            ]
        if session_id is not None:
            events = [event for event in events if event.session_id == session_id]
        events.sort(key=lambda event: event.created_at)
        return events

    async def delete_events(self, event_ids: list[str]) -> None:
        async with self._lock:
            for event_id in event_ids:
                self._events.pop(event_id, None)
