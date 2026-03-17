"""Shared scheduler state transitions and waiter notifications."""

from datetime import datetime, timezone

from agiwo.agent.input import UserInput
from agiwo.scheduler.coordinator import SchedulerCoordinator
from agiwo.scheduler.models import AgentState, AgentStateStatus, WakeCondition
from agiwo.scheduler.store import AgentStateStorage


def _apply_fields(state: AgentState, **kwargs: object) -> None:
    """Apply only provided (non-sentinel) fields to state."""
    for key, value in kwargs.items():
        if value is not ...:
            setattr(state, key, value)


class SchedulerStateOps:
    """Centralize persisted state transitions plus in-memory waiter wakeups."""

    def __init__(
        self,
        *,
        store: AgentStateStorage,
        coordinator: SchedulerCoordinator,
    ) -> None:
        self._store = store
        self._coordinator = coordinator

    async def _save(self, state: AgentState, *, notify: bool = False) -> None:
        state.updated_at = datetime.now(timezone.utc)
        await self._store.save_state(state)
        if notify:
            self._coordinator.notify_state_change(state.id)

    async def mark_running(
        self,
        state: AgentState,
        *,
        task: UserInput | object = ...,
        pending_input: UserInput | None | object = ...,
        wake_condition: WakeCondition | None | object = ...,
        result_summary: str | None | object = ...,
        explain: str | None | object = ...,
        wake_count: int | object = ...,
    ) -> None:
        state.status = AgentStateStatus.RUNNING
        _apply_fields(
            state,
            task=task,
            pending_input=pending_input,
            wake_condition=wake_condition,
            result_summary=result_summary,
            explain=explain,
            wake_count=wake_count,
        )
        await self._save(state)

    async def mark_waiting(
        self,
        state: AgentState,
        *,
        wake_condition: WakeCondition,
        result_summary: str | None | object = ...,
        explain: str | None | object = ...,
    ) -> None:
        state.status = AgentStateStatus.WAITING
        state.wake_condition = wake_condition
        _apply_fields(state, result_summary=result_summary, explain=explain)
        await self._save(state)

    async def mark_idle(
        self,
        state: AgentState,
        *,
        result_summary: str | None | object = ...,
        explain: str | None | object = ...,
    ) -> None:
        state.status = AgentStateStatus.IDLE
        state.pending_input = None
        state.wake_condition = None
        _apply_fields(state, result_summary=result_summary, explain=explain)
        await self._save(state, notify=True)

    async def mark_queued(
        self,
        state: AgentState,
        *,
        pending_input: UserInput,
    ) -> None:
        state.status = AgentStateStatus.QUEUED
        state.pending_input = pending_input
        state.wake_condition = None
        state.explain = None
        await self._save(state)

    async def mark_completed(
        self,
        state: AgentState,
        *,
        result_summary: str | None | object = ...,
    ) -> None:
        state.status = AgentStateStatus.COMPLETED
        state.wake_condition = None
        state.pending_input = None
        _apply_fields(state, result_summary=result_summary)
        await self._save(state, notify=True)

    async def mark_failed(self, state: AgentState, reason: str) -> None:
        state.status = AgentStateStatus.FAILED
        state.result_summary = reason
        state.wake_condition = None
        await self._save(state, notify=True)


__all__ = ["SchedulerStateOps"]
