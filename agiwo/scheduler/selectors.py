"""Pure selector helpers for scheduler orchestration."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from agiwo.scheduler.models import AgentState, AgentStateStatus, PendingEvent


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def select_timed_out_states(
    states: list[AgentState],
    *,
    now: datetime,
) -> list[AgentState]:
    return [
        state
        for state in states
        if state.status == AgentStateStatus.WAITING
        and state.wake_condition is not None
        and state.wake_condition.is_timed_out(now)
    ]


def select_ready_waiting_states(
    states: list[AgentState],
    *,
    now: datetime,
) -> list[AgentState]:
    return [
        state
        for state in states
        if state.status == AgentStateStatus.WAITING
        and state.wake_condition is not None
        and state.wake_condition.is_satisfied(now)
    ]


def select_debounced_event_targets(
    events: list[PendingEvent],
    *,
    min_count: int,
    max_wait_seconds: float,
    now: datetime,
) -> list[tuple[str, str]]:
    cutoff = _ensure_utc(now - timedelta(seconds=max_wait_seconds))
    grouped: dict[tuple[str, str], list[PendingEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.target_agent_id, event.session_id)].append(event)

    selected: list[tuple[str, str]] = []
    for key, group in grouped.items():
        if len(group) >= min_count:
            selected.append(key)
            continue
        oldest = min(group, key=lambda item: item.created_at)
        if _ensure_utc(oldest.created_at) <= cutoff:
            selected.append(key)
    return selected


def select_unpropagated_children(states: list[AgentState]) -> list[AgentState]:
    return [
        state
        for state in states
        if state.is_child
        and state.status in (AgentStateStatus.COMPLETED, AgentStateStatus.FAILED)
        and not state.signal_propagated
    ]


__all__ = [
    "select_debounced_event_targets",
    "select_ready_waiting_states",
    "select_timed_out_states",
    "select_unpropagated_children",
]
