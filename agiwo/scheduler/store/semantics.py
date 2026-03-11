"""Shared state/event semantics used across scheduler storage backends."""

from datetime import datetime, timedelta, timezone

from agiwo.scheduler.models import AgentState, PendingEvent, SchedulerEventType

RECENT_STEPS_MAX = 10


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def apply_status_update(
    state: AgentState,
    *,
    status,
    wake_condition,
    result_summary,
    explain,
    last_activity_at,
    recent_steps,
    now: datetime,
) -> None:
    """Apply sentinel-aware status updates to an in-memory AgentState."""
    state.status = status
    state.updated_at = now
    if wake_condition is not ...:
        state.wake_condition = wake_condition
    if result_summary is not ...:
        state.result_summary = result_summary
    if explain is not ...:
        state.explain = explain
    if last_activity_at is not ...:
        state.last_activity_at = last_activity_at
    if recent_steps is not ...:
        state.recent_steps = recent_steps


def is_wakeable_state(state: AgentState, now: datetime) -> bool:
    return (
        state.status.value == "sleeping"
        and state.wake_condition is not None
        and state.wake_condition.is_satisfied(now)
    )


def is_timed_out_state(state: AgentState, now: datetime) -> bool:
    return (
        state.status.value == "sleeping"
        and state.wake_condition is not None
        and state.wake_condition.is_timed_out(now)
    )


def append_recent_steps(existing: list[dict] | None, step: dict) -> list[dict]:
    merged = (existing or []) + [step]
    return merged[-RECENT_STEPS_MAX:]


def find_debounced_agent_sessions(
    events: list[PendingEvent],
    *,
    min_count: int,
    max_wait_seconds: float,
    now: datetime,
) -> list[tuple[str, str]]:
    cutoff = _ensure_utc(now - timedelta(seconds=max_wait_seconds))

    groups: dict[tuple[str, str], list[PendingEvent]] = {}
    for event in events:
        key = (event.target_agent_id, event.session_id)
        groups.setdefault(key, []).append(event)

    result: list[tuple[str, str]] = []
    for (agent_id, session_id), grouped_events in groups.items():
        if len(grouped_events) >= min_count:
            result.append((agent_id, session_id))
            continue

        oldest = min(grouped_events, key=lambda e: e.created_at)
        oldest_ts = _ensure_utc(oldest.created_at)
        if oldest_ts <= cutoff:
            result.append((agent_id, session_id))

    return result


def has_recent_health_warning_event(
    events: list[PendingEvent],
    *,
    target_agent_id: str,
    source_agent_id: str,
    within_seconds: float,
    now: datetime,
) -> bool:
    cutoff = _ensure_utc(now - timedelta(seconds=within_seconds))

    for event in events:
        if (
            event.target_agent_id == target_agent_id
            and event.source_agent_id == source_agent_id
            and event.event_type == SchedulerEventType.HEALTH_WARNING
        ):
            if _ensure_utc(event.created_at) >= cutoff:
                return True

    return False
