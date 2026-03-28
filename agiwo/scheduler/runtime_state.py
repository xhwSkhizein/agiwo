"""Private scheduler runtime containers and tick helper functions."""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

from agiwo.agent import Agent, AgentStreamItem, RunOutput, UserInput, UserMessage
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
)
from agiwo.scheduler.store.base import AgentStateStorage
from agiwo.scheduler.stream import StreamChannelState
from agiwo.utils.abort_signal import AbortSignal


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def group_events(
    events: list[PendingEvent],
) -> dict[tuple[str, str], list[PendingEvent]]:
    grouped: dict[tuple[str, str], list[PendingEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.target_agent_id, event.session_id)].append(event)
    return grouped


def select_debounced_event_targets(
    events: list[PendingEvent],
    *,
    min_count: int,
    max_wait_seconds: float,
    now: datetime,
) -> set[tuple[str, str]]:
    cutoff = ensure_utc(now - timedelta(seconds=max_wait_seconds))
    grouped = group_events(events)
    selected: set[tuple[str, str]] = set()
    for key, group in grouped.items():
        if len(group) >= min_count:
            selected.add(key)
            continue
        oldest = min(group, key=lambda item: item.created_at)
        if ensure_utc(oldest.created_at) <= cutoff:
            selected.add(key)
    return selected


def build_mailbox_input(
    pending_input: UserInput | None,
    events: tuple[PendingEvent, ...],
) -> UserInput | None:
    if not events:
        return pending_input

    base_text = ""
    if pending_input is not None:
        base_text = UserMessage.from_value(pending_input).extract_text() or str(
            pending_input
        )

    hints = [
        event.payload.get("hint", "").strip()
        for event in events
        if event.event_type == SchedulerEventType.USER_HINT
        and event.payload.get("hint", "").strip()
    ]
    if not hints:
        return pending_input

    hint_block = "\n".join(f"- {hint}" for hint in hints)
    if not base_text:
        return f"Additional queued user input:\n{hint_block}"
    return f"{base_text}\n\nAdditional queued user input:\n{hint_block}"


async def list_all_states(
    store: AgentStateStorage,
    *,
    statuses: tuple[AgentStateStatus, ...] | None = None,
    page_size: int = 1000,
) -> list[AgentState]:
    states: list[AgentState] = []
    offset = 0
    while True:
        batch = await store.list_states(
            statuses=statuses, limit=page_size, offset=offset
        )
        states.extend(batch)
        if len(batch) < page_size:
            return states
        offset += page_size


class ExecutionHandleLike(Protocol):
    run_id: str
    session_id: str

    def stream(self) -> AsyncIterator[AgentStreamItem]: ...

    async def wait(self) -> RunOutput: ...

    async def steer(self, user_input: UserInput) -> bool: ...

    def cancel(self, reason: str | None = None) -> None: ...


@dataclass
class RuntimeState:
    """Process-local runtime state only."""

    agents: dict[str, Agent] = field(default_factory=dict)
    execution_handles: dict[str, ExecutionHandleLike] = field(default_factory=dict)
    abort_signals: dict[str, AbortSignal] = field(default_factory=dict)
    waiters: dict[str, set[asyncio.Event]] = field(default_factory=dict)
    dispatched: set[str] = field(default_factory=set)
    active_tasks: set[asyncio.Task] = field(default_factory=set)
    stream_channels: dict[str, StreamChannelState] = field(default_factory=dict)
    nudge: asyncio.Event = field(default_factory=asyncio.Event)
    shutdown_requested: set[str] = field(default_factory=set)


__all__ = [
    "ExecutionHandleLike",
    "RuntimeState",
    "build_mailbox_input",
    "ensure_utc",
    "group_events",
    "list_all_states",
    "select_debounced_event_targets",
]
