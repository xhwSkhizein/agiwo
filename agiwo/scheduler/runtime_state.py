"""Private scheduler runtime containers and tick helper functions."""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol

from agiwo.agent import (
    Agent,
    AgentStreamItem,
    ChannelContext,
    ContentPart,
    ContentType,
    RunOutput,
    UserInput,
    UserMessage,
)
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
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
    """Merge a pending root input with queued USER_HINT events.

    The result preserves structured content:

    - ``ContentPart`` entries from the pending input and each hint ``UserMessage``
      are combined (text parts are flattened into a single "Additional queued
      user input" block; non-text parts are appended intact so attachments
      survive into the next run).
    - ``ChannelContext`` from the pending input takes precedence; otherwise we
      fall back to the first USER_HINT with a ``ChannelContext``.
    """
    hint_messages = _extract_hint_messages(events)
    if not hint_messages and pending_input is None:
        return None
    if not hint_messages:
        return pending_input

    base_message = (
        UserMessage.from_value(pending_input) if pending_input is not None else None
    )
    base_text_parts, base_media_parts = _split_content_parts(base_message)

    hint_text_fragments: list[str] = []
    hint_media_parts: list[ContentPart] = []
    hint_context: ChannelContext | None = None
    for hint_message in hint_messages:
        text_parts, media_parts = _split_content_parts(hint_message)
        for text_part in text_parts:
            if text_part.text and text_part.text.strip():
                hint_text_fragments.append(text_part.text.strip())
        hint_media_parts.extend(media_parts)
        if hint_context is None and hint_message.context is not None:
            hint_context = hint_message.context

    if not hint_text_fragments and not hint_media_parts:
        return pending_input

    merged_parts: list[ContentPart] = []
    base_text = "\n".join(part.text for part in base_text_parts if part.text).strip()
    hint_block = "\n".join(f"- {fragment}" for fragment in hint_text_fragments)
    if base_text and hint_block:
        merged_parts.append(
            ContentPart(
                type=ContentType.TEXT,
                text=f"{base_text}\n\nAdditional queued user input:\n{hint_block}",
            )
        )
    elif hint_block:
        merged_parts.append(
            ContentPart(
                type=ContentType.TEXT,
                text=f"Additional queued user input:\n{hint_block}",
            )
        )
    elif base_text:
        merged_parts.append(ContentPart(type=ContentType.TEXT, text=base_text))

    merged_parts.extend(base_media_parts)
    merged_parts.extend(hint_media_parts)

    context = None
    if base_message is not None and base_message.context is not None:
        context = base_message.context
    elif hint_context is not None:
        context = hint_context

    return UserMessage(content=merged_parts, context=context)


def _extract_hint_messages(
    events: tuple[PendingEvent, ...],
) -> list[UserMessage]:
    messages: list[UserMessage] = []
    for event in events:
        payload = event.get_payload_user_hint()
        if payload is None:
            continue
        stored = payload.user_input
        decoded = UserMessage.from_storage_value(stored)
        if decoded is None:
            continue
        message = UserMessage.from_value(decoded)
        if message.has_content():
            messages.append(message)
    return messages


def _split_content_parts(
    message: UserMessage | None,
) -> tuple[list[ContentPart], list[ContentPart]]:
    if message is None:
        return [], []
    text_parts: list[ContentPart] = []
    media_parts: list[ContentPart] = []
    for part in message.content:
        if part.type == ContentType.TEXT:
            text_parts.append(part)
        else:
            media_parts.append(part)
    return text_parts, media_parts


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
    """Process-local runtime state only.

    ``agents`` holds the scheduler-managed runtime agents (clones of the
    caller-supplied canonical Agent with scheduler system tools injected).
    ``canonical_agents`` records the last canonical Agent *identity* mapped
    to each state_id so the scheduler can decide whether a ``submit`` /
    ``enqueue_input`` call should reuse the cached runtime agent or rebind.
    ``state_locks`` holds per-state_id locks to prevent concurrent submit/enqueue
    operations from racing on the same agent.
    """

    agents: dict[str, Agent] = field(default_factory=dict)
    canonical_agents: dict[str, Agent] = field(default_factory=dict)
    execution_handles: dict[str, ExecutionHandleLike] = field(default_factory=dict)
    abort_signals: dict[str, AbortSignal] = field(default_factory=dict)
    waiters: dict[str, set[asyncio.Event]] = field(default_factory=dict)
    dispatched: set[str] = field(default_factory=set)
    active_tasks: set[asyncio.Task] = field(default_factory=set)
    stream_channels: dict[str, StreamChannelState] = field(default_factory=dict)
    state_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
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
