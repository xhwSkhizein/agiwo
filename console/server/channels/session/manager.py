"""
Session message batching and debounce scheduling.

Collects incoming messages per session key, applies debounce and max-batch-window
logic, then fires a callback when the batch is ready for execution.
"""

import asyncio
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from agiwo.utils.logging import get_logger

from server.channels.session.models import BatchContext, InboundMessage

logger = get_logger(__name__)

OnBatchReady = Callable[
    [str, BatchContext, list[InboundMessage]],
    Coroutine[Any, Any, None],
]


@dataclass
class _SessionState:
    pending_messages: list[InboundMessage] = field(default_factory=list)
    first_pending_at_ms: int | None = None
    flush_task: asyncio.Task[None] | None = None
    running: bool = False
    latest_context: BatchContext | None = None


class SessionManager:
    def __init__(
        self,
        *,
        on_batch_ready: OnBatchReady,
        debounce_ms: int,
        max_batch_window_ms: int,
    ) -> None:
        self._on_batch_ready = on_batch_ready
        self._debounce_ms = debounce_ms
        self._max_batch_window_ms = max_batch_window_ms

        self._states: dict[str, _SessionState] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def session_count(self) -> int:
        return len(self._states)

    async def enqueue(
        self,
        chat_context_scope_id: str,
        message: InboundMessage,
        context: BatchContext,
    ) -> None:
        lock = self._get_lock(chat_context_scope_id)
        now_ms = int(time.time() * 1000)

        async with lock:
            state = self._states.setdefault(chat_context_scope_id, _SessionState())
            state.pending_messages.append(message)
            state.latest_context = context
            if state.first_pending_at_ms is None:
                state.first_pending_at_ms = now_ms
            self._reschedule_flush_locked(chat_context_scope_id, state, now_ms)

    def reset_chat_context(self, chat_context_scope_id: str) -> None:
        self._states.pop(chat_context_scope_id, None)

    async def close(self) -> None:
        flush_tasks: list[asyncio.Task[None]] = []
        for state in self._states.values():
            task = state.flush_task
            if task is None or task.done():
                continue
            task.cancel()
            flush_tasks.append(task)

        if flush_tasks:
            await asyncio.gather(*flush_tasks, return_exceptions=True)

    # -- Internal ------------------------------------------------------------

    def _get_lock(self, chat_context_scope_id: str) -> asyncio.Lock:
        lock = self._locks.get(chat_context_scope_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[chat_context_scope_id] = lock
        return lock

    def _reschedule_flush_locked(
        self,
        chat_context_scope_id: str,
        state: _SessionState,
        now_ms: int,
    ) -> None:
        if state.running or not state.pending_messages:
            return

        existing_task = state.flush_task
        if existing_task is not None and not existing_task.done():
            existing_task.cancel()

        first_pending_at = state.first_pending_at_ms or now_ms
        elapsed_ms = max(0, now_ms - first_pending_at)
        remaining_window_ms = max(0, self._max_batch_window_ms - elapsed_ms)
        delay_ms = min(self._debounce_ms, remaining_window_ms)

        state.flush_task = asyncio.create_task(
            self._flush_after_delay(chat_context_scope_id, delay_ms)
        )

    async def _flush_after_delay(
        self,
        chat_context_scope_id: str,
        delay_ms: int,
    ) -> None:
        if delay_ms > 0:
            try:
                await asyncio.sleep(delay_ms / 1000)
            except asyncio.CancelledError:
                return
        await self._flush_session(chat_context_scope_id)

    async def _flush_session(self, chat_context_scope_id: str) -> None:
        lock = self._get_lock(chat_context_scope_id)

        async with lock:
            state = self._states.get(chat_context_scope_id)
            if state is None or state.running or not state.pending_messages:
                return
            if state.latest_context is None:
                state.pending_messages.clear()
                state.first_pending_at_ms = None
                return

            messages = list(state.pending_messages)
            context = state.latest_context
            state.pending_messages.clear()
            state.first_pending_at_ms = None
            state.running = True

        try:
            await self._on_batch_ready(chat_context_scope_id, context, messages)
        finally:
            async with lock:
                refreshed = self._states.get(chat_context_scope_id)
                if refreshed is None:
                    return
                refreshed.running = False
                if refreshed.pending_messages:
                    self._reschedule_flush_locked(
                        chat_context_scope_id,
                        refreshed,
                        int(time.time() * 1000),
                    )
