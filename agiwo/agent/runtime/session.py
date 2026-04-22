"""Session-scoped runtime state shared by root and child runs."""

import asyncio
from collections.abc import AsyncIterator

from agiwo.agent.models.log import RunLogEntry
from agiwo.agent.models.run import CompactMetadata
from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.storage.base import (
    RunLogStorage,
)
from agiwo.agent.models.stream import AgentStreamItem
from agiwo.agent.trace_writer import AgentTraceCollector
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SessionRuntime:
    """Session-scoped runtime state shared by root and child runs."""

    _SENTINEL = object()
    _MAX_SUBSCRIBER_QUEUE_SIZE = 256

    def __init__(
        self,
        *,
        session_id: str,
        run_log_storage: RunLogStorage,
        trace_runtime: AgentTraceCollector | None = None,
        abort_signal: AbortSignal | None = None,
        steering_queue: asyncio.Queue[object] | None = None,
    ) -> None:
        self.session_id = session_id
        self.run_log_storage = run_log_storage
        self.trace_runtime = trace_runtime
        self.abort_signal = abort_signal or AbortSignal()
        self.steering_queue = steering_queue or asyncio.Queue()
        self._subscribers: set[asyncio.Queue[AgentStreamItem | object]] = set()
        self._closed = False

    # ------------------------------------------------------------------
    # Storage convenience methods
    # ------------------------------------------------------------------

    async def allocate_sequence(self) -> int:
        return await self.run_log_storage.allocate_sequence(self.session_id)

    async def append_run_log_entries(self, entries: list[RunLogEntry]) -> None:
        await self.run_log_storage.append_entries(entries)
        if self.trace_runtime is not None:
            await self.trace_runtime.on_run_log_entries(entries)

    async def list_run_log_entries(
        self,
        *,
        run_id: str | None = None,
        agent_id: str | None = None,
        after_sequence: int | None = None,
        limit: int = 1000,
    ) -> list[RunLogEntry]:
        return await self.run_log_storage.list_entries(
            session_id=self.session_id,
            run_id=run_id,
            agent_id=agent_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    async def get_latest_compact_metadata(
        self, agent_id: str
    ) -> CompactMetadata | None:
        """Retrieve the latest compact metadata for the given agent."""
        return await self.run_log_storage.get_latest_compact_metadata(
            self.session_id, agent_id
        )

    def subscribe(self) -> AsyncIterator[AgentStreamItem]:
        queue: asyncio.Queue[AgentStreamItem | object] = asyncio.Queue(
            maxsize=self._MAX_SUBSCRIBER_QUEUE_SIZE
        )
        if self._closed:
            queue.put_nowait(self._SENTINEL)
        self._subscribers.add(queue)

        async def _iterator() -> AsyncIterator[AgentStreamItem]:
            try:
                while True:
                    item = await queue.get()
                    if item is self._SENTINEL:
                        break
                    yield item
            finally:
                self._subscribers.discard(queue)

        return _iterator()

    async def enqueue_steer(self, user_input: UserInput) -> bool:
        if self._closed:
            return False
        message = UserMessage.from_value(user_input)
        if not message.has_content():
            return False
        await self.steering_queue.put(message)
        return True

    async def publish(self, item: AgentStreamItem) -> None:
        if self._closed:
            return
        for subscriber in list(self._subscribers):
            try:
                subscriber.put_nowait(item)
            except asyncio.QueueFull:
                self._subscribers.discard(subscriber)
                try:
                    subscriber.put_nowait(self._SENTINEL)
                except asyncio.QueueFull:
                    pass
                logger.warning("subscriber_dropped_slow_consumer")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.trace_runtime is not None:
            await self.trace_runtime.stop()
        for subscriber in list(self._subscribers):
            try:
                subscriber.put_nowait(self._SENTINEL)
            except asyncio.QueueFull:
                pass


__all__ = ["SessionRuntime"]
