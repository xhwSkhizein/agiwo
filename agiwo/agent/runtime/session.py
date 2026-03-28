"""Session-scoped runtime state shared by root and child runs."""

import asyncio
from collections.abc import AsyncIterator

from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.session import SessionStorage
from agiwo.agent.models.stream import AgentStreamItem
from agiwo.agent.trace_writer import AgentTraceCollector
from agiwo.utils.abort_signal import AbortSignal


class SessionRuntime:
    """Session-scoped runtime state shared by root and child runs."""

    _SENTINEL = object()

    def __init__(
        self,
        *,
        session_id: str,
        run_step_storage: RunStepStorage,
        session_storage: SessionStorage,
        trace_runtime: AgentTraceCollector | None = None,
        abort_signal: AbortSignal | None = None,
        steering_queue: asyncio.Queue[object] | None = None,
    ) -> None:
        self.session_id = session_id
        self.run_step_storage = run_step_storage
        self.session_storage = session_storage
        self.trace_runtime = trace_runtime
        self.abort_signal = abort_signal or AbortSignal()
        self.steering_queue = steering_queue or asyncio.Queue()
        self._subscribers: set[asyncio.Queue[AgentStreamItem | object]] = set()
        self._closed = False

    async def allocate_sequence(self) -> int:
        return await self.run_step_storage.allocate_sequence(self.session_id)

    def subscribe(self) -> AsyncIterator[AgentStreamItem]:
        queue: asyncio.Queue[AgentStreamItem | object] = asyncio.Queue()
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
            await subscriber.put(item)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.trace_runtime is not None:
            await self.trace_runtime.stop()
        for subscriber in list(self._subscribers):
            await subscriber.put(self._SENTINEL)


__all__ = ["SessionRuntime"]
