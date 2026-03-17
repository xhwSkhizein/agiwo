"""Console-owned wrapper that adds real-time pub/sub on top of any BaseTraceStorage."""

import asyncio
from typing import Any

from agiwo.observability.base import BaseTraceStorage, TraceQuery
from agiwo.observability.trace import Trace


class NotifyingTraceStorage(BaseTraceStorage):
    """Decorator that notifies subscribers whenever a trace is saved.

    SDK storage stays pure (save/get/query/close).
    Real-time notification is a Console concern for the trace SSE endpoint.
    """

    def __init__(
        self,
        inner: BaseTraceStorage,
        queue_maxsize: int = 100,
    ) -> None:
        self._inner = inner
        self._subscribers: list[asyncio.Queue[Trace]] = []
        self._queue_maxsize = queue_maxsize

    def subscribe(self) -> asyncio.Queue[Trace]:
        queue: asyncio.Queue[Trace] = asyncio.Queue(maxsize=self._queue_maxsize)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Trace]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def initialize(self) -> None:
        await self._inner.initialize()

    async def save_trace(self, trace: Trace) -> None:
        await self._inner.save_trace(trace)
        for queue in self._subscribers:
            try:
                queue.put_nowait(trace)
            except asyncio.QueueFull:
                pass

    async def get_trace(self, trace_id: str) -> Trace | None:
        return await self._inner.get_trace(trace_id)

    async def query_traces(self, query: TraceQuery | dict[str, Any]) -> list[Trace]:
        return await self._inner.query_traces(query)

    async def close(self) -> None:
        self._subscribers.clear()
        await self._inner.close()


__all__ = ["NotifyingTraceStorage"]
