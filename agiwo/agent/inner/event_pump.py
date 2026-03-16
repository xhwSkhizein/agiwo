import asyncio
from collections.abc import AsyncIterator

from agiwo.agent.hooks import AgentHooks
from agiwo.agent.inner.storage_sink import StorageSink
from agiwo.agent.runtime import StreamEvent
from agiwo.agent.trace import AgentTraceCollector
from agiwo.agent.stream_channel import StreamChannel


class EventPump:
    """Single-consumer channel pump with explicit downstream consumers."""

    _SENTINEL = object()

    def __init__(
        self,
        *,
        channel: StreamChannel,
        storage_sink: StorageSink,
        hooks: AgentHooks,
        trace_collector: AgentTraceCollector | None = None,
    ) -> None:
        self._channel = channel
        self._storage_sink = storage_sink
        self._hooks = hooks
        self._trace_collector = trace_collector
        self._task: asyncio.Task[None] | None = None
        self._subscribers: set[asyncio.Queue[StreamEvent | object]] = set()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def wait(self) -> None:
        if self._task is not None:
            await self._task

    def subscribe(self) -> AsyncIterator[StreamEvent]:
        queue: asyncio.Queue[StreamEvent | object] = asyncio.Queue()
        self._subscribers.add(queue)

        async def _iterator() -> AsyncIterator[StreamEvent]:
            try:
                while True:
                    item = await queue.get()
                    if item is self._SENTINEL:
                        break
                    yield item
            finally:
                self._subscribers.discard(queue)

        return _iterator()

    async def _run(self) -> None:
        try:
            async for event in self._channel.read():
                await self._storage_sink.consume(event)
                if self._trace_collector is not None:
                    await self._trace_collector.collect(event)
                if self._hooks.on_event is not None:
                    await self._hooks.on_event(event)
                for subscriber in list(self._subscribers):
                    await subscriber.put(event)
        except Exception as error:
            if self._trace_collector is not None:
                self._trace_collector.fail(error)
            raise
        finally:
            if self._trace_collector is not None:
                await self._trace_collector.stop()
            for subscriber in list(self._subscribers):
                await subscriber.put(self._SENTINEL)


__all__ = ["EventPump"]
