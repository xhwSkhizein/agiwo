import asyncio
from typing import AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from agiwo.agent.schema import StreamEvent


class StreamChannel:
    """
    Event streaming channel for Runnable execution.

    StreamChannel is a simple wrapper around asyncio.Queue that provides:
    - write(): Put an event into the channel
    - read(): Async iterate over events until closed
    - close(): Signal that no more events will be written

    Thread-safe and supports multiple concurrent writers.
    Single-consumer: read() can only be claimed once.
    """

    # Sentinel value to signal end of stream
    _SENTINEL = object()

    def __init__(self, maxsize: int = 0) -> None:
        """
        Initialize StreamChannel.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        self._reader_claimed = False

    async def write(self, event: "StreamEvent") -> None:
        """
        Write an event to the channel.

        Args:
            event: StreamEvent to write

        Raises:
            RuntimeError: If channel is already closed
        """
        if self._closed:
            # Silently ignore writes after close (graceful degradation)
            return
        await self._queue.put(event)

    def write_nowait(self, event: "StreamEvent") -> None:
        """
        Write an event without waiting (non-blocking).

        Use this for synchronous contexts where await is not possible.
        """
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop event if queue is full (shouldn't happen with unlimited queue)
            pass

    async def close(self) -> None:
        """
        Close the channel, signaling no more events will be written.

        This puts a sentinel value in the queue to stop readers.
        """
        if self._closed:
            return
        self._closed = True
        await self._queue.put(self._SENTINEL)

    async def read(self) -> AsyncIterator["StreamEvent"]:
        """
        Read events from the channel until closed.

        Yields:
            StreamEvent: Events written to the channel
        """
        if self._reader_claimed:
            raise RuntimeError("channel_read_already_claimed")
        self._reader_claimed = True
        while True:
            item = await self._queue.get()
            if item is self._SENTINEL:
                # Re-put sentinel for other readers (if any)
                await self._queue.put(self._SENTINEL)
                break
            yield item

    @property
    def closed(self) -> bool:
        """Check if channel is closed."""
        return self._closed

    def __repr__(self) -> str:
        return f"StreamChannel(closed={self._closed}, qsize={self._queue.qsize()})"


__all__ = ["StreamChannel"]
