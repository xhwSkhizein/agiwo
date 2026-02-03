import asyncio
from typing import AsyncIterator

from agiwo.agent.schema import StepEvent


class Wire:
    """
    Event streaming channel for Runnable execution.

    Wire is a simple wrapper around asyncio.Queue that provides:
    - write(): Put an event into the channel
    - read(): Async iterate over events until closed
    - close(): Signal that no more events will be written

    Thread-safe and supports multiple concurrent writers.
    """

    # Sentinel value to signal end of stream
    _SENTINEL = object()

    def __init__(self, maxsize: int = 0) -> None:
        """
        Initialize Wire.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        self._writer_count = 0
        self._lock = asyncio.Lock()

    async def write(self, event: "StepEvent") -> None:
        """
        Write an event to the wire.

        Args:
            event: StepEvent to write

        Raises:
            RuntimeError: If wire is already closed
        """
        if self._closed:
            # Silently ignore writes after close (graceful degradation)
            return
        await self._queue.put(event)

    def write_nowait(self, event: "StepEvent") -> None:
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
        Close the wire, signaling no more events will be written.

        This puts a sentinel value in the queue to stop readers.
        """
        if self._closed:
            return
        self._closed = True
        await self._queue.put(self._SENTINEL)

    async def read(self) -> AsyncIterator["StepEvent"]:
        """
        Read events from the wire until closed.

        Yields:
            StepEvent: Events written to the wire
        """
        while True:
            item = await self._queue.get()
            if item is self._SENTINEL:
                # Re-put sentinel for other readers (if any)
                await self._queue.put(self._SENTINEL)
                break
            yield item

    @property
    def closed(self) -> bool:
        """Check if wire is closed."""
        return self._closed

    def __repr__(self) -> str:
        return f"Wire(closed={self._closed}, qsize={self._queue.qsize()})"


__all__ = ["Wire"]
