"""
In-memory trace storage implementation.
"""

from typing import Any

from agiwo.observability.base import BaseTraceStorage, TraceQuery
from agiwo.observability.trace import Trace


class InMemoryTraceStorage(BaseTraceStorage):
    """
    In-memory implementation of BaseTraceStorage.

    Simple memory-only storage with ring buffer for real-time access.
    No persistence - traces are lost when process exits.
    """

    def __init__(self, buffer_size: int = 200) -> None:
        self._initialize_runtime_state(buffer_size=buffer_size)
        self._initialized = True  # No async init needed

    async def initialize(self) -> None:
        """No-op for in-memory storage."""
        pass

    async def save_trace(self, trace: Trace) -> None:
        """Save trace to in-memory buffer."""
        self._append_to_buffer(trace)
        await self._notify_subscribers(trace)

    async def get_trace(self, trace_id: str) -> Trace | None:
        """Get a trace by ID from buffer."""
        return self._get_buffered_trace(trace_id)

    async def query_traces(
        self,
        query: TraceQuery | dict[str, Any],
    ) -> list[Trace]:
        """Query traces from buffer."""
        return self._query_buffer(query)

    async def close(self) -> None:
        """Clear buffer and subscribers."""
        self._buffer.clear()
        self._subscribers.clear()


__all__ = ["InMemoryTraceStorage"]
