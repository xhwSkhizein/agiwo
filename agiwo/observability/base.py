"""
Base interface for trace storage.
"""

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agiwo.observability.trace import SpanStatus, Trace


class TraceQuery(BaseModel):
    """Trace query parameters"""

    agent_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    status: SpanStatus | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    min_duration_ms: float | None = None
    max_duration_ms: float | None = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class BaseTraceStorage(ABC):
    """
    Abstract base class for trace storage.

    All trace store implementations must extend this class.
    """

    def _initialize_runtime_state(
        self,
        *,
        buffer_size: int,
        subscriber_queue_maxsize: int = 100,
    ) -> None:
        self._buffer: deque[Trace] = deque(maxlen=buffer_size)
        self._subscribers: list[asyncio.Queue[Trace]] = []
        self._subscriber_queue_maxsize = subscriber_queue_maxsize

    def _append_to_buffer(self, trace: Trace) -> None:
        self._buffer.append(trace)

    def _get_buffered_trace(self, trace_id: str) -> Trace | None:
        for trace in self._buffer:
            if trace.trace_id == trace_id:
                return trace
        return None

    def _coerce_query(self, query: TraceQuery | dict[str, Any]) -> TraceQuery:
        if isinstance(query, TraceQuery):
            return query
        return TraceQuery(**query)

    def _matches_query(self, trace: Trace, query: TraceQuery) -> bool:
        if query.agent_id and trace.agent_id != query.agent_id:
            return False
        if query.session_id and trace.session_id != query.session_id:
            return False
        if query.user_id and trace.user_id != query.user_id:
            return False
        if query.status and trace.status != query.status:
            return False

        start_time = trace.start_time
        if query.start_time and (start_time is None or start_time < query.start_time):
            return False
        if query.end_time and (start_time is None or start_time > query.end_time):
            return False

        duration_ms = trace.duration_ms
        if (
            query.min_duration_ms is not None
            and (duration_ms is None or duration_ms < query.min_duration_ms)
        ):
            return False
        if (
            query.max_duration_ms is not None
            and (duration_ms is None or duration_ms > query.max_duration_ms)
        ):
            return False
        return True

    def _query_buffer(self, query: TraceQuery | dict[str, Any]) -> list[Trace]:
        coerced = self._coerce_query(query)
        results = [
            trace
            for trace in reversed(self._buffer)
            if self._matches_query(trace, coerced)
        ]
        start = coerced.offset
        end = start + coerced.limit
        return results[start:end]

    async def initialize(self) -> None:
        """Initialize the store (e.g. create connections, tables). Called internally on first use."""
        pass

    @abstractmethod
    async def save_trace(self, trace: Trace) -> None:
        """Save or update a trace."""
        ...

    @abstractmethod
    async def get_trace(self, trace_id: str) -> Trace | None:
        """Get a single trace by ID."""
        ...

    @abstractmethod
    async def query_traces(self, query: TraceQuery | dict[str, Any]) -> list[Trace]:
        """Query traces with filters."""
        ...

    def get_recent(self, limit: int = 20) -> list[Trace]:
        """Get recent traces from in-memory buffer."""
        return list(reversed(list(self._buffer)))[:limit]

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to real-time trace updates."""
        queue: asyncio.Queue[Trace] = asyncio.Queue(
            maxsize=self._subscriber_queue_maxsize
        )
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from updates."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    async def _notify_subscribers(self, trace: Trace) -> None:
        for queue in self._subscribers:
            try:
                queue.put_nowait(trace)
            except asyncio.QueueFull:
                pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and release resources."""
        ...


__all__ = ["BaseTraceStorage", "TraceQuery"]
