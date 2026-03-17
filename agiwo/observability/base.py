"""
Base interface for trace storage.
"""

from abc import ABC, abstractmethod
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


def _coerce_query(query: TraceQuery | dict[str, Any]) -> TraceQuery:
    if isinstance(query, TraceQuery):
        return query
    return TraceQuery(**query)


def _matches_query(trace: Trace, query: TraceQuery) -> bool:
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


class BaseTraceStorage(ABC):
    """
    Abstract base class for trace storage.

    All trace store implementations must extend this class.
    """

    async def initialize(self) -> None:
        """Initialize the store (e.g. create connections, tables)."""
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

    @abstractmethod
    async def close(self) -> None:
        """Close connections and release resources."""
        ...


__all__ = ["BaseTraceStorage", "TraceQuery", "_coerce_query", "_matches_query"]
