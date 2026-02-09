"""
Base interface for trace storage.
"""

import asyncio
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


class BaseTraceStorage(ABC):
    """
    Abstract base class for trace storage.

    All trace store implementations must extend this class.
    """

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

    @abstractmethod
    def get_recent(self, limit: int = 20) -> list[Trace]:
        """Get recent traces from in-memory buffer."""
        ...

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to real-time trace updates. Override if supported."""
        raise NotImplementedError

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from updates. Override if supported."""
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """Close connections and release resources."""
        ...


__all__ = ["BaseTraceStorage", "TraceQuery"]
