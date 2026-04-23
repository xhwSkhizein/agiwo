"""
In-memory trace storage implementation.
"""

from collections import deque
from typing import Any

from agiwo.observability.base import (
    BaseTraceStorage,
    TraceQuery,
    _coerce_query,
    _matches_query,
)
from agiwo.observability.trace import Trace


class InMemoryTraceStorage(BaseTraceStorage):
    """
    In-memory implementation of BaseTraceStorage.

    Simple memory-only storage. No persistence - traces are lost when process exits.
    """

    def __init__(self, buffer_size: int = 200) -> None:
        self._traces: deque[Trace] = deque(maxlen=buffer_size)

    async def save_trace(self, trace: Trace) -> None:
        existing = next(
            (
                index
                for index, item in enumerate(self._traces)
                if item.trace_id == trace.trace_id
            ),
            None,
        )
        if existing is not None:
            del self._traces[existing]
        self._traces.append(trace)

    async def get_trace(self, trace_id: str) -> Trace | None:
        for trace in reversed(self._traces):
            if trace.trace_id == trace_id:
                return trace
        return None

    async def query_traces(
        self,
        query: TraceQuery | dict[str, Any],
    ) -> list[Trace]:
        coerced = _coerce_query(query)
        results = [
            trace for trace in reversed(self._traces) if _matches_query(trace, coerced)
        ]
        start = coerced.offset
        end = start + coerced.limit
        return results[start:end]

    async def close(self) -> None:
        self._traces.clear()


__all__ = ["InMemoryTraceStorage"]
