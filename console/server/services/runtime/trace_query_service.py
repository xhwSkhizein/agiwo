"""Trace-backed query facade for console observability views."""

from dataclasses import dataclass

from agiwo.observability.base import BaseTraceStorage, TraceQuery
from agiwo.observability.trace import Trace

from server.models.session import PageSlice


@dataclass(slots=True)
class TraceQueryService:
    trace_storage: BaseTraceStorage

    async def list_traces(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        status: str | None = None,
        limit: int,
        offset: int,
    ) -> PageSlice[Trace]:
        query = TraceQuery(
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            status=status,
            limit=limit + 1,
            offset=offset,
        )
        traces = await self.trace_storage.query_traces(query)
        has_more = len(traces) > limit
        return PageSlice(
            items=traces[:limit],
            limit=limit,
            offset=offset,
            has_more=has_more,
            total=None,
        )

    async def get_trace(self, trace_id: str) -> Trace | None:
        return await self.trace_storage.get_trace(trace_id)

    async def list_session_recent_traces(
        self,
        session_id: str,
        *,
        limit: int = 5,
    ) -> list[Trace]:
        page = await self.list_traces(session_id=session_id, limit=limit, offset=0)
        return page.items


__all__ = ["TraceQueryService"]
