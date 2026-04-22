"""Run metrics aggregation — for standalone runs, traces, and agent counts."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from agiwo.agent import RunLogStorage, RunStatus, RunView
from agiwo.observability.base import BaseTraceStorage
from agiwo.observability.trace import Trace

from server.models.metrics import RunMetricsSummary
from server.services.agent_registry import AgentRegistry


RUN_METRICS_PAGE_SIZE = 500
TRACE_METRICS_PAGE_SIZE = 200
AGENT_COUNT_PAGE_SIZE = 200


@dataclass
class TraceAggregateSummary:
    trace_count: int = 0
    total_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_token_cost: float = 0.0


# ── Core accumulator ────────────────────────────────────────────────────────


def add_run_to_summary(summary: RunMetricsSummary, run: RunView) -> None:
    summary.run_count += 1
    if run.status is RunStatus.COMPLETED:
        summary.completed_run_count += 1
    metrics = run.metrics
    if metrics is None:
        return
    summary.step_count += metrics.steps_count or 0
    summary.tool_calls_count += metrics.tool_calls_count or 0
    summary.input_tokens += metrics.input_tokens or 0
    summary.output_tokens += metrics.output_tokens or 0
    summary.total_tokens += metrics.total_tokens or 0
    summary.cache_read_tokens += metrics.cache_read_tokens or 0
    summary.cache_creation_tokens += metrics.cache_creation_tokens or 0
    summary.duration_ms += metrics.duration_ms or 0.0
    summary.token_cost += metrics.token_cost or 0.0


# ── Paginated run iteration ─────────────────────────────────────────────────


async def iter_run_views_paginated(
    run_storage: RunLogStorage,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    page_size: int = RUN_METRICS_PAGE_SIZE,
) -> AsyncIterator[list[RunView]]:
    offset = 0
    while True:
        page = await run_storage.list_run_views(
            user_id=user_id,
            session_id=session_id,
            limit=page_size,
            offset=offset,
        )
        if not page:
            return
        yield page
        if len(page) < page_size:
            return
        offset += len(page)


async def summarize_run_views_paginated(
    run_storage: RunLogStorage,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    page_size: int = RUN_METRICS_PAGE_SIZE,
) -> RunMetricsSummary:
    summary = RunMetricsSummary()
    async for page in iter_run_views_paginated(
        run_storage,
        user_id=user_id,
        session_id=session_id,
        page_size=page_size,
    ):
        for run in page:
            add_run_to_summary(summary, run)
    return summary


# ── Paginated trace iteration ───────────────────────────────────────────────


async def iter_traces_paginated(
    trace_storage: BaseTraceStorage,
    *,
    agent_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    status: str | None = None,
    page_size: int = TRACE_METRICS_PAGE_SIZE,
) -> AsyncIterator[list[Trace]]:
    offset = 0
    while True:
        query: dict[str, object] = {
            "limit": page_size,
            "offset": offset,
        }
        if agent_id is not None:
            query["agent_id"] = agent_id
        if session_id is not None:
            query["session_id"] = session_id
        if user_id is not None:
            query["user_id"] = user_id
        if status is not None:
            query["status"] = status

        page = await trace_storage.query_traces(query)
        if not page:
            return
        yield page
        if len(page) < page_size:
            return
        offset += len(page)


def add_trace_to_summary(summary: TraceAggregateSummary, trace: Trace) -> None:
    summary.trace_count += 1
    summary.total_tokens += trace.total_tokens or 0
    summary.total_llm_calls += trace.total_llm_calls or 0
    summary.total_tool_calls += trace.total_tool_calls or 0
    summary.total_token_cost += trace.total_token_cost or 0.0


async def summarize_traces_paginated(
    trace_storage: BaseTraceStorage,
    *,
    agent_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    status: str | None = None,
    page_size: int = TRACE_METRICS_PAGE_SIZE,
) -> TraceAggregateSummary:
    summary = TraceAggregateSummary()
    async for page in iter_traces_paginated(
        trace_storage,
        agent_id=agent_id,
        session_id=session_id,
        user_id=user_id,
        status=status,
        page_size=page_size,
    ):
        for trace in page:
            add_trace_to_summary(summary, trace)
    return summary


async def count_agents_paginated(
    agent_registry: AgentRegistry,
    *,
    page_size: int = AGENT_COUNT_PAGE_SIZE,
) -> int:
    offset = 0
    count = 0
    while True:
        page = await agent_registry.list_agents(limit=page_size, offset=offset)
        if not page:
            return count
        count += len(page)
        if len(page) < page_size:
            return count
        offset += len(page)
