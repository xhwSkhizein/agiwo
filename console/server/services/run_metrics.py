"""Helpers for aggregating run-level metrics for API responses."""

from collections.abc import AsyncIterator
from typing import Protocol

from agiwo.agent.schema import RunStatus

from server.domain.run_metrics import RunMetricsSummary


RUN_METRICS_PAGE_SIZE = 500

_INT_FIELDS: list[tuple[str, str]] = [
    ("step_count", "steps_count"),
    ("tool_calls_count", "tool_calls_count"),
    ("input_tokens", "input_tokens"),
    ("output_tokens", "output_tokens"),
    ("total_tokens", "total_tokens"),
    ("cache_read_tokens", "cache_read_tokens"),
    ("cache_creation_tokens", "cache_creation_tokens"),
]

_FLOAT_FIELDS: list[tuple[str, str]] = [
    ("duration_ms", "duration_ms"),
    ("token_cost", "token_cost"),
]


class RunStoragePort(Protocol):
    async def list_runs(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[object]:
        ...


def add_run_to_summary(summary: RunMetricsSummary, run: object) -> None:
    summary.run_count += 1
    if getattr(run, "status", None) == RunStatus.COMPLETED:
        summary.completed_run_count += 1
    metrics = getattr(run, "metrics", None)
    if metrics is None:
        return
    for summary_field, metrics_field in _INT_FIELDS:
        value = getattr(metrics, metrics_field, 0) or 0
        setattr(summary, summary_field, getattr(summary, summary_field) + int(value))
    for summary_field, metrics_field in _FLOAT_FIELDS:
        value = getattr(metrics, metrics_field, 0.0) or 0.0
        setattr(summary, summary_field, getattr(summary, summary_field) + float(value))


async def iter_runs_paginated(
    run_storage: RunStoragePort,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    page_size: int = RUN_METRICS_PAGE_SIZE,
) -> AsyncIterator[list[object]]:
    offset = 0
    while True:
        page = await run_storage.list_runs(
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


async def summarize_runs_paginated(
    run_storage: RunStoragePort,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    page_size: int = RUN_METRICS_PAGE_SIZE,
) -> RunMetricsSummary:
    summary = RunMetricsSummary()
    async for page in iter_runs_paginated(
        run_storage,
        user_id=user_id,
        session_id=session_id,
        page_size=page_size,
    ):
        for run in page:
            add_run_to_summary(summary, run)
    return summary
