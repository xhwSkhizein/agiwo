"""Run metrics aggregation — for sessions, scheduler states, and standalone runs."""

from collections.abc import AsyncIterator
from typing import Protocol

from agiwo.agent import Run, RunStatus
from agiwo.scheduler.models import AgentState

from server.domain.run_metrics import RunMetricsSummary
from server.domain.sessions import SessionAggregate


RUN_METRICS_PAGE_SIZE = 500


class RunStoragePort(Protocol):
    async def list_runs(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Run]:
        ...


# ── Core accumulator ────────────────────────────────────────────────────────


def add_run_to_summary(summary: RunMetricsSummary, run: Run) -> None:
    summary.run_count += 1
    if run.status == RunStatus.COMPLETED:
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


async def iter_runs_paginated(
    run_storage: RunStoragePort,
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    page_size: int = RUN_METRICS_PAGE_SIZE,
) -> AsyncIterator[list[Run]]:
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


# ── Session aggregation ─────────────────────────────────────────────────────


def _run_sort_key(run: Run) -> tuple[object | None, object | None, str]:
    return (run.updated_at, run.created_at, run.id)


def _merge_run_into_session(
    session_map: dict[str, SessionAggregate],
    run: Run,
) -> None:
    session = session_map.get(run.session_id)
    if session is None:
        session = SessionAggregate(
            session_id=run.session_id,
            agent_id=run.agent_id,
            last_run=run,
            metrics=RunMetricsSummary(),
            created_at=run.created_at,
            updated_at=run.updated_at,
        )
        session_map[run.session_id] = session

    if run.created_at and (
        session.created_at is None or run.created_at < session.created_at
    ):
        session.created_at = run.created_at
    if run.updated_at and (
        session.updated_at is None or run.updated_at > session.updated_at
    ):
        session.updated_at = run.updated_at
    if session.last_run is None or _run_sort_key(run) > _run_sort_key(session.last_run):
        session.last_run = run

    add_run_to_summary(session.metrics, run)


async def collect_session_aggregates(
    run_storage: RunStoragePort,
    *,
    agent_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> list[SessionAggregate]:
    session_map: dict[str, SessionAggregate] = {}
    async for runs in iter_runs_paginated(
        run_storage,
        user_id=user_id,
        session_id=session_id,
    ):
        for run in runs:
            if agent_id is not None and run.agent_id != agent_id:
                continue
            _merge_run_into_session(session_map, run)

    return sorted(
        session_map.values(),
        key=lambda session: session.updated_at or "",
        reverse=True,
    )


# ── Scheduler state metrics ─────────────────────────────────────────────────


async def build_metrics_by_state(
    states: list[AgentState],
    run_storage: RunStoragePort,
) -> dict[tuple[str, str], RunMetricsSummary]:
    if not states:
        return {}
    session_to_agent_ids: dict[str, set[str]] = {}
    for state in states:
        runtime_sid = state.resolve_runtime_session_id()
        session_to_agent_ids.setdefault(runtime_sid, set()).add(state.id)

    metrics_by_state: dict[tuple[str, str], RunMetricsSummary] = {}
    for session_id, agent_ids in session_to_agent_ids.items():
        async for runs in iter_runs_paginated(run_storage, session_id=session_id):
            for run in runs:
                if run.agent_id not in agent_ids:
                    continue
                key = (session_id, run.agent_id)
                summary = metrics_by_state.setdefault(key, RunMetricsSummary())
                add_run_to_summary(summary, run)

    for state in states:
        key = (state.resolve_runtime_session_id(), state.id)
        if key not in metrics_by_state:
            metrics_by_state[key] = RunMetricsSummary()

    return metrics_by_state
