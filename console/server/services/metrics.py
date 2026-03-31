"""Run metrics aggregation — for sessions, scheduler states, and standalone runs."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field

from agiwo.agent import Run, RunStatus, UserInput
from agiwo.agent.models.input import UserMessage
from agiwo.agent.storage.base import RunStepStorage
from agiwo.scheduler.models import AgentState
from agiwo.utils.serialization import serialize_optional_datetime

from server.schemas import RunMetricsSummary


@dataclass
class SessionAggregate:
    session_id: str
    agent_id: str | None
    last_run: Run | None
    metrics: RunMetricsSummary
    created_at: datetime | None
    updated_at: datetime | None


class SessionSummaryData(BaseModel):
    session_id: str
    agent_id: str | None = None
    last_user_input: UserInput | None = None
    last_response: str | None = None
    run_count: int = 0
    step_count: int = 0
    metrics: RunMetricsSummary = Field(default_factory=RunMetricsSummary)
    created_at: str | None = None
    updated_at: str | None = None


def session_aggregate_to_summary_data(session: SessionAggregate) -> SessionSummaryData:
    last_run = session.last_run
    last_user_input = (
        UserMessage.to_transport_payload(last_run.user_input)
        if last_run is not None
        else None
    )
    return SessionSummaryData(
        session_id=session.session_id,
        agent_id=session.agent_id,
        last_user_input=last_user_input,
        last_response=(
            last_run.response_content[:200]
            if last_run and last_run.response_content
            else None
        ),
        run_count=session.metrics.run_count,
        step_count=session.metrics.step_count,
        metrics=session.metrics,
        created_at=serialize_optional_datetime(session.created_at),
        updated_at=serialize_optional_datetime(session.updated_at),
    )


def session_aggregate_to_chat_summary(session: SessionAggregate) -> dict[str, object]:
    summary = session_aggregate_to_summary_data(session)
    last_input = None
    if session.last_run is not None:
        raw = UserMessage.to_storage_value(session.last_run.user_input)
        if isinstance(raw, str) and raw:
            last_input = raw[:200]
    return {
        "session_id": summary.session_id,
        "run_count": summary.run_count,
        "last_input": last_input,
        "last_response": summary.last_response,
        "updated_at": summary.updated_at,
    }


RUN_METRICS_PAGE_SIZE = 500


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
    run_storage: RunStepStorage,
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
    run_storage: RunStepStorage,
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


# TODO(perf): This function scans all runs to derive sessions since there is
# no dedicated session table. For large datasets, consider adding a persistent
# session index or pushing pagination into the storage layer.
async def collect_session_aggregates(
    run_storage: RunStepStorage,
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
    run_storage: RunStepStorage,
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
