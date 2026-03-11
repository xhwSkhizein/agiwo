"""Shared session aggregation helpers for console APIs."""

from server.domain.run_metrics import RunMetricsSummary
from server.domain.sessions import SessionAggregate
from server.services.run_metrics import RunStoragePort, add_run_to_summary, iter_runs_paginated


def _run_sort_key(run: object) -> tuple[object | None, object | None, str]:
    return (
        getattr(run, "updated_at", None),
        getattr(run, "created_at", None),
        getattr(run, "id", ""),
    )


def _merge_run_into_session(
    session_map: dict[str, SessionAggregate],
    run: object,
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
