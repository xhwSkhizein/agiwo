"""Run-metric aggregation helpers for scheduler state APIs."""

from agiwo.scheduler.models import AgentState

from server.dependencies import ConsoleRuntime
from server.domain.run_metrics import RunMetricsSummary
from server.services.run_metrics import add_run_to_summary, iter_runs_paginated


async def build_metrics_by_state(
    states: list[AgentState],
    runtime: ConsoleRuntime,
) -> dict[tuple[str, str], RunMetricsSummary]:
    if not states:
        return {}

    run_storage = runtime.storage_manager.run_step_storage
    session_to_agent_ids: dict[str, set[str]] = {}
    for state in states:
        session_to_agent_ids.setdefault(state.session_id, set()).add(state.id)

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
        key = (state.session_id, state.id)
        if key not in metrics_by_state:
            metrics_by_state[key] = RunMetricsSummary()

    return metrics_by_state


__all__ = ["build_metrics_by_state"]
