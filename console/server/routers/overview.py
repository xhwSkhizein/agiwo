"""Dashboard overview API router."""

from fastapi import APIRouter

from server.dependencies import ConsoleRuntimeDep
from server.models.view import DashboardOverviewResponse, SchedulerStatsResponse
from server.services.metrics import (
    collect_session_aggregates,
    count_agents_paginated,
    summarize_traces_paginated,
)

router = APIRouter(prefix="/api", tags=["overview"])

_EMPTY_SCHEDULER_STATS = {
    "total": 0,
    "pending": 0,
    "running": 0,
    "waiting": 0,
    "idle": 0,
    "queued": 0,
    "completed": 0,
    "failed": 0,
}


@router.get("/overview", response_model=DashboardOverviewResponse)
async def get_dashboard_overview(
    runtime: ConsoleRuntimeDep,
) -> DashboardOverviewResponse:
    """Return real aggregate counts for the dashboard top-line stats."""
    sessions = await collect_session_aggregates(runtime.run_step_storage)
    traces = await summarize_traces_paginated(runtime.trace_storage)
    total_agents = await count_agents_paginated(runtime.agent_registry)
    scheduler_stats = (
        await runtime.scheduler.get_stats()
        if runtime.scheduler is not None
        else _EMPTY_SCHEDULER_STATS
    )
    return DashboardOverviewResponse(
        total_sessions=len(sessions),
        total_traces=traces.trace_count,
        total_agents=total_agents,
        total_tokens=traces.total_tokens,
        scheduler=SchedulerStatsResponse(**scheduler_stats),
    )
