"""Compatibility facade over RuntimeState run-log queries.

Prefer calling the query methods on ``RuntimeState`` directly. This wrapper
remains for public export and existing tests.
"""

from agiwo.agent import RunView, RuntimeDecisionState, StepView
from agiwo.agent.models.log import RunLogEntry, RunLogEntryKind
from agiwo.scheduler.models import AgentState
from agiwo.scheduler.runtime_state import RuntimeState


class SchedulerRuntimeFacts:
    """Read scheduler-visible runtime facts from the live runtime agent."""

    def __init__(self, rt: RuntimeState) -> None:
        self._rt = rt

    def has_runtime_agent(self, state: AgentState) -> bool:
        return self._rt.has_runtime_agent(state)

    async def get_latest_run_view(self, state: AgentState) -> RunView | None:
        return await self._rt.get_latest_run_view(state)

    async def get_run_view(self, run_id: str) -> RunView | None:
        return await self._rt.get_run_view(run_id)

    async def list_run_log_entries(
        self,
        run_id: str,
        *,
        kinds: list[RunLogEntryKind] | None = None,
        limit: int = 10_000,
    ) -> list[RunLogEntry]:
        return await self._rt.list_run_log_entries(run_id, kinds=kinds, limit=limit)

    async def list_step_views(
        self,
        state: AgentState,
        *,
        include_rolled_back: bool = False,
        run_id: str | None = None,
        limit: int = 1000,
    ) -> list[StepView]:
        return await self._rt.list_step_views(
            state,
            include_rolled_back=include_rolled_back,
            run_id=run_id,
            limit=limit,
        )

    async def get_runtime_decision_state(
        self,
        state: AgentState,
        *,
        run_id: str | None = None,
    ) -> RuntimeDecisionState:
        return await self._rt.get_runtime_decision_state(state, run_id=run_id)

    async def get_result_summary(self, state: AgentState) -> str | None:
        return await self._rt.get_result_summary(state)


__all__ = ["SchedulerRuntimeFacts"]
