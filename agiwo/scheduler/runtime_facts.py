"""Scheduler-facing helpers for replaying runtime facts from RunLog."""

from agiwo.agent import RunView, RuntimeDecisionState, StepView
from agiwo.agent.models.log import RunLogEntry, RunLogEntryKind
from agiwo.scheduler.models import AgentState
from agiwo.scheduler.runtime_state import RuntimeState


class SchedulerRuntimeFacts:
    """Read scheduler-visible runtime facts from the live runtime agent."""

    def __init__(self, rt: RuntimeState) -> None:
        self._rt = rt

    def has_runtime_agent(self, state: AgentState) -> bool:
        return state.id in self._rt.agents

    async def get_latest_run_view(self, state: AgentState) -> RunView | None:
        agent = self._rt.agents.get(state.id)
        if agent is None:
            return None
        return await agent.run_log_storage.get_latest_run_view(
            state.resolve_runtime_session_id()
        )

    async def get_run_view(self, run_id: str) -> RunView | None:
        """Locate a RunView by scanning runtime agents' RunLog storage."""
        for agent in self._rt.agents.values():
            view = await agent.run_log_storage.get_run_view(run_id)
            if view is not None:
                return view
        return None

    async def list_run_log_entries(
        self,
        run_id: str,
        *,
        kinds: list[RunLogEntryKind] | None = None,
        limit: int = 10_000,
    ) -> list[RunLogEntry]:
        """List RunLog entries for a run from the runtime agent that owns it."""
        for agent in self._rt.agents.values():
            view = await agent.run_log_storage.get_run_view(run_id)
            if view is None:
                continue
            return await agent.run_log_storage.list_entries(
                session_id=view.session_id,
                run_id=run_id,
                kinds=kinds,
                limit=limit,
            )
        return []

    async def list_step_views(
        self,
        state: AgentState,
        *,
        include_rolled_back: bool = False,
        run_id: str | None = None,
        limit: int = 1000,
    ) -> list[StepView]:
        agent = self._rt.agents.get(state.id)
        if agent is None:
            return []
        return await agent.run_log_storage.list_step_views(
            session_id=state.resolve_runtime_session_id(),
            agent_id=state.id,
            run_id=run_id,
            include_rolled_back=include_rolled_back,
            limit=limit,
        )

    async def get_runtime_decision_state(
        self,
        state: AgentState,
        *,
        run_id: str | None = None,
    ) -> RuntimeDecisionState:
        agent = self._rt.agents.get(state.id)
        if agent is None:
            return RuntimeDecisionState()
        return await agent.run_log_storage.get_runtime_decision_state(
            session_id=state.resolve_runtime_session_id(),
            agent_id=state.id,
            run_id=run_id,
        )

    async def get_result_summary(self, state: AgentState) -> str | None:
        latest_run = await self.get_latest_run_view(state)
        if latest_run is not None and latest_run.response:
            return latest_run.response
        return state.result_summary


__all__ = ["SchedulerRuntimeFacts"]
