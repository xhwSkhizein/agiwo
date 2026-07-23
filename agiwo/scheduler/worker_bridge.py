"""Scheduler-backed implementation of ``WorkerSchedulerPort``."""

from typing import TYPE_CHECKING

from agiwo.agent.worker_port import (
    WorkerSchedulerPort,
    WorkerSpawnRequest,
    WorkerStateStatus,
    WorkerStateView,
)
from agiwo.scheduler.commands import SpawnChildRequest
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import AgentState, AgentStateStatus

if TYPE_CHECKING:
    from agiwo.agent.agent import Agent


def _to_worker_status(status: AgentStateStatus) -> WorkerStateStatus:
    return WorkerStateStatus(status.value)


def _to_worker_view(state: AgentState) -> WorkerStateView:
    return WorkerStateView(
        worker_id=state.id,
        status=_to_worker_status(state.status),
        result_summary=state.result_summary,
    )


class SchedulerWorkerPort:
    """Adapts ``Scheduler`` to the agent-side Worker port via public facade APIs."""

    def __init__(self, scheduler: Scheduler) -> None:
        self._scheduler = scheduler

    async def register_parent(
        self,
        *,
        state_id: str,
        session_id: str,
        agent: "Agent",
    ) -> None:
        await self._scheduler.register_worker_parent(
            state_id=state_id,
            session_id=session_id,
            agent=agent,
        )

    async def start(self) -> None:
        await self._scheduler.start()

    async def spawn_worker(self, request: WorkerSpawnRequest) -> WorkerStateView:
        state = await self._scheduler.spawn_worker(
            SpawnChildRequest(
                parent_agent_id=request.parent_agent_id,
                session_id=request.session_id,
                task=request.task,
                instruction=request.instruction,
            )
        )
        return _to_worker_view(state)

    def nudge(self) -> None:
        self._scheduler.nudge()

    async def list_children(
        self,
        *,
        parent_id: str,
        session_id: str,
        limit: int,
    ) -> list[WorkerStateView]:
        children = await self._scheduler.list_states(
            parent_id=parent_id,
            session_id=session_id,
            limit=limit,
        )
        return [_to_worker_view(child) for child in children]

    async def cancel_worker(self, worker_id: str, reason: str) -> None:
        await self._scheduler.cancel(worker_id, reason)

    async def wait_for_worker(self, worker_id: str) -> None:
        await self._scheduler.wait_for(worker_id)

    async def get_worker_state(self, worker_id: str) -> WorkerStateView | None:
        state = await self._scheduler.get_state(worker_id)
        if state is None:
            return None
        return _to_worker_view(state)

    async def get_worker_report(self, worker_id: str) -> str:
        await self._scheduler.wait_for(worker_id)
        state = await self._scheduler.get_state(worker_id)
        if state is None:
            return "Worker state missing after completion."
        summary = await self._scheduler.get_result_summary(state)
        if summary:
            return summary
        if state.status is AgentStateStatus.FAILED:
            return state.result_summary or "Worker failed."
        return "Worker completed."

    async def sync_parent_idle(self, parent_id: str) -> None:
        await self._scheduler.mark_parent_idle(parent_id)


def scheduler_worker_port(scheduler: Scheduler) -> WorkerSchedulerPort:
    """Build a Worker port backed by a live Scheduler instance."""
    return SchedulerWorkerPort(scheduler)


__all__ = ["SchedulerWorkerPort", "scheduler_worker_port"]
