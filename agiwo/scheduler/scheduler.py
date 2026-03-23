"""
Scheduler — public facade for agent scheduling orchestration.

The facade owns lifecycle and construction only. All scheduling behavior is
delegated to SchedulerEngine.
"""

import asyncio
from typing import Any, AsyncIterator

from agiwo.agent.agent import Agent
from agiwo.agent.input import UserInput
from agiwo.agent.runtime import AgentStreamItem, RunOutput
from agiwo.agent.scheduler_port import adapt_scheduler_agent
from agiwo.scheduler.commands import RouteResult
from agiwo.scheduler.engine import SchedulerEngine
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import AgentState, PendingEvent, SchedulerConfig
from agiwo.scheduler.runtime_tools import (
    CancelAgentTool,
    ListAgentsTool,
    QuerySpawnedAgentTool,
    SleepAndWaitTool,
    SpawnAgentTool,
)
from agiwo.scheduler.store import create_agent_state_storage
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class Scheduler:
    """Public facade over the scheduler engine."""

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self._config = config or SchedulerConfig()
        self._store = create_agent_state_storage(self._config.state_storage)
        self._guard = TaskGuard(self._config.task_limits, self._store)
        self._check_interval = self._config.check_interval
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._engine = SchedulerEngine(
            store=self._store,
            config=self._config,
            guard=self._guard,
            semaphore=self._semaphore,
        )
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._scheduling_tools: list[object] = self._create_scheduling_tools()
        self._engine.set_scheduling_tools(self._scheduling_tools)

    def _create_scheduling_tools(self) -> list[object]:
        return [
            SpawnAgentTool(self._engine),
            SleepAndWaitTool(self._engine),
            QuerySpawnedAgentTool(self._engine),
            CancelAgentTool(self._engine),
            ListAgentsTool(self._engine),
        ]

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._loop())
        logger.info("scheduler_started", check_interval=self._check_interval)

    async def stop(self) -> None:
        self._running = False

        if self._engine._rt.active_tasks:
            logger.info(
                "scheduler_waiting_for_active_tasks",
                count=len(self._engine._rt.active_tasks),
            )
            _done, pending = await asyncio.wait(
                self._engine._rt.active_tasks,
                timeout=self._config.graceful_shutdown_wait_seconds,
            )
            for task in pending:
                task.cancel()

        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        await self._store.close()
        logger.info("scheduler_stopped")

    async def __aenter__(self) -> "Scheduler":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    def get_registered_agent(self, state_id: str) -> Agent | None:
        return self._engine.get_registered_agent(state_id)

    def _adapt_agent(self, agent: Agent | None):
        if agent is None:
            return None
        return adapt_scheduler_agent(agent)

    async def run(
        self,
        agent: Agent,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        timeout: float | None = None,
        abort_signal: AbortSignal | None = None,
        persistent: bool = False,
    ) -> RunOutput:
        return await self._engine.run(
            self._adapt_agent(agent),
            user_input,
            session_id=session_id,
            timeout=timeout,
            abort_signal=abort_signal,
            persistent=persistent,
        )

    async def submit(
        self,
        agent: Agent,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
        persistent: bool = False,
        agent_config_id: str | None = None,
    ) -> str:
        return await self._engine.submit(
            self._adapt_agent(agent),
            user_input,
            session_id=session_id,
            abort_signal=abort_signal,
            persistent=persistent,
            agent_config_id=agent_config_id,
        )

    async def enqueue_input(
        self,
        state_id: str,
        user_input: UserInput,
        *,
        agent: Agent | None = None,
    ) -> None:
        await self._engine.enqueue_input(
            state_id,
            user_input,
            agent=self._adapt_agent(agent),
        )

    async def route_root_input(
        self,
        user_input: UserInput,
        *,
        agent: Agent,
        state_id: str | None = None,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
        persistent: bool = True,
        agent_config_id: str | None = None,
        timeout: float | None = None,
        include_child_events: bool = True,
    ) -> RouteResult:
        return await self._engine.route_root_input(
            user_input,
            agent=self._adapt_agent(agent),
            state_id=state_id,
            session_id=session_id,
            abort_signal=abort_signal,
            persistent=persistent,
            agent_config_id=agent_config_id,
            timeout=timeout,
            include_child_events=include_child_events,
        )

    async def stream(
        self,
        user_input: UserInput,
        *,
        agent: Agent | None = None,
        state_id: str | None = None,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
        persistent: bool = False,
        agent_config_id: str | None = None,
        timeout: float | None = None,
        include_child_events: bool = True,
    ) -> AsyncIterator[AgentStreamItem]:
        async for item in self._engine.stream(
            user_input,
            agent=self._adapt_agent(agent),
            state_id=state_id,
            session_id=session_id,
            abort_signal=abort_signal,
            persistent=persistent,
            agent_config_id=agent_config_id,
            timeout=timeout,
            include_child_events=include_child_events,
        ):
            yield item

    async def wait_for(
        self,
        state_id: str,
        timeout: float | None = None,
    ) -> RunOutput:
        return await self._engine.wait_for(state_id, timeout=timeout)

    async def get_state(self, state_id: str) -> AgentState | None:
        return await self._engine.get_state(state_id)

    async def list_states(
        self,
        *,
        statuses=None,
        parent_id: str | None = None,
        session_id: str | None = None,
        signal_propagated: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentState]:
        return await self._engine.list_states(
            statuses=statuses,
            parent_id=parent_id,
            session_id=session_id,
            signal_propagated=signal_propagated,
            limit=limit,
            offset=offset,
        )

    async def list_events(
        self,
        *,
        target_agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[PendingEvent]:
        return await self._engine.list_events(
            target_agent_id=target_agent_id,
            session_id=session_id,
        )

    async def get_stats(self) -> dict[str, int]:
        return await self._engine.get_stats()

    async def cancel(self, state_id: str, reason: str = "Cancelled by user") -> bool:
        return await self._engine.cancel(state_id, reason=reason)

    async def steer(
        self,
        state_id: str,
        user_input: UserInput,
        *,
        urgent: bool = False,
    ) -> bool:
        return await self._engine.steer(state_id, user_input, urgent=urgent)

    async def shutdown(self, state_id: str) -> bool:
        return await self._engine.shutdown(state_id)

    async def rebind_agent(self, state_id: str, agent: Agent) -> bool:
        return await self._engine.rebind_agent(state_id, self._adapt_agent(agent))

    async def _loop(self) -> None:
        logger.info("scheduler_loop_started")
        while self._running:
            try:
                await self._engine.tick()
                await self._engine.wait_for_nudge(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("scheduler_tick_error")


__all__ = ["Scheduler"]
