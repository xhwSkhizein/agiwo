"""
Scheduler — public facade for agent scheduling orchestration.

The facade owns lifecycle and construction only. All scheduling behavior is
delegated to SchedulerEngine.
"""

import asyncio
from typing import Any, AsyncIterator

from agiwo.agent.agent import Agent
from agiwo.agent.input import UserInput
from agiwo.agent.scheduler_port import adapt_scheduler_agent
from agiwo.agent.runtime import RunOutput
from agiwo.scheduler.coordinator import SchedulerCoordinator
from agiwo.scheduler.engine import SchedulerEngine
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import AgentState, SchedulerConfig, SchedulerOutput
from agiwo.scheduler.runtime_tools import (
    CancelAgentTool,
    ListAgentsTool,
    QuerySpawnedAgentTool,
    SleepAndWaitTool,
    SpawnAgentTool,
)
from agiwo.scheduler.runner import SchedulerRunner
from agiwo.scheduler.store import AgentStateStorage, create_agent_state_storage
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
        self._coordinator = SchedulerCoordinator()
        self._runner = SchedulerRunner(
            store=self._store,
            coordinator=self._coordinator,
            semaphore=self._semaphore,
        )
        self._engine = SchedulerEngine(
            config=self._config,
            store=self._store,
            guard=self._guard,
            coordinator=self._coordinator,
            runner=self._runner,
        )
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._scheduling_tools: list = self._create_scheduling_tools()
        self._engine.set_scheduling_tools(self._scheduling_tools)

    def _create_scheduling_tools(self) -> list:
        """Create the scheduling tools injected into agents."""
        return [
            SpawnAgentTool(self._engine),
            SleepAndWaitTool(self._engine),
            QuerySpawnedAgentTool(self._engine),
            CancelAgentTool(self._engine),
            ListAgentsTool(self._engine),
        ]

    async def start(self) -> None:
        """Start the background scheduling loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._loop())
        logger.info("scheduler_started", check_interval=self._check_interval)

    async def stop(self) -> None:
        """Gracefully stop the scheduler."""
        self._running = False

        if self._coordinator.active_tasks:
            logger.info(
                "scheduler_waiting_for_active_tasks",
                count=len(self._coordinator.active_tasks),
            )
            _done, pending = await asyncio.wait(
                self._coordinator.active_tasks,
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

    @property
    def store(self) -> AgentStateStorage:
        """Expose the underlying state storage for read-only external access."""
        return self._store

    def get_registered_agent(self, state_id: str) -> Agent | None:
        """Return the in-memory agent object for a scheduler state."""
        return self._engine.get_registered_agent(state_id)

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
        agent_port = adapt_scheduler_agent(agent)
        return await self._engine.run(
            agent_port,
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
        agent_port = adapt_scheduler_agent(agent)
        return await self._engine.submit(
            agent_port,
            user_input,
            session_id=session_id,
            abort_signal=abort_signal,
            persistent=persistent,
            agent_config_id=agent_config_id,
        )

    async def submit_task(
        self,
        state_id: str,
        task: UserInput,
        *,
        agent: Agent | None = None,
    ) -> None:
        agent_port = adapt_scheduler_agent(agent) if agent is not None else None
        await self._engine.submit_task(state_id, task, agent=agent_port)

    async def submit_and_subscribe(
        self,
        agent: Agent,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
        persistent: bool = False,
        agent_config_id: str | None = None,
        timeout: float | None = None,
        include_child_outputs: bool = True,
    ) -> AsyncIterator[SchedulerOutput]:
        agent_port = adapt_scheduler_agent(agent)
        async for output in self._engine.submit_and_subscribe(
            agent_port,
            user_input,
            session_id=session_id,
            abort_signal=abort_signal,
            persistent=persistent,
            agent_config_id=agent_config_id,
            timeout=timeout,
            include_child_outputs=include_child_outputs,
        ):
            yield output

    async def submit_task_and_subscribe(
        self,
        state_id: str,
        task: UserInput,
        *,
        agent: Agent | None = None,
        timeout: float | None = None,
        include_child_outputs: bool = True,
    ) -> AsyncIterator[SchedulerOutput]:
        agent_port = adapt_scheduler_agent(agent) if agent is not None else None
        async for output in self._engine.submit_task_and_subscribe(
            state_id,
            task,
            agent=agent_port,
            timeout=timeout,
            include_child_outputs=include_child_outputs,
        ):
            yield output

    async def wait_for(
        self,
        state_id: str,
        timeout: float | None = None,
    ) -> RunOutput:
        return await self._engine.wait_for(state_id, timeout=timeout)

    async def get_state(self, state_id: str) -> AgentState | None:
        return await self._engine.get_state(state_id)

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

    async def _loop(self) -> None:
        """Background scheduler loop."""
        logger.info("scheduler_loop_started")
        while self._running:
            try:
                await self._engine.tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("scheduler_tick_error")
            await asyncio.sleep(self._check_interval)


__all__ = ["Scheduler"]
