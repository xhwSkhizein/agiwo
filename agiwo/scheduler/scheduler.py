"""
Scheduler — The orchestration layer for agent scheduling.

Manages agent lifecycles, sleep/wake cycles, and child agent execution.
Dependency direction: scheduler → agent (one-way). Agent has no knowledge of Scheduler.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agiwo.agent.agent import Agent
from agiwo.agent.schema import RunOutput, TerminationReason
from agiwo.scheduler.executor import SchedulerExecutor
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    SchedulerConfig,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store import AgentStateStorage, create_agent_state_storage
from agiwo.scheduler.tools import (
    QuerySpawnedAgentTool,
    SleepAndWaitTool,
    SpawnAgentTool,
)
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class Scheduler:
    """
    Agent scheduling orchestrator.

    Manages a pool of agent executions, handling spawn, sleep, wake, and
    completion lifecycles. Provides both blocking (run) and non-blocking
    (submit/wait_for) APIs.

    Usage (blocking):
        config = SchedulerConfig(
            state_storage=AgentStateStorageConfig(storage_type="sqlite", config={"db_path": "scheduler.db"}),
        )
        async with Scheduler(config) as scheduler:
            result = await scheduler.run(agent, "Complex task")

    Usage (non-blocking):
        async with Scheduler() as scheduler:
            state_id = await scheduler.submit(agent, "Task A")
            result = await scheduler.wait_for(state_id)
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self._config = config or SchedulerConfig()
        self._store = create_agent_state_storage(self._config.state_storage)
        self._guard = TaskGuard(self._config.task_limits, self._store)
        self._check_interval = self._config.check_interval
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._active_tasks: set[asyncio.Task] = set()
        self._agents: dict[str, Agent] = {}
        self._abort_signals: dict[str, AbortSignal] = {}
        self._state_events: dict[str, asyncio.Event] = {}
        self._dispatched_state_ids: set[str] = set()
        self._scheduling_tools: list = self._create_scheduling_tools()
        self._executor = SchedulerExecutor(
            scheduler=self,
            store=self._store,
            agents=self._agents,
            abort_signals=self._abort_signals,
            state_events=self._state_events,
            semaphore=self._semaphore,
            dispatched_state_ids=self._dispatched_state_ids,
        )

    def _create_scheduling_tools(self) -> list:
        """Create the scheduling tools that will be injected into agents."""
        return [
            SpawnAgentTool(self._store, self._guard),
            SleepAndWaitTool(self._store, self._guard),
            QuerySpawnedAgentTool(self._store),
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

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

        if self._active_tasks:
            logger.info(
                "scheduler_waiting_for_active_tasks",
                count=len(self._active_tasks),
            )
            done, pending = await asyncio.wait(
                self._active_tasks, timeout=self._config.graceful_shutdown_wait_seconds
            )
            for t in pending:
                t.cancel()

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

    # ──────────────────────────────────────────────────────────────────────
    # Public API — Blocking
    # ──────────────────────────────────────────────────────────────────────

    @property
    def store(self) -> AgentStateStorage:
        """Expose the underlying state storage (read-only access)."""
        return self._store

    async def run(
        self,
        agent: Agent,
        user_input: str,
        *,
        session_id: str | None = None,
        timeout: float | None = None,
        abort_signal: AbortSignal | None = None,
        persistent: bool = False,
    ) -> RunOutput:
        """
        Submit an agent task and block until the entire orchestration completes.

        This is equivalent to submit() + wait_for().
        """
        state_id = await self.submit(
            agent,
            user_input,
            session_id=session_id,
            abort_signal=abort_signal,
            persistent=persistent,
        )
        return await self.wait_for(state_id, timeout=timeout)

    # ──────────────────────────────────────────────────────────────────────
    # Public API — Non-blocking
    # ──────────────────────────────────────────────────────────────────────

    async def submit(
        self,
        agent: Agent,
        user_input: str,
        *,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
        persistent: bool = False,
    ) -> str:
        """
        Submit an agent task, return immediately with the state_id.

        Args:
            agent: The agent to run.
            user_input: User message to the agent.
            session_id: Optional session ID for conversation continuity.
            abort_signal: Optional signal to cancel the orchestration.
            persistent: If True, the root agent stays SLEEPING after completion
                        and accepts new tasks via submit_task().

        Returns:
            state_id (= agent.id) that can be used with wait_for() or get_state().
        """
        existing = await self._store.get_state(agent.id)
        if existing is not None and existing.status in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.SLEEPING,
            AgentStateStatus.PENDING,
        ):
            raise RuntimeError(
                f"Agent '{agent.id}' is already active (status={existing.status.value}). "
                f"Cannot submit concurrently. Use a different agent_id or submit_task()."
            )

        self._prepare_agent(agent)

        resolved_session_id = session_id or str(uuid4())
        state = AgentState(
            id=agent.id,
            session_id=resolved_session_id,
            status=AgentStateStatus.RUNNING,
            task=user_input,
            parent_id=None,
            is_persistent=persistent,
            depth=0,
        )
        await self._store.save_state(state)

        if abort_signal is not None:
            self._abort_signals[state.id] = abort_signal

        task = asyncio.create_task(
            self._run_root_agent(agent, user_input, resolved_session_id, state)
        )
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        return state.id

    async def submit_task(self, state_id: str, task: str) -> None:
        """
        Submit a new task to a persistent root agent that is SLEEPING.

        The agent will be woken up on the next tick with TASK_SUBMITTED condition.

        Args:
            state_id: The state_id of the persistent root agent.
            task: The new task content.

        Raises:
            RuntimeError: If the agent is not persistent or not SLEEPING.
        """
        state = await self._store.get_state(state_id)
        if state is None:
            raise RuntimeError(f"Agent state '{state_id}' not found")
        if not state.is_persistent:
            raise RuntimeError(f"Agent '{state_id}' is not persistent. Use submit() instead.")
        if state.status != AgentStateStatus.SLEEPING:
            raise RuntimeError(
                f"Agent '{state_id}' is not SLEEPING (status={state.status.value}). "
                f"Cannot submit task."
            )

        wc = WakeCondition(type=WakeType.TASK_SUBMITTED, submitted_task=task)
        await self._store.update_status(
            state_id, AgentStateStatus.SLEEPING, wake_condition=wc
        )
        logger.info("task_submitted_to_persistent_agent", state_id=state_id)

    async def wait_for(
        self,
        state_id: str,
        timeout: float | None = None,
    ) -> RunOutput:
        """
        Block until a submitted task reaches COMPLETED or FAILED.

        For persistent agents, this returns when the current task completes
        (agent goes back to SLEEPING), not when the agent itself terminates.

        Uses event-based notification for low-latency response instead of polling.
        """
        start = time.time()
        initial_state = await self._store.get_state(state_id)
        is_persistent = initial_state.is_persistent if initial_state else False

        event = self._state_events.setdefault(state_id, asyncio.Event())

        try:
            while True:
                state = await self._store.get_state(state_id)
                if state is not None:
                    if state.status == AgentStateStatus.COMPLETED:
                        return RunOutput(
                            response=state.result_summary,
                            termination_reason=TerminationReason.COMPLETED,
                        )
                    if state.status == AgentStateStatus.FAILED:
                        return RunOutput(
                            error=state.result_summary,
                            termination_reason=TerminationReason.ERROR,
                        )
                    if is_persistent and state.status == AgentStateStatus.SLEEPING:
                        return RunOutput(
                            response=state.result_summary,
                            termination_reason=TerminationReason.COMPLETED,
                        )

                remaining = None
                if timeout is not None:
                    elapsed = time.time() - start
                    if elapsed >= timeout:
                        return RunOutput(termination_reason=TerminationReason.TIMEOUT)
                    remaining = timeout - elapsed

                event.clear()
                try:
                    await asyncio.wait_for(event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return RunOutput(termination_reason=TerminationReason.TIMEOUT)
        finally:
            self._state_events.pop(state_id, None)

    async def get_state(self, state_id: str) -> AgentState | None:
        """Query the current state of a scheduled agent."""
        return await self._store.get_state(state_id)

    async def cancel(self, state_id: str, reason: str = "Cancelled by user") -> bool:
        """
        Hard cancel a running or sleeping orchestration (user-triggered only).

        Triggers the AbortSignal for the target agent and recursively cancels
        all descendant agents.
        """
        state = await self._store.get_state(state_id)
        if state is None:
            return False
        if state.status not in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.SLEEPING,
            AgentStateStatus.PENDING,
        ):
            return False

        await self._recursive_cancel(state_id, reason)
        logger.info("scheduler_cancel", state_id=state_id, reason=reason)
        return True

    async def shutdown(self, state_id: str) -> bool:
        """
        Graceful shutdown — let the agent produce a final report then exit (user-triggered only).

        For SLEEPING agents: wakes with a shutdown message so the agent can summarize.
        For RUNNING agents: sets timeout_at to force TIMEOUT path -> triggers summarize.
        Recursively shuts down all descendant agents.
        """
        state = await self._store.get_state(state_id)
        if state is None:
            return False
        if state.status not in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.SLEEPING,
            AgentStateStatus.PENDING,
        ):
            return False

        await self._recursive_shutdown(state_id)
        logger.info("scheduler_shutdown", state_id=state_id)
        return True

    # ──────────────────────────────────────────────────────────────────────
    # Agent Preparation
    # ──────────────────────────────────────────────────────────────────────

    def _prepare_agent(self, agent: Agent) -> None:
        """Inject scheduling tools into agent (idempotent). Register agent."""
        existing_names = {t.get_name() for t in agent.tools}
        for tool in self._scheduling_tools:
            if tool.get_name() not in existing_names:
                agent.tools.append(tool)
        if agent.options is not None:
            agent.options.enable_termination_summary = True
        self._agents[agent.id] = agent

    def _create_child_agent(self, state: AgentState) -> Agent:
        """Create a child Agent by copying the parent's configuration."""
        return self._executor.create_child_agent(state)

    # ──────────────────────────────────────────────────────────────────────
    # Scheduling Loop
    # ──────────────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main scheduling loop running in the background."""
        logger.info("scheduler_loop_started")
        while self._running:
            try:
                await self._tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("scheduler_tick_error")
            await asyncio.sleep(self._check_interval)

    async def _tick(self) -> None:
        """Single scheduling tick: propagate signals, enforce timeouts, start pending, wake sleeping."""
        await self._propagate_signals()
        await self._enforce_timeouts()
        await self._start_pending()
        await self._wake_sleeping()

    async def _propagate_signals(self) -> None:
        """Propagate completion/failure signals from child to parent agents."""
        completed = await self._store.find_unpropagated_completed()
        for state in completed:
            if state.parent_id is not None:
                await self._store.mark_child_completed(state.parent_id, state.id)
                logger.info(
                    "signal_propagated",
                    child_id=state.id,
                    parent_id=state.parent_id,
                    child_status=state.status.value,
                )
            await self._store.mark_propagated(state.id)

    async def _enforce_timeouts(self) -> None:
        """Find timed-out SLEEPING agents and wake them for summarization."""
        now = datetime.now(timezone.utc)
        timed_out = await self._guard.find_timed_out(now)
        for state in timed_out:
            task = asyncio.create_task(self._executor.wake_for_timeout(state))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

    async def _start_pending(self) -> None:
        """Start all PENDING agents, skipping already-dispatched ones."""
        pending = await self._store.find_pending()
        for state in pending:
            if state.id in self._dispatched_state_ids:
                continue
            self._dispatched_state_ids.add(state.id)
            task = asyncio.create_task(self._executor.run_agent(state))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

    async def _wake_sleeping(self) -> None:
        """Wake agents whose conditions are satisfied, skipping already-dispatched ones."""
        now = datetime.now(timezone.utc)
        wakeable = await self._store.find_wakeable(now)
        for state in wakeable:
            if state.id in self._dispatched_state_ids:
                continue
            rejection = await self._guard.check_wake(state)
            if rejection is not None:
                logger.warning(
                    "wake_rejected",
                    state_id=state.id,
                    reason=rejection,
                )
                await self._store.update_status(
                    state.id,
                    AgentStateStatus.FAILED,
                    result_summary=f"Wake rejected: {rejection}",
                )
                continue
            self._dispatched_state_ids.add(state.id)
            task = asyncio.create_task(self._executor.wake_agent(state))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

    # ──────────────────────────────────────────────────────────────────────
    # Agent Execution (delegated to SchedulerExecutor)
    # ──────────────────────────────────────────────────────────────────────

    async def _run_root_agent(
        self,
        agent: Agent,
        user_input: str,
        session_id: str,
        state: AgentState,
    ) -> None:
        """Run the root agent (submitted by user)."""
        await self._executor.run_root_agent(agent, user_input, session_id, state)

    async def _build_wake_message(self, state: AgentState) -> str:
        """Build a wake message with auto-injected child results."""
        return await self._executor._build_wake_message(state)

    # ──────────────────────────────────────────────────────────────────────
    # Cancel / Shutdown
    # ──────────────────────────────────────────────────────────────────────

    async def _recursive_cancel(self, state_id: str, reason: str) -> None:
        """Recursively cancel an agent and all its descendants."""
        signal = self._abort_signals.get(state_id)
        if signal is not None:
            signal.abort(reason)

        children = await self._store.get_states_by_parent(state_id)
        for child in children:
            if child.status in (
                AgentStateStatus.RUNNING,
                AgentStateStatus.SLEEPING,
                AgentStateStatus.PENDING,
            ):
                await self._recursive_cancel(child.id, reason)

        await self._executor._update_status_and_notify(
            state_id, AgentStateStatus.FAILED, result_summary=reason
        )

    async def _recursive_shutdown(self, state_id: str) -> None:
        """Recursively graceful-shutdown an agent and all its descendants."""
        children = await self._store.get_states_by_parent(state_id)
        for child in children:
            if child.status in (
                AgentStateStatus.RUNNING,
                AgentStateStatus.SLEEPING,
                AgentStateStatus.PENDING,
            ):
                await self._recursive_shutdown(child.id)

        state = await self._store.get_state(state_id)
        if state is None:
            return

        if state.status == AgentStateStatus.SLEEPING:
            wc = WakeCondition(
                type=WakeType.TASK_SUBMITTED,
                submitted_task="System shutdown requested. Please produce a final summary report of all work done so far.",
            )
            await self._store.update_status(
                state_id, AgentStateStatus.SLEEPING, wake_condition=wc
            )
        elif state.status == AgentStateStatus.PENDING:
            await self._executor._update_status_and_notify(
                state_id, AgentStateStatus.FAILED, result_summary="Shutdown before execution"
            )
