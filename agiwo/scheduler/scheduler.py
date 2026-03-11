"""
Scheduler — The orchestration layer for agent scheduling.

Manages agent lifecycles, sleep/wake cycles, and child agent execution.
Dependency direction: scheduler → agent (one-way). Agent has no knowledge of Scheduler.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import uuid4

from agiwo.agent.agent import Agent
from agiwo.agent.schema import RunOutput, TerminationReason, UserInput, extract_text
from agiwo.scheduler.executor import SchedulerExecutor
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerConfig,
    SchedulerEventType,
    SchedulerOutput,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.runtime import SchedulerRuntime
from agiwo.scheduler.services import SchedulerTickEngine
from agiwo.scheduler.store import AgentStateStorage, create_agent_state_storage
from agiwo.scheduler.tool_port import SchedulerToolPort
from agiwo.scheduler.tools import (
    CancelAgentTool,
    ListAgentsTool,
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
        self._runtime = SchedulerRuntime(self._store)
        self._tool_port = SchedulerToolPort(self._store, self._guard, self._runtime)
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._scheduling_tools: list = self._create_scheduling_tools()
        self._executor = SchedulerExecutor(
            store=self._store,
            runtime=self._runtime,
            semaphore=self._semaphore,
        )
        self._tick_engine = SchedulerTickEngine(
            store=self._store,
            guard=self._guard,
            executor=self._executor,
            runtime=self._runtime,
            config=self._config,
        )

    def _create_scheduling_tools(self) -> list:
        """Create the scheduling tools that will be injected into agents."""
        return [
            SpawnAgentTool(self._tool_port),
            SleepAndWaitTool(self._tool_port),
            QuerySpawnedAgentTool(self._tool_port),
            CancelAgentTool(self._tool_port),
            ListAgentsTool(self._tool_port),
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

        if self._runtime.active_tasks:
            logger.info(
                "scheduler_waiting_for_active_tasks",
                count=len(self._runtime.active_tasks),
            )
            done, pending = await asyncio.wait(
                self._runtime.active_tasks,
                timeout=self._config.graceful_shutdown_wait_seconds,
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

    def get_registered_agent(self, state_id: str) -> Agent | None:
        """Return the in-memory agent object for a scheduler state."""
        return self._runtime.get_registered_agent(state_id)

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
        user_input: UserInput,
        *,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
        persistent: bool = False,
        agent_config_id: str | None = None,
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
            agent_config_id: Optional config template ID that created this instance.

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
            agent_config_id=agent_config_id,
            is_persistent=persistent,
            depth=0,
        )
        await self._store.save_state(state)

        if abort_signal is not None:
            self._runtime.set_abort_signal(state.id, abort_signal)

        task = asyncio.create_task(
            self._executor.run_root_agent(agent, user_input, resolved_session_id, state)
        )
        self._runtime.track_active_task(task)
        return state.id

    async def submit_task(
        self,
        state_id: str,
        task: UserInput,
        *,
        agent: Agent | None = None,
    ) -> None:
        """
        Submit a new task to a persistent root agent.

        Accepts agents in SLEEPING, COMPLETED, or FAILED states.
        Terminal states (COMPLETED/FAILED) are first reset to SLEEPING.
        The agent will be woken up on the next tick with TASK_SUBMITTED condition.

        Args:
            state_id: The state_id of the persistent root agent.
            task: The new task content.
            agent: Optional agent object to (re-)register in memory.  Required
                after a server restart when the state is persisted but the
                in-memory agent registry is empty.

        Raises:
            RuntimeError: If the agent is not persistent or in an incompatible state.
        """
        state = await self._store.get_state(state_id)
        if state is None:
            raise RuntimeError(f"Agent state '{state_id}' not found")
        if not state.is_persistent:
            raise RuntimeError(f"Agent '{state_id}' is not persistent. Use submit() instead.")

        if state.status in (AgentStateStatus.COMPLETED, AgentStateStatus.FAILED):
            await self._store.update_status(state_id, AgentStateStatus.SLEEPING)
            logger.info("reset_terminal_to_sleeping", state_id=state_id, from_status=state.status.value)
        elif state.status != AgentStateStatus.SLEEPING:
            raise RuntimeError(
                f"Agent '{state_id}' is {state.status.value}. "
                f"Cannot submit task (expected SLEEPING, COMPLETED, or FAILED)."
            )

        if agent is not None:
            self._prepare_agent(agent)

        wc = WakeCondition(type=WakeType.TASK_SUBMITTED, submitted_task=task)
        await self._store.update_status(
            state_id, AgentStateStatus.SLEEPING, wake_condition=wc
        )
        logger.info("task_submitted_to_persistent_agent", state_id=state_id)

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
        """Submit a new agent and yield text outputs as they are produced.

        Combines submit() with an output subscription so that the channel is
        created before execution starts — no outputs are lost.
        """
        self._runtime.open_output_channel(
            agent.id,
            include_child_outputs=include_child_outputs,
        )

        try:
            await self.submit(
                agent,
                user_input,
                session_id=session_id,
                abort_signal=abort_signal,
                persistent=persistent,
                agent_config_id=agent_config_id,
            )
            async for output in self._runtime.consume_output_channel(agent.id, timeout):
                yield output
        finally:
            self._runtime.close_output_channel(agent.id)

    async def submit_task_and_subscribe(
        self,
        state_id: str,
        task: UserInput,
        *,
        agent: Agent | None = None,
        timeout: float | None = None,
        include_child_outputs: bool = True,
    ) -> AsyncIterator[SchedulerOutput]:
        """Submit a new task to a persistent agent and yield outputs as they arrive.

        Combines submit_task() with an output subscription so that the channel
        is created before the scheduler tick wakes the agent.

        Pass ``agent`` to ensure the agent object is registered in memory
        (required after server restart when persistent state survives but the
        in-memory registry is empty).
        """
        self._runtime.open_output_channel(
            state_id,
            include_child_outputs=include_child_outputs,
        )

        try:
            await self.submit_task(state_id, task, agent=agent)
            async for output in self._runtime.consume_output_channel(state_id, timeout):
                yield output
        finally:
            self._runtime.close_output_channel(state_id)

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

        event = self._runtime.get_or_create_state_event(state_id)

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
                        wc = state.wake_condition
                        has_pending_task = (
                            wc is not None
                            and wc.type == WakeType.TASK_SUBMITTED
                            and wc.submitted_task is not None
                        )
                        if not has_pending_task:
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
            self._runtime.pop_state_event(state_id)

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

        await self._runtime.recursive_cancel(state_id, reason)
        logger.info("scheduler_cancel", state_id=state_id, reason=reason)
        return True

    async def steer(self, state_id: str, user_input: UserInput, *, urgent: bool = False) -> bool:
        """
        Send a steering message to an agent.

        Dispatches through the appropriate channel based on agent state:
        - RUNNING: in-memory steering queue (no persistence, consumed in current run)
        - Non-RUNNING: persisted as USER_HINT PendingEvent (consumed on next wake)
        - urgent + SLEEPING: also triggers immediate wake bypassing debounce

        Returns:
            True if the message was delivered or persisted successfully.
        """
        message = extract_text(user_input)
        if not message:
            return False

        state = await self._store.get_state(state_id)
        if state is None:
            return False

        if state.status == AgentStateStatus.RUNNING:
            agent = self._runtime.get_registered_agent(state_id)
            if agent is None:
                return False
            queue = agent.get_steering_queue()
            if queue is None:
                return False
            await queue.put(message)
            logger.info("steer_queued", state_id=state_id)
            return True

        event = PendingEvent(
            id=str(uuid4()),
            target_agent_id=state_id,
            session_id=state.session_id,
            event_type=SchedulerEventType.USER_HINT,
            payload={"hint": message},
            source_agent_id=None,
            created_at=datetime.now(timezone.utc),
        )
        await self._store.save_event(event)

        if urgent and state.status == AgentStateStatus.SLEEPING:
            await self._tick_engine.try_urgent_wake(state)

        logger.info("steer_persisted", state_id=state_id, urgent=urgent)
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

        await self._runtime.recursive_shutdown(state_id)
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
        self._runtime.register_agent(agent)

    # ──────────────────────────────────────────────────────────────────────
    # Scheduling Loop
    # ──────────────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main scheduling loop running in the background."""
        logger.info("scheduler_loop_started")
        while self._running:
            try:
                await self._tick_engine.tick()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("scheduler_tick_error")
            await asyncio.sleep(self._check_interval)
