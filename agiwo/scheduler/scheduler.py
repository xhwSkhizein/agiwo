"""
Scheduler — The orchestration layer for agent scheduling.

Manages agent lifecycles, sleep/wake cycles, and child agent execution.
Dependency direction: scheduler → agent (one-way). Agent has no knowledge of Scheduler.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from agiwo.agent.agent import Agent
from agiwo.agent.schema import RunOutput, TerminationReason
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    SchedulerConfig,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store import AgentStateStorage, create_agent_state_storage
from agiwo.scheduler.tools import QuerySpawnedAgentTool, SleepAndWaitTool, SpawnAgentTool
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
        self._check_interval = self._config.check_interval
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._active_tasks: set[asyncio.Task] = set()
        self._agents: dict[str, Agent] = {}
        self._abort_signals: dict[str, AbortSignal] = {}
        self._scheduling_tools: list = self._create_scheduling_tools()

    def _create_scheduling_tools(self) -> list:
        """Create the scheduling tools that will be injected into agents."""
        return [
            SpawnAgentTool(self._store),
            SleepAndWaitTool(self._store),
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
            done, pending = await asyncio.wait(self._active_tasks, timeout=30)
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
    ) -> RunOutput:
        """
        Submit an agent task and block until the entire orchestration completes.

        This is equivalent to submit() + wait_for().

        Args:
            agent: The agent to run.
            user_input: User message to the agent.
            session_id: Optional session ID for conversation continuity.
            timeout: Optional timeout in seconds for wait_for.
            abort_signal: Optional signal to cancel the orchestration.

        Returns:
            RunOutput with the final result.
        """
        state_id = await self.submit(agent, user_input, session_id=session_id, abort_signal=abort_signal)
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
    ) -> str:
        """
        Submit an agent task, return immediately with the state_id.

        Args:
            agent: The agent to run.
            user_input: User message to the agent.
            session_id: Optional session ID for conversation continuity.
            abort_signal: Optional signal to cancel the orchestration.

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
                f"Cannot submit concurrently. Use a different agent_id."
            )

        self._prepare_agent(agent)

        resolved_session_id = session_id or str(uuid4())
        state = AgentState(
            id=agent.id,
            session_id=resolved_session_id,
            agent_id=agent.id,
            parent_agent_id=agent.id,
            parent_state_id=None,
            status=AgentStateStatus.RUNNING,
            task=user_input,
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

    async def wait_for(
        self,
        state_id: str,
        timeout: float | None = None,
    ) -> RunOutput:
        """
        Block until a submitted task reaches COMPLETED or FAILED.

        Args:
            state_id: The state_id returned by submit().
            timeout: Optional timeout in seconds.

        Returns:
            RunOutput with the final result.
        """
        start = time.time()
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
            if timeout is not None and (time.time() - start) > timeout:
                return RunOutput(termination_reason=TerminationReason.TIMEOUT)
            await asyncio.sleep(self._check_interval)

    async def get_state(self, state_id: str) -> AgentState | None:
        """Query the current state of a scheduled agent."""
        return await self._store.get_state(state_id)

    async def cancel(self, state_id: str, reason: str = "Cancelled by user") -> bool:
        """
        Cancel a running or sleeping orchestration.

        Triggers the AbortSignal for the target agent and all its children,
        then updates their states to FAILED.

        Args:
            state_id: The state_id of the agent to cancel.
            reason: Cancellation reason.

        Returns:
            True if the agent was found and cancellation was triggered.
        """
        state = await self._store.get_state(state_id)
        if state is None:
            return False
        if state.status not in (AgentStateStatus.RUNNING, AgentStateStatus.SLEEPING, AgentStateStatus.PENDING):
            return False

        signal = self._abort_signals.get(state_id)
        if signal is not None:
            signal.abort(reason)

        children = await self._store.get_states_by_parent(state_id)
        for child in children:
            if child.status in (AgentStateStatus.RUNNING, AgentStateStatus.SLEEPING, AgentStateStatus.PENDING):
                child_signal = self._abort_signals.get(child.id)
                if child_signal is not None:
                    child_signal.abort(reason)
                await self._store.update_status(
                    child.id, AgentStateStatus.FAILED, result_summary=reason
                )

        await self._store.update_status(
            state_id, AgentStateStatus.FAILED, result_summary=reason
        )
        logger.info("scheduler_cancel", state_id=state_id, reason=reason)
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
        self._agents[agent.id] = agent

    def _create_child_agent(self, state: AgentState) -> Agent:
        """Create a child Agent by copying the parent's configuration."""
        parent = self._agents.get(state.parent_agent_id)
        if parent is None:
            raise RuntimeError(
                f"Parent agent '{state.parent_agent_id}' not found in scheduler"
            )

        overrides = state.config_overrides or {}
        child = Agent(
            id=state.agent_id,
            description=parent.description,
            model=parent.model,
            tools=list(parent.tools),
            system_prompt=overrides.get("system_prompt", parent.system_prompt),
            options=parent.options,
            hooks=parent.hooks,
        )
        self._agents[child.id] = child
        return child

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
        """Single scheduling tick: propagate signals, start pending, wake sleeping."""
        await self._propagate_signals()
        await self._start_pending()
        await self._wake_sleeping()

    async def _propagate_signals(self) -> None:
        """Propagate completion signals from child to parent agents."""
        completed = await self._store.find_unpropagated_completed()
        for state in completed:
            if state.parent_state_id is not None:
                await self._store.increment_completed_children(state.parent_state_id)
                logger.info(
                    "signal_propagated",
                    child_id=state.id,
                    parent_id=state.parent_state_id,
                )
            await self._store.mark_propagated(state.id)

    async def _start_pending(self) -> None:
        """Start all PENDING agents."""
        pending = await self._store.find_pending()
        for state in pending:
            task = asyncio.create_task(self._run_agent(state))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

    async def _wake_sleeping(self) -> None:
        """Wake agents whose conditions are satisfied."""
        now = datetime.now(timezone.utc)
        wakeable = await self._store.find_wakeable(now)
        for state in wakeable:
            task = asyncio.create_task(self._wake_agent(state))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

    # ──────────────────────────────────────────────────────────────────────
    # Agent Execution
    # ──────────────────────────────────────────────────────────────────────

    async def _run_root_agent(
        self,
        agent: Agent,
        user_input: str,
        session_id: str,
        state: AgentState,
    ) -> None:
        """Run the root agent (submitted by user)."""
        abort_signal = self._abort_signals.get(state.id)
        try:
            output = await agent.run(user_input, session_id=session_id, abort_signal=abort_signal)
            await self._handle_agent_output(state, output)
        except Exception as e:
            logger.exception(
                "root_agent_failed",
                agent_id=agent.id,
                error=str(e),
            )
            await self._store.update_status(
                state.id,
                AgentStateStatus.FAILED,
                result_summary=str(e),
            )
        finally:
            self._abort_signals.pop(state.id, None)
            self._maybe_cleanup_agent(state.id)

    async def _run_agent(self, state: AgentState) -> None:
        """Run a PENDING child agent."""
        abort_signal = AbortSignal()
        parent_signal = self._abort_signals.get(state.parent_state_id or "")
        self._abort_signals[state.id] = abort_signal

        async with self._semaphore:
            if parent_signal is not None and parent_signal.is_aborted():
                await self._store.update_status(
                    state.id, AgentStateStatus.FAILED, result_summary="Parent cancelled"
                )
                return

            await self._store.update_status(state.id, AgentStateStatus.RUNNING)
            try:
                child = self._create_child_agent(state)
                child_session_id = state.id
                output = await child.run(state.task, session_id=child_session_id, abort_signal=abort_signal)
                await self._handle_agent_output(state, output)
            except Exception as e:
                logger.exception(
                    "child_agent_failed",
                    agent_id=state.agent_id,
                    error=str(e),
                )
                await self._store.update_status(
                    state.id,
                    AgentStateStatus.FAILED,
                    result_summary=str(e),
                )
            finally:
                self._abort_signals.pop(state.id, None)
                self._maybe_cleanup_agent(state.id)

    async def _wake_agent(self, state: AgentState) -> None:
        """Wake a SLEEPING agent by running it with a wake message."""
        agent = self._agents.get(state.agent_id)
        if agent is None:
            logger.error("wake_agent_not_found", agent_id=state.agent_id)
            await self._store.update_status(
                state.id,
                AgentStateStatus.FAILED,
                result_summary=f"Agent '{state.agent_id}' not found in scheduler for wake",
            )
            return

        abort_signal = self._abort_signals.get(state.id)
        async with self._semaphore:
            await self._store.update_status(state.id, AgentStateStatus.RUNNING)
            try:
                wake_message = self._build_wake_message(state)
                output = await agent.run(wake_message, session_id=state.session_id, abort_signal=abort_signal)
                await self._handle_agent_output(state, output)
            except Exception as e:
                logger.exception(
                    "wake_agent_failed",
                    agent_id=state.agent_id,
                    error=str(e),
                )
                await self._store.update_status(
                    state.id,
                    AgentStateStatus.FAILED,
                    result_summary=str(e),
                )
            finally:
                self._abort_signals.pop(state.id, None)
                self._maybe_cleanup_agent(state.id)

    async def _handle_agent_output(
        self, state: AgentState, output: RunOutput
    ) -> None:
        """Handle agent output: mark COMPLETED or leave as SLEEPING."""
        if output.termination_reason == TerminationReason.SLEEPING:
            pass
        else:
            await self._store.update_status(
                state.id,
                AgentStateStatus.COMPLETED,
                result_summary=output.response,
            )

    def _maybe_cleanup_agent(self, agent_id: str) -> None:
        """Remove agent from registry if COMPLETED or FAILED (non-root only)."""
        # We intentionally do NOT remove root agents so they can be re-used.
        # Only remove dynamically created child agents.
        agent = self._agents.get(agent_id)
        if agent is None:
            return
        # Check if this is a child agent (has parent_agent_id != self)
        # We can detect this by checking if the agent_id contains a '_' separator
        # which is the pattern used by SpawnAgentTool.
        # Root agents (registered via _prepare_agent) won't have this pattern.
        if "_" in agent_id:
            self._agents.pop(agent_id, None)

    def _build_wake_message(self, state: AgentState) -> str:
        """Build a wake message that gives the agent context about what happened."""
        wc = state.wake_condition
        if wc is None:
            return "You have been woken up. Please continue your task."

        if wc.type == WakeType.CHILDREN_COMPLETE:
            return (
                f"All {wc.total_children} child agents have completed their tasks. "
                f"Use the query_spawned_agent tool to check their results and "
                f"synthesize a final response."
            )

        if wc.type == WakeType.DELAY:
            return (
                "The scheduled delay has elapsed. "
                "Please continue your task."
            )

        if wc.type == WakeType.INTERVAL:
            return (
                "A scheduled interval check has triggered. "
                "Please check progress and decide whether to continue waiting "
                "or produce a final result."
            )

        return "You have been woken up. Please continue your task."
