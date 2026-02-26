"""
Scheduler — The orchestration layer for agent scheduling.

Manages agent lifecycles, sleep/wake cycles, and child agent execution.
Dependency direction: scheduler → agent (one-way). Agent has no knowledge of Scheduler.
"""

import asyncio
import copy
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from agiwo.agent.agent import Agent
from agiwo.agent.schema import RunOutput, TerminationReason
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
        self._scheduling_tools: list = self._create_scheduling_tools()

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
            agent_id=agent.id,
            parent_agent_id=agent.id,
            parent_state_id=None,
            status=AgentStateStatus.RUNNING,
            task=user_input,
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
        """
        start = time.time()
        initial_state = await self._store.get_state(state_id)
        is_persistent = initial_state.is_persistent if initial_state else False

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
            if timeout is not None and (time.time() - start) > timeout:
                return RunOutput(termination_reason=TerminationReason.TIMEOUT)
            await asyncio.sleep(self._check_interval)

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
        parent = self._agents.get(state.parent_agent_id)
        if parent is None:
            raise RuntimeError(
                f"Parent agent '{state.parent_agent_id}' not found in scheduler"
            )

        overrides = state.config_overrides or {}
        child_options = copy.copy(parent.options) if parent.options else None
        if child_options is not None:
            child_options.enable_termination_summary = True

        child = Agent(
            name=parent.name,
            description=parent.description,
            model=parent.model,
            id=state.agent_id,
            tools=list(parent.tools),
            system_prompt=overrides.get("system_prompt", parent.system_prompt),
            options=child_options,
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
        """Single scheduling tick: propagate signals, enforce timeouts, start pending, wake sleeping."""
        await self._propagate_signals()
        await self._enforce_timeouts()
        await self._start_pending()
        await self._wake_sleeping()

    async def _propagate_signals(self) -> None:
        """Propagate completion/failure signals from child to parent agents."""
        completed = await self._store.find_unpropagated_completed()
        for state in completed:
            if state.parent_state_id is not None:
                await self._store.mark_child_completed(state.parent_state_id, state.id)
                logger.info(
                    "signal_propagated",
                    child_id=state.id,
                    parent_id=state.parent_state_id,
                    child_status=state.status.value,
                )
            await self._store.mark_propagated(state.id)

    async def _enforce_timeouts(self) -> None:
        """Find timed-out SLEEPING agents and wake them for summarization."""
        now = datetime.now(timezone.utc)
        timed_out = await self._guard.find_timed_out(now)
        for state in timed_out:
            task = asyncio.create_task(self._wake_for_timeout(state))
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)

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
            rejection = await self._guard.check_wake(state)
            if rejection is not None:
                logger.warning(
                    "wake_rejected",
                    agent_id=state.agent_id,
                    reason=rejection,
                )
                await self._store.update_status(
                    state.id,
                    AgentStateStatus.FAILED,
                    result_summary=f"Wake rejected: {rejection}",
                )
                continue
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
            output = await agent.run(
                user_input, session_id=session_id, abort_signal=abort_signal
            )
            await self._handle_agent_output(state, output)
        except Exception as e:
            logger.exception(
                "root_agent_failed",
                agent_id=agent.id,
                state_id=state.id,
                state_status=state.status.value,
                error=str(e),
                error_type=type(e).__name__,
            )
            await self._store.update_status(
                state.id,
                AgentStateStatus.FAILED,
                result_summary=str(e),
            )
        finally:
            self._abort_signals.pop(state.id, None)
            self._maybe_cleanup_agent(state)

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
                output = await child.run(
                    state.task, session_id=child_session_id, abort_signal=abort_signal
                )
                await self._handle_agent_output(state, output)
            except Exception as e:
                logger.exception(
                    "child_agent_failed",
                    agent_id=state.agent_id,
                    state_id=state.id,
                    parent_state_id=state.parent_state_id,
                    depth=state.depth,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await self._store.update_status(
                    state.id,
                    AgentStateStatus.FAILED,
                    result_summary=str(e),
                )
            finally:
                self._abort_signals.pop(state.id, None)
                self._maybe_cleanup_agent(state)

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

        await self._store.increment_wake_count(state.id)
        abort_signal = self._abort_signals.get(state.id)
        async with self._semaphore:
            await self._store.update_status(state.id, AgentStateStatus.RUNNING)
            try:
                wake_message = await self._build_wake_message(state)
                output = await agent.run(
                    wake_message, session_id=state.session_id, abort_signal=abort_signal
                )
                await self._handle_agent_output(state, output)
            except Exception as e:
                logger.exception(
                    "wake_agent_failed",
                    agent_id=state.agent_id,
                    state_id=state.id,
                    wake_count=state.wake_count,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await self._store.update_status(
                    state.id,
                    AgentStateStatus.FAILED,
                    result_summary=str(e),
                )
            finally:
                self._abort_signals.pop(state.id, None)
                self._maybe_cleanup_agent(state)

    async def _wake_for_timeout(self, state: AgentState) -> None:
        """Wake a timed-out SLEEPING agent so it can produce a summary report."""
        agent = self._agents.get(state.agent_id)
        if agent is None:
            logger.error("timeout_wake_agent_not_found", agent_id=state.agent_id)
            await self._store.update_status(
                state.id,
                AgentStateStatus.FAILED,
                result_summary=f"Agent '{state.agent_id}' not found for timeout wake",
            )
            return

        partial_results = await self._collect_child_results(state)
        wc = state.wake_condition
        total = len(wc.wait_for) if wc else 0
        done = len(partial_results)

        wake_msg = (
            f"Wait timeout reached.\n"
            f"Completed children: {done}/{total}\n\n"
        )
        if partial_results:
            wake_msg += "## Child Agent Results\n"
            for cid, summary in partial_results.items():
                wake_msg += f"- [{cid}] {summary}\n"
            wake_msg += "\n"
        wake_msg += "Please produce a summary report with whatever results are available."

        await self._store.increment_wake_count(state.id)
        async with self._semaphore:
            await self._store.update_status(state.id, AgentStateStatus.RUNNING, wake_condition=None)
            try:
                output = await agent.run(
                    wake_msg, session_id=state.session_id
                )
                await self._handle_agent_output(state, output)
            except Exception as e:
                logger.exception(
                    "timeout_wake_failed",
                    agent_id=state.agent_id,
                    state_id=state.id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await self._store.update_status(
                    state.id,
                    AgentStateStatus.FAILED,
                    result_summary=str(e),
                )

    async def _handle_agent_output(self, state: AgentState, output: RunOutput) -> None:
        """Handle agent output: SLEEPING, persistent idle, PERIODIC reschedule, or COMPLETED."""
        if output.termination_reason == TerminationReason.SLEEPING:
            return

        refreshed = await self._store.get_state(state.id)
        is_persistent = refreshed.is_persistent if refreshed else state.is_persistent
        original_wc = state.wake_condition

        if original_wc is not None and original_wc.type == WakeType.PERIODIC:
            secs = original_wc.to_seconds()
            if secs is not None:
                now = datetime.now(timezone.utc)
                new_wc = WakeCondition(
                    type=WakeType.PERIODIC,
                    time_value=original_wc.time_value,
                    time_unit=original_wc.time_unit,
                    wakeup_at=now + timedelta(seconds=secs),
                    timeout_at=original_wc.timeout_at,
                )
                await self._store.update_status(
                    state.id,
                    AgentStateStatus.SLEEPING,
                    wake_condition=new_wc,
                    result_summary=output.response,
                )
                return

        if is_persistent:
            await self._store.update_status(
                state.id,
                AgentStateStatus.SLEEPING,
                wake_condition=WakeCondition(type=WakeType.TASK_SUBMITTED),
                result_summary=output.response,
            )
            return

        await self._store.update_status(
            state.id,
            AgentStateStatus.COMPLETED,
            result_summary=output.response,
        )

    def _maybe_cleanup_agent(self, state: AgentState) -> None:
        """Remove agent from registry if COMPLETED or FAILED (non-root only)."""
        if state.parent_state_id is None:
            return
        self._agents.pop(state.agent_id, None)

    # ──────────────────────────────────────────────────────────────────────
    # Wake Message Building
    # ──────────────────────────────────────────────────────────────────────

    async def _build_wake_message(self, state: AgentState) -> str:
        """Build a wake message with auto-injected child results."""
        wc = state.wake_condition
        if wc is None:
            return "You have been woken up. Please continue your task."

        if wc.type == WakeType.WAITSET:
            child_results = await self._collect_child_results(state)
            msg = f"Child agents completed ({len(child_results)}/{len(wc.wait_for)}).\n\n"
            if child_results:
                msg += "## Child Agent Results\n"
                for cid, summary in child_results.items():
                    msg += f"- [{cid}] {summary}\n"
                msg += "\n"
            msg += "Please synthesize a final response based on the above results."
            return msg

        if wc.type == WakeType.TIMER:
            return "The scheduled delay has elapsed. Please continue your task."

        if wc.type == WakeType.PERIODIC:
            return (
                "A scheduled periodic check has triggered. "
                "Please check progress and decide whether to continue waiting "
                "or produce a final result."
            )

        if wc.type == WakeType.TASK_SUBMITTED:
            task = wc.submitted_task or ""
            return f"New task submitted:\n\n{task}"

        return "You have been woken up. Please continue your task."

    async def _collect_child_results(self, state: AgentState) -> dict[str, str]:
        """Collect results from child agents referenced in the waitset."""
        results: dict[str, str] = {}
        wc = state.wake_condition
        child_ids = wc.wait_for if wc else []
        if not child_ids:
            children = await self._store.get_states_by_parent(state.id)
            child_ids = [c.id for c in children]
        for cid in child_ids:
            child = await self._store.get_state(cid)
            if child is not None:
                summary = child.result_summary or f"status={child.status.value}"
                results[cid] = summary
        return results

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

        await self._store.update_status(
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
            await self._store.update_status(
                state_id,
                AgentStateStatus.FAILED,
                result_summary="Shutdown before execution",
            )
