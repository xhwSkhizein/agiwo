"""Scheduler facade — lifecycle, public API, delegation to sub-modules."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agiwo.agent import (
    Agent,
    RunOutput,
    TerminationReason,
    UserInput,
    UserMessage,
)
from agiwo.scheduler._stream import route_with_stream
from agiwo.scheduler._tick import dispatch_action, tick as _tick
from agiwo.scheduler._tree_ops import cancel_subtree, shutdown_subtree
from agiwo.scheduler.commands import (
    DispatchAction,
    DispatchReason,
    RouteResult,
    RouteStreamMode,
)
from agiwo.scheduler.engine_context import EngineContext
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    ACTIVE_AGENT_STATUSES,
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerConfig,
)
from agiwo.scheduler.runner import RunnerContext, SchedulerRunner
from agiwo.scheduler.runtime_state import RuntimeState, list_all_states
from agiwo.scheduler.runtime_tools import (
    CancelAgentTool,
    ListAgentsTool,
    QuerySpawnedAgentTool,
    RetrospectToolResultTool,
    SleepAndWaitTool,
    SpawnAgentTool,
)
from agiwo.scheduler.store import create_agent_state_storage
from agiwo.scheduler.store.base import AgentStateStorage
from agiwo.scheduler.tool_control import SchedulerToolControl
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class Scheduler:
    """Scheduler: lifecycle, tick loop, API, state machine, and runtime coordination."""

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        *,
        store: AgentStateStorage | None = None,
        guard: TaskGuard | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._config = config or SchedulerConfig()
        self._store = store or create_agent_state_storage(self._config.state_storage)
        self._guard = guard or TaskGuard(
            self._config.task_limits,
            self._store,
            state_list_page_size=self._config.state_list_page_size,
        )
        self._rt = RuntimeState()
        self._tool_control = SchedulerToolControl(
            store=self._store,
            guard=self._guard,
            rt=self._rt,
            save_state=self._save_state,
            cancel_subtree=self._cancel_subtree,
            state_list_page_size=self._config.state_list_page_size,
        )
        self._scheduling_tools = (
            SpawnAgentTool(self._tool_control),
            SleepAndWaitTool(self._tool_control),
            QuerySpawnedAgentTool(self._tool_control),
            CancelAgentTool(self._tool_control),
            ListAgentsTool(self._tool_control),
            RetrospectToolResultTool(self._tool_control),
        )
        self._runner = SchedulerRunner(
            RunnerContext(
                store=self._store,
                rt=self._rt,
                notify_state_change=self._notify_state_change,
                nudge=self.nudge,
                semaphore=semaphore or asyncio.Semaphore(self._config.max_concurrent),
                state_list_page_size=self._config.state_list_page_size,
            )
        )
        self._ctx = EngineContext(
            config=self._config,
            store=self._store,
            rt=self._rt,
            guard=self._guard,
            runner=self._runner,
            save_state=self._save_state,
            track_active_task=self._track_active_task,
        )
        self._running = False
        self._loop_task: asyncio.Task | None = None

    # -- Lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._loop())
        logger.info("scheduler_started", check_interval=self._config.check_interval)

    async def stop(self) -> None:
        self._running = False

        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        active_tasks = set(self._rt.active_tasks)
        if active_tasks:
            logger.info(
                "scheduler_waiting_for_active_tasks",
                count=len(active_tasks),
            )
            _done, pending = await asyncio.wait(
                active_tasks,
                timeout=self._config.graceful_shutdown_wait_seconds,
            )
            if pending:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

        remaining_agents = list(self._rt.agents.values())
        self._rt.agents.clear()
        self._rt.canonical_agents.clear()
        self._rt.execution_handles.clear()
        if remaining_agents:
            results = await asyncio.gather(
                *[agent.close() for agent in remaining_agents],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.error("scheduler_stop_agent_close_failed", error=str(result))

        await self._store.close()
        logger.info("scheduler_stopped")

    async def __aenter__(self) -> "Scheduler":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()

    async def _loop(self) -> None:
        logger.info("scheduler_loop_started")
        failure_backoff = min(self._config.check_interval, 0.1)
        while self._running:
            try:
                await self.tick()
                failure_backoff = min(self._config.check_interval, 0.1)
                await self.wait_for_nudge(self._config.check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("scheduler_tick_error")
                await asyncio.sleep(failure_backoff)
                failure_backoff = min(max(failure_backoff * 2, 0.1), 5.0)

    # -- Public API -----------------------------------------------------------

    def get_registered_agent(self, state_id: str):
        return self._rt.agents.get(state_id)

    async def wait_for_nudge(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._rt.nudge.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return
        finally:
            self._rt.nudge.clear()

    def nudge(self) -> None:
        self._rt.nudge.set()

    async def _submit(
        self,
        agent: Agent,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
        persistent: bool = False,
        agent_config_id: str | None = None,
    ) -> str:
        # Get or create lock for this agent_id to prevent concurrent submit
        lock = self._rt.state_locks.setdefault(agent.id, asyncio.Lock())
        async with lock:
            # Double-check after acquiring lock
            existing = await self._store.get_state(agent.id)
            if existing is not None and existing.status in ACTIVE_AGENT_STATUSES:
                raise RuntimeError(
                    f"Agent '{agent.id}' is already active (status={existing.status.value}). "
                    f"Cannot submit concurrently. Use a different agent_id or enqueue_input()."
                )

            await self._ensure_root_runtime_agent(agent, agent.id)
            resolved_session_id = session_id or str(uuid4())
            state = AgentState(
                id=agent.id,
                session_id=resolved_session_id,
                status=AgentStateStatus.RUNNING,
                task=user_input,
                agent_config_id=agent_config_id,
                is_persistent=persistent,
                depth=0,
            )
            await self._save_state(state)
            if abort_signal is not None:
                self._rt.abort_signals[state.id] = abort_signal

            await dispatch_action(
                self._ctx,
                DispatchAction(
                    state=state,
                    reason=DispatchReason.ROOT_SUBMIT,
                    input_override=user_input,
                ),
            )
            self.nudge()
            return state.id

    async def enqueue_input(
        self,
        state_id: str,
        user_input: UserInput,
        *,
        agent: Agent | None = None,
    ) -> None:
        # Get or create lock for this state_id to prevent concurrent enqueue
        lock = self._rt.state_locks.setdefault(state_id, asyncio.Lock())
        async with lock:
            state = await self._store.get_state(state_id)
            if state is None:
                raise RuntimeError(f"Agent state '{state_id}' not found")
            if not state.is_root or not state.is_persistent:
                raise RuntimeError(
                    f"Agent '{state_id}' is not persistent. Use submit() instead."
                )
            if not state.can_accept_enqueue_input():
                raise RuntimeError(
                    f"Agent '{state_id}' is {state.status.value}. "
                    f"Cannot enqueue input (expected IDLE or FAILED)."
                )
            if agent is not None:
                await self._ensure_root_runtime_agent(agent, state_id)

            await self._save_state(state.with_queued(pending_input=user_input))
            self.nudge()

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
        stream_mode: RouteStreamMode = RouteStreamMode.RUN_END,
    ) -> RouteResult:
        return await self._route_root_input_impl(
            user_input,
            agent=agent,
            state_id=state_id,
            session_id=session_id,
            abort_signal=abort_signal,
            persistent=persistent,
            agent_config_id=agent_config_id,
            timeout=timeout,
            include_child_events=include_child_events,
            close_on_root_run_end=stream_mode == RouteStreamMode.RUN_END,
        )

    async def _route_root_input_impl(
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
        close_on_root_run_end: bool,
    ) -> RouteResult:
        lookup_id = state_id or agent.id
        deadline = None if timeout is None else time.monotonic() + timeout

        async def do_submit() -> str:
            return await self._submit(
                agent,
                user_input,
                session_id=session_id,
                abort_signal=abort_signal,
                persistent=persistent,
                agent_config_id=agent_config_id,
            )

        current_state = await self._resolve_routable_state(lookup_id, deadline)

        if current_state is None:
            return await route_with_stream(
                self._ctx,
                root_state_id=agent.id,
                action="submitted",
                timeout=self._deadline_remaining(deadline) if deadline else timeout,
                include_child_events=include_child_events,
                close_on_root_run_end=close_on_root_run_end,
                operation=do_submit,
            )

        if current_state.status in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.WAITING,
            AgentStateStatus.QUEUED,
        ):
            if current_state.status == AgentStateStatus.RUNNING:
                return await self._steer_into_running(
                    current_state,
                    user_input,
                )

            return await self._steer_with_stream(
                current_state,
                user_input,
                timeout=timeout,
                include_child_events=include_child_events,
                close_on_root_run_end=close_on_root_run_end,
            )

        if (
            current_state.is_root
            and current_state.is_persistent
            and current_state.status in (AgentStateStatus.IDLE, AgentStateStatus.FAILED)
        ):
            return await route_with_stream(
                self._ctx,
                root_state_id=current_state.id,
                action="enqueued",
                timeout=timeout,
                include_child_events=include_child_events,
                close_on_root_run_end=close_on_root_run_end,
                operation=lambda: self._enqueue_and_return_state_id(
                    state_id=current_state.id,
                    agent=agent,
                    user_input=user_input,
                ),
            )

        return await route_with_stream(
            self._ctx,
            root_state_id=agent.id,
            action="submitted",
            timeout=timeout,
            include_child_events=include_child_events,
            close_on_root_run_end=close_on_root_run_end,
            operation=do_submit,
        )

    async def wait_for(
        self,
        state_id: str,
        timeout: float | None = None,
    ) -> RunOutput:
        start = time.monotonic()
        event = asyncio.Event()
        waiters = self._rt.waiters.setdefault(state_id, set())
        waiters.add(event)

        try:
            while True:
                state = await self._store.get_state(state_id)
                if state is not None:
                    if state.last_run_result is not None and (
                        state.status
                        in (
                            AgentStateStatus.IDLE,
                            AgentStateStatus.COMPLETED,
                            AgentStateStatus.FAILED,
                        )
                    ):
                        last_run_result = state.last_run_result
                        return RunOutput(
                            run_id=last_run_result.run_id,
                            response=(
                                last_run_result.summary
                                if last_run_result.error is None
                                else None
                            ),
                            error=last_run_result.error,
                            termination_reason=last_run_result.termination_reason,
                        )

                remaining = None
                if timeout is not None:
                    elapsed = time.monotonic() - start
                    if elapsed >= timeout:
                        return RunOutput(termination_reason=TerminationReason.TIMEOUT)
                    remaining = timeout - elapsed

                try:
                    await asyncio.wait_for(event.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return RunOutput(termination_reason=TerminationReason.TIMEOUT)
                event.clear()
        finally:
            waiters = self._rt.waiters.get(state_id)
            if waiters is not None:
                waiters.discard(event)
                if not waiters:
                    self._rt.waiters.pop(state_id, None)

    async def get_state(self, state_id: str) -> AgentState | None:
        return await self._store.get_state(state_id)

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
        return await self._store.list_states(
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
        return await self._store.list_events(
            target_agent_id=target_agent_id,
            session_id=session_id,
        )

    async def get_stats(self) -> dict[str, int]:
        states = await list_all_states(self._store)
        counts: dict[str, int] = {
            "pending": 0,
            "running": 0,
            "waiting": 0,
            "idle": 0,
            "queued": 0,
            "completed": 0,
            "failed": 0,
        }
        for state in states:
            counts[state.status.value] += 1
        return {"total": len(states), **counts}

    async def cancel(self, state_id: str, reason: str = "Cancelled by user") -> bool:
        state = await self._store.get_state(state_id)
        if state is None or not state.is_active():
            return False
        await self._cancel_subtree(state_id, reason)
        return True

    async def _steer_into_running(
        self,
        state: AgentState,
        user_input: UserInput,
    ) -> RouteResult:
        steered = await self.steer(state.id, user_input, urgent=False)
        if not steered:
            refreshed = await self._store.get_state(state.id)
            if refreshed is not None and refreshed.status in ACTIVE_AGENT_STATUSES:
                raise RuntimeError(
                    f"Failed to steer active scheduler state '{state.id}'"
                )
        return RouteResult(action="steered", state_id=state.id)

    async def _steer_with_stream(
        self,
        state: AgentState,
        user_input: UserInput,
        *,
        timeout: float | None,
        include_child_events: bool,
        close_on_root_run_end: bool,
    ) -> RouteResult:
        urgent = state.status == AgentStateStatus.WAITING
        sid = state.id

        async def _do_steer() -> str:
            steered = await self.steer(sid, user_input, urgent=urgent)
            if not steered:
                raise RuntimeError(f"Failed to steer scheduler state '{sid}'")
            return sid

        return await route_with_stream(
            self._ctx,
            root_state_id=sid,
            action="steered",
            timeout=timeout,
            include_child_events=include_child_events,
            close_on_root_run_end=close_on_root_run_end,
            operation=_do_steer,
        )

    async def steer(
        self,
        state_id: str,
        user_input: UserInput,
        *,
        urgent: bool = False,
    ) -> bool:
        message = UserMessage.from_value(user_input)
        if not message.has_content():
            return False

        state = await self._store.get_state(state_id)
        if state is None:
            return False

        if state.status == AgentStateStatus.RUNNING:
            handle = self._rt.execution_handles.get(state_id)
            if handle is None:
                return False
            return await handle.steer(message)

        event = PendingEvent.create_user_hint(
            id=str(uuid4()),
            target_agent_id=state_id,
            session_id=state.session_id,
            user_input=UserMessage.to_storage_value(message),
            created_at=datetime.now(timezone.utc),
            urgent=urgent,
        )
        await self._store.save_event(event)
        self.nudge()
        return True

    async def shutdown(self, state_id: str) -> bool:
        state = await self._store.get_state(state_id)
        if state is None or not state.is_active():
            return False
        await self._shutdown_subtree(state_id)
        self.nudge()
        return True

    async def rebind_agent(self, state_id: str, agent: Agent) -> bool:
        state = await self._store.get_state(state_id)
        if state is not None and state.status not in (
            AgentStateStatus.IDLE,
            AgentStateStatus.COMPLETED,
            AgentStateStatus.FAILED,
        ):
            return False

        await self._ensure_root_runtime_agent(agent, state_id)
        return True

    async def tick(self) -> None:
        await _tick(self._ctx)

    # -- Internal helpers (used by extracted modules) --------------------------

    async def _save_state(self, state: AgentState) -> None:
        await self._store.save_state(state)
        self._notify_state_change(state.id)
        self.nudge()

    async def _cancel_subtree(self, state_id: str, reason: str) -> None:
        await cancel_subtree(self._ctx, state_id, reason)

    async def _shutdown_subtree(self, state_id: str) -> None:
        await shutdown_subtree(self._ctx, state_id)

    def _track_active_task(self, task: asyncio.Task) -> None:
        self._rt.active_tasks.add(task)
        task.add_done_callback(self._rt.active_tasks.discard)

    def _notify_state_change(self, state_id: str) -> None:
        for waiter in self._rt.waiters.get(state_id, set()):
            waiter.set()

    async def _resolve_routable_state(
        self,
        state_id: str,
        deadline: float | None,
    ) -> AgentState | None:
        while True:
            state = await self._store.get_state(state_id)
            if state is None:
                return None
            if state.status != AgentStateStatus.PENDING:
                return state
            state = await self._wait_until_not_pending(state.id, deadline=deadline)
            if state is None:
                return None

    async def _wait_until_not_pending(
        self,
        state_id: str,
        *,
        deadline: float | None,
    ) -> AgentState | None:
        event = asyncio.Event()
        waiters = self._rt.waiters.setdefault(state_id, set())
        waiters.add(event)
        try:
            while True:
                state = await self._store.get_state(state_id)
                if state is None or state.status != AgentStateStatus.PENDING:
                    return state
                try:
                    await asyncio.wait_for(
                        event.wait(),
                        timeout=self._deadline_remaining(deadline),
                    )
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(
                        f"Timed out waiting for scheduler state '{state_id}' "
                        "to become routable"
                    ) from exc
                event.clear()
        finally:
            waiters = self._rt.waiters.get(state_id)
            if waiters is not None:
                waiters.discard(event)
                if not waiters:
                    self._rt.waiters.pop(state_id, None)

    async def _ensure_root_runtime_agent(
        self,
        canonical_agent: Agent,
        state_id: str,
    ) -> Agent:
        """Ensure a scheduler-managed runtime agent exists for ``state_id``.

        **Identity rule (strict reuse):** if the cached canonical agent for
        ``state_id`` is the *same Python object* as the caller-supplied one,
        we reuse the existing runtime agent (preserving its ``run_step_storage``
        / ``trace_storage`` / workspace state across turns).  Otherwise we
        build a new runtime agent (clone + inject scheduler system tools),
        close the previous runtime agent, and record the new canonical.

        Common same-``state_id`` races are already constrained by
        ``submit()`` rejecting concurrent roots in ``ACTIVE_AGENT_STATUSES``
        and by ``_cleanup_after_run()`` eagerly closing non-persistent roots.
        A rare concurrent rebind/pop edge case would still need a per-state
        runtime lock if future changes keep returned runtime-agent references
        alive across canonical swaps.
        """
        cached_canonical = self._rt.canonical_agents.get(state_id)
        cached_runtime = self._rt.agents.get(state_id)
        if cached_canonical is canonical_agent and cached_runtime is not None:
            return cached_runtime

        runtime_agent = Agent(
            canonical_agent.config,
            id=state_id,
            model=canonical_agent.model,
            tools=list(canonical_agent.extra_tools) or None,
            hooks=canonical_agent.hooks,
        )
        runtime_agent._inject_system_tools(list(self._scheduling_tools))

        self._rt.agents[state_id] = runtime_agent
        self._rt.canonical_agents[state_id] = canonical_agent
        if cached_runtime is not None:
            try:
                await cached_runtime.close()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "scheduler_root_runtime_agent_close_failed",
                    state_id=state_id,
                )
        return runtime_agent

    async def _enqueue_and_return_state_id(
        self,
        *,
        state_id: str,
        agent: Agent,
        user_input: UserInput,
    ) -> str:
        await self.enqueue_input(state_id, user_input, agent=agent)
        return state_id

    def _deadline_remaining(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0
        return remaining


__all__ = ["Scheduler"]
