"""Scheduler facade — lifecycle, public API, delegation to sub-modules."""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agiwo.agent import (
    Agent,
    RunOutput,
    RunStatus,
    UserInput,
    UserMessage,
)
from agiwo.agent.models.run import RUN_TERMINAL_STATUSES
from agiwo.scheduler._runtime_agents import ensure_root_runtime_agent
from agiwo.scheduler.route_stream import route_with_stream
from agiwo.scheduler._tick import dispatch_action, tick as _tick
from agiwo.scheduler._tree_ops import cancel_subtree, shutdown_subtree
from agiwo.scheduler._wait import (
    deadline_remaining,
    resolve_routable_state,
    wait_for_state_result,
)
from agiwo.scheduler.commands import (
    DispatchAction,
    DispatchReason,
    RouteResult,
    RouteStreamMode,
)
from agiwo.scheduler.execution import (
    ExecutionDispatchResult,
    ExecutionTreeNode,
    SchedulerExecutionRequest,
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
    ForkChildAgentTool,
    ListAgentsTool,
    QuerySpawnedAgentTool,
    SleepAndWaitTool,
    SpawnChildAgentTool,
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
            SpawnChildAgentTool(self._tool_control),
            ForkChildAgentTool(self._tool_control),
            SleepAndWaitTool(self._tool_control),
            QuerySpawnedAgentTool(self._tool_control),
            CancelAgentTool(self._tool_control),
            ListAgentsTool(self._tool_control),
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
        self._rt.prepared_resumes.clear()
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
        UserMessage.require_user_provided(user_input)
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
        UserMessage.require_user_provided(user_input)
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
        close_on_root_run_end = stream_mode == RouteStreamMode.RUN_END
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

        current_state = await resolve_routable_state(
            store=self._store,
            rt=self._rt,
            state_id=lookup_id,
            deadline=deadline,
        )

        if current_state is None:
            return await route_with_stream(
                self._ctx,
                root_state_id=agent.id,
                action="submitted",
                timeout=(
                    deadline_remaining(deadline) if deadline is not None else timeout
                ),
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
        return await wait_for_state_result(
            store=self._store,
            rt=self._rt,
            state_id=state_id,
            timeout=timeout,
        )

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
        UserMessage.require_user_provided(user_input)
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

    # -- Session / root execution facade (ADR 0048) ---------------------------

    async def dispatch_execution(
        self,
        agent: Agent,
        request: SchedulerExecutionRequest,
    ) -> ExecutionDispatchResult:
        """Start or attach a root Run with a preallocated ``run_id``.

        Deterministic for duplicate delivery:
        - no RunStarted yet → start
        - RUNNING with live handle → attach
        - terminal → do not restart
        """
        run_id = request.execution.run_id
        state_id = request.state_id
        await self._ensure_root_runtime_agent(agent, state_id)

        existing = await self._rt.get_run_view(run_id)
        if existing is not None:
            if existing.status in RUN_TERMINAL_STATUSES:
                return ExecutionDispatchResult(
                    state_id=state_id,
                    run_id=run_id,
                    attached=False,
                    status=existing.status,
                )
            if existing.status == RunStatus.RUNNING:
                handle = self._rt.execution_handles.get(state_id)
                if handle is not None and handle.run_id == run_id:
                    return ExecutionDispatchResult(
                        state_id=state_id,
                        run_id=run_id,
                        attached=True,
                        status=RunStatus.RUNNING,
                    )
                # RunLog says RUNNING but no handle — treat as attachable/no restart
                return ExecutionDispatchResult(
                    state_id=state_id,
                    run_id=run_id,
                    attached=True,
                    status=RunStatus.RUNNING,
                )

        lock = self._rt.state_locks.setdefault(state_id, asyncio.Lock())
        async with lock:
            # Re-check after lock (another worker may have started).
            existing = await self._rt.get_run_view(run_id)
            if existing is not None:
                return ExecutionDispatchResult(
                    state_id=state_id,
                    run_id=run_id,
                    attached=True,
                    status=existing.status,
                )

            state = await self._store.get_state(state_id)
            # Persistent roots rest in IDLE between Assignments; only busy
            # statuses block a new preallocated root Run.
            busy = frozenset(
                {
                    AgentStateStatus.PENDING,
                    AgentStateStatus.RUNNING,
                    AgentStateStatus.WAITING,
                    AgentStateStatus.QUEUED,
                }
            )
            if state is not None and state.status in busy:
                raise RuntimeError(
                    f"Agent '{state_id}' is already active "
                    f"(status={state.status.value}); cannot dispatch another root"
                )

            if state is None:
                state = AgentState(
                    id=state_id,
                    session_id=request.session_id,
                    status=AgentStateStatus.RUNNING,
                    task=(
                        request.user_input
                        if request.user_input is not None
                        else UserMessage.from_system("")
                    ),
                    agent_config_id=request.agent_config_id,
                    is_persistent=request.persistent,
                    depth=0,
                )
            else:
                state = state.with_updates(
                    session_id=request.session_id,
                    status=AgentStateStatus.RUNNING,
                    task=(
                        request.user_input
                        if request.user_input is not None
                        else UserMessage.from_system("")
                    ),
                    agent_config_id=request.agent_config_id
                    if request.agent_config_id is not None
                    else state.agent_config_id,
                    is_persistent=request.persistent,
                    pending_input=None,
                    wake_condition=None,
                    last_run_result=None,
                )
            await self._save_state(state)
            await dispatch_action(
                self._ctx,
                DispatchAction(
                    state=state,
                    reason=DispatchReason.SESSION_ROOT,
                    input_override=request.user_input,
                    execution_request=request.execution,
                ),
            )
            self.nudge()
            return ExecutionDispatchResult(
                state_id=state_id,
                run_id=run_id,
                attached=False,
                status=RunStatus.RUNNING,
            )

    async def get_run_view(self, run_id: str):
        return await self._rt.get_run_view(run_id)

    async def list_run_log_entries(
        self, run_id: str, *, kinds=None, limit: int = 10_000
    ):
        return await self._rt.list_run_log_entries(run_id, kinds=kinds, limit=limit)

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        view = await self.get_run_view(run_id)
        return view.status if view is not None else None

    async def list_execution_tree(
        self,
        root_run_id: str,
    ) -> list[ExecutionTreeNode]:
        """List the root run and its direct children (depth-1 only)."""
        root = await self.get_run_view(root_run_id)
        if root is None:
            return []
        nodes = [
            ExecutionTreeNode(
                run_id=root.run_id,
                agent_id=root.agent_id,
                status=root.status,
                parent_run_id=root.parent_run_id,
                run_tree_role=root.run_tree_role,
                depth=0,
            )
        ]
        seen = {root.run_id}
        # Children share the same session; scan runtime agents for descendants.
        for agent in self._rt.agents.values():
            views = await agent.run_log_storage.list_run_views(
                session_id=root.session_id, limit=1000
            )
            for view in views:
                if view.parent_run_id != root_run_id or view.run_id in seen:
                    continue
                seen.add(view.run_id)
                nodes.append(
                    ExecutionTreeNode(
                        run_id=view.run_id,
                        agent_id=view.agent_id,
                        status=view.status,
                        parent_run_id=view.parent_run_id,
                        run_tree_role=view.run_tree_role,
                        depth=1,
                    )
                )
        return nodes

    async def request_recoverable_pause(
        self,
        run_ids: list[str],
        reason: str,
        *,
        timeout: float = 120.0,
    ) -> None:
        """Cooperatively pause runs at the next safe boundary (not cancel)."""
        targets = {rid for rid in run_ids if rid}
        if not targets:
            return
        for handle in list(self._rt.execution_handles.values()):
            if handle.run_id in targets:
                handle.request_pause(reason)
        deadline = time.monotonic() + timeout
        pending = set(targets)
        while pending and time.monotonic() < deadline:
            done: set[str] = set()
            for run_id in pending:
                status = await self.get_run_status(run_id)
                if status is None:
                    done.add(run_id)
                elif status is RunStatus.PAUSED:
                    done.add(run_id)
                elif status in RUN_TERMINAL_STATUSES:
                    # Terminal without pause: treat as converged for barrier.
                    done.add(run_id)
            pending -= done
            if pending:
                await asyncio.sleep(0.05)
        if pending:
            raise TimeoutError(
                f"recoverable pause timed out for run_ids={sorted(pending)}"
            )

    async def prepare_resume(self, run_ids: list[str]) -> None:
        """Validate and park resume runtimes; status stays PAUSED until release."""
        prepared: dict[str, tuple[Agent, str, str]] = {}
        try:
            for run_id in run_ids:
                view = await self.get_run_view(run_id)
                if view is None:
                    raise ValueError(f"unknown run_id={run_id!r}")
                if view.status is not RunStatus.PAUSED:
                    raise ValueError(
                        f"run {run_id!r} status={view.status.value} is not PAUSED"
                    )
                agent = self._rt.find_agent(view.agent_id)
                if agent is None:
                    raise ValueError(
                        f"no scheduler agent available to resume run {run_id!r}"
                    )
                checkpoint_id = await agent.prepare_resume(
                    run_id=run_id, session_id=view.session_id
                )
                prepared[run_id] = (agent, view.session_id, checkpoint_id)
        except Exception:
            self._rt.prepared_resumes.clear()
            raise
        self._rt.prepared_resumes.update(prepared)

    async def release_resume_barrier(self) -> None:
        """Release prepared resumes and continue each paused run."""
        prepared = dict(self._rt.prepared_resumes)
        self._rt.prepared_resumes.clear()
        if not prepared:
            return
        tasks = [
            agent.resume_paused_run(run_id=run_id, session_id=session_id)
            for run_id, (agent, session_id, _checkpoint) in prepared.items()
        ]
        await asyncio.gather(*tasks)

    async def inject_user_message(
        self,
        root_run_id: str,
        message: UserInput,
    ) -> None:
        """Inject a system-notice user message into the live root Run context."""
        view = await self.get_run_view(root_run_id)
        if view is None:
            raise ValueError(f"unknown run_id={root_run_id!r}")
        handle = self._rt.find_handle_by_run_id(root_run_id)
        if handle is None:
            raise ValueError(
                f"no live execution handle for run_id={root_run_id!r}; "
                "cannot inject into an inactive Run"
            )
        ok = await handle.inject_system_user_message(message)
        if not ok:
            raise ValueError(f"inject rejected for run_id={root_run_id!r}")

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

    async def _ensure_root_runtime_agent(
        self,
        canonical_agent: Agent,
        state_id: str,
    ) -> Agent:
        """Ensure a scheduler-managed runtime agent exists for ``state_id``.

        **Identity rule (strict reuse):** if the cached canonical agent for
        ``state_id`` is the *same Python object* as the caller-supplied one,
        we reuse the existing runtime agent (preserving its ``run_log_storage``
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
        return await ensure_root_runtime_agent(
            rt=self._rt,
            scheduling_tools=self._scheduling_tools,
            canonical_agent=canonical_agent,
            state_id=state_id,
        )

    async def _enqueue_and_return_state_id(
        self,
        *,
        state_id: str,
        agent: Agent,
        user_input: UserInput,
    ) -> str:
        await self.enqueue_input(state_id, user_input, agent=agent)
        return state_id


__all__ = ["Scheduler"]
