import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from agiwo.agent import (
    Agent,
    AgentStreamItem,
    RunOutput,
    TerminationReason,
    UserInput,
    UserMessage,
)
from agiwo.scheduler.commands import (
    DispatchAction,
    DispatchReason,
    RouteResult,
)
from agiwo.scheduler.formatting import SHUTDOWN_SUMMARY_TASK
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    ACTIVE_AGENT_STATUSES,
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerConfig,
    SchedulerEventType,
)
from agiwo.scheduler.runner import RunnerContext, SchedulerRunner
from agiwo.scheduler.runtime_tools import (
    CancelAgentTool,
    ListAgentsTool,
    QuerySpawnedAgentTool,
    SleepAndWaitTool,
    SpawnAgentTool,
)
from agiwo.scheduler.runtime_state import (
    RuntimeState,
    build_mailbox_input,
    group_events,
    list_all_states,
    select_debounced_event_targets,
)
from agiwo.scheduler.stream import (
    close_stream_channel,
    consume_stream_channel,
    finish_stream_channel,
    open_stream_channel,
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
        self._guard = guard or TaskGuard(self._config.task_limits, self._store)
        self._rt = RuntimeState()
        self._tool_control = SchedulerToolControl(
            store=self._store,
            guard=self._guard,
            rt=self._rt,
            save_state=self._save_state,
            cancel_subtree=self._cancel_subtree,
        )
        self._scheduling_tools = (
            SpawnAgentTool(self._tool_control),
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
                scheduling_tools=self._scheduling_tools,
            )
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
        state_id = await self.submit(
            agent,
            user_input,
            session_id=session_id,
            abort_signal=abort_signal,
            persistent=persistent,
        )
        return await self.wait_for(state_id, timeout=timeout)

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
        prepared_agent = await self._prepare_root_agent(agent, agent.id)
        existing = await self._store.get_state(prepared_agent.id)
        if existing is not None and existing.status in ACTIVE_AGENT_STATUSES:
            raise RuntimeError(
                f"Agent '{prepared_agent.id}' is already active (status={existing.status.value}). "
                f"Cannot submit concurrently. Use a different agent_id or enqueue_input()."
            )

        self._rt.agents[prepared_agent.id] = prepared_agent
        resolved_session_id = session_id or str(uuid4())
        state = AgentState(
            id=prepared_agent.id,
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

        await self._dispatch_action(
            DispatchAction(
                state=state,
                reason=DispatchReason.ROOT_SUBMIT,
                input_override=user_input,
            )
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
            prepared = await self._prepare_root_agent(agent, state_id)
            await self._rebind_registered_agent(state_id, prepared)

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
    ) -> RouteResult:
        lookup_id = state_id or agent.id
        deadline = None if timeout is None else time.monotonic() + timeout

        async def do_submit() -> str:
            return await self.submit(
                agent,
                user_input,
                session_id=session_id,
                abort_signal=abort_signal,
                persistent=persistent,
                agent_config_id=agent_config_id,
            )

        current_state = await self._resolve_routable_state(lookup_id, deadline)

        if current_state is None:
            return await self._route_with_stream(
                root_state_id=agent.id,
                action="submitted",
                timeout=self._deadline_remaining(deadline) if deadline else timeout,
                include_child_events=include_child_events,
                operation=do_submit,
            )

        if current_state.status in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.WAITING,
            AgentStateStatus.QUEUED,
        ):
            urgent = current_state.status == AgentStateStatus.WAITING
            steered = await self.steer(current_state.id, user_input, urgent=urgent)
            if not steered:
                refreshed = await self._store.get_state(current_state.id)
                if refreshed is not None and refreshed.status in (
                    AgentStateStatus.RUNNING,
                    AgentStateStatus.WAITING,
                    AgentStateStatus.QUEUED,
                ):
                    raise RuntimeError(
                        f"Failed to steer active scheduler state '{current_state.id}'"
                    )
            return RouteResult(action="steered", state_id=current_state.id)

        if (
            current_state.is_root
            and current_state.is_persistent
            and current_state.status in (AgentStateStatus.IDLE, AgentStateStatus.FAILED)
        ):
            return await self._route_with_stream(
                root_state_id=current_state.id,
                action="enqueued",
                timeout=timeout,
                include_child_events=include_child_events,
                operation=lambda: self._enqueue_and_return_state_id(
                    state_id=current_state.id,
                    agent=agent,
                    user_input=user_input,
                ),
            )

        return await self._route_with_stream(
            root_state_id=agent.id,
            action="submitted",
            timeout=timeout,
            include_child_events=include_child_events,
            operation=do_submit,
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
        if state_id is None and agent is None:
            raise RuntimeError("scheduler.stream requires either agent or state_id")

        if state_id is None:
            result = await self._route_with_stream(
                root_state_id=agent.id,
                action="submitted",
                timeout=timeout,
                include_child_events=include_child_events,
                operation=lambda: self.submit(
                    agent,
                    user_input,
                    session_id=session_id,
                    abort_signal=abort_signal,
                    persistent=persistent,
                    agent_config_id=agent_config_id,
                ),
            )
        else:
            result = await self._route_with_stream(
                root_state_id=state_id,
                action="enqueued",
                timeout=timeout,
                include_child_events=include_child_events,
                operation=lambda: self._enqueue_and_return_state_id(
                    state_id=state_id,
                    agent=agent,
                    user_input=user_input,
                ),
            )

        assert result.stream is not None
        async for item in result.stream:
            yield item

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
                    if state.status in (
                        AgentStateStatus.IDLE,
                        AgentStateStatus.COMPLETED,
                    ):
                        return RunOutput(
                            response=state.result_summary,
                            termination_reason=TerminationReason.COMPLETED,
                        )
                    if state.status == AgentStateStatus.FAILED:
                        return RunOutput(
                            error=state.result_summary,
                            termination_reason=TerminationReason.ERROR,
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

    async def steer(
        self,
        state_id: str,
        user_input: UserInput,
        *,
        urgent: bool = False,
    ) -> bool:
        message = UserMessage.from_value(user_input).extract_text()
        if not message:
            return False

        state = await self._store.get_state(state_id)
        if state is None:
            return False

        if state.status == AgentStateStatus.RUNNING:
            handle = self._rt.execution_handles.get(state_id)
            if handle is None:
                return False
            return await handle.steer(message)

        event = PendingEvent(
            id=str(uuid4()),
            target_agent_id=state_id,
            session_id=state.session_id,
            event_type=SchedulerEventType.USER_HINT,
            payload={"hint": message},
            created_at=datetime.now(timezone.utc),
        )
        await self._store.save_event(event)
        self.nudge()
        if urgent and state.status == AgentStateStatus.WAITING:
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

        prepared = await self._prepare_root_agent(agent, state_id)
        await self._rebind_registered_agent(state_id, prepared)
        return True

    async def tick(self) -> None:
        await self._propagate_signals()
        states = await list_all_states(
            self._store,
            statuses=(
                AgentStateStatus.PENDING,
                AgentStateStatus.WAITING,
                AgentStateStatus.QUEUED,
            ),
        )
        events = await self._store.list_events()
        now = datetime.now(timezone.utc)
        actions = self._plan_tick(states, events, now=now)
        for action in actions:
            await self._dispatch_action(action)

    def _plan_tick(
        self,
        states: list[AgentState],
        events: list[PendingEvent],
        *,
        now: datetime,
    ) -> list[DispatchAction]:
        actions: list[DispatchAction] = []
        event_groups = group_events(events)
        debounced_targets = select_debounced_event_targets(
            events,
            min_count=self._config.event_debounce_min_count,
            max_wait_seconds=self._config.event_debounce_max_wait_seconds,
            now=now,
        )

        for state in states:
            if state.status == AgentStateStatus.PENDING:
                actions.append(
                    DispatchAction(state=state, reason=DispatchReason.CHILD_PENDING)
                )
                continue

            if state.is_root and state.status == AgentStateStatus.QUEUED:
                mailbox = tuple(
                    event
                    for event in event_groups.get((state.id, state.session_id), [])
                    if event.event_type == SchedulerEventType.USER_HINT
                )
                actions.append(
                    DispatchAction(
                        state=state,
                        reason=DispatchReason.ROOT_QUEUED_INPUT,
                        input_override=build_mailbox_input(
                            state.pending_input, mailbox
                        ),
                        events=mailbox,
                    )
                )
                continue

            if state.status != AgentStateStatus.WAITING:
                continue

            if state.wake_condition is not None:
                if state.wake_condition.is_timed_out(now):
                    actions.append(
                        DispatchAction(state=state, reason=DispatchReason.WAKE_TIMEOUT)
                    )
                    continue

                if state.wake_condition.is_satisfied(now):
                    actions.append(
                        DispatchAction(state=state, reason=DispatchReason.WAKE_READY)
                    )
                    continue

            key = (state.id, state.session_id)
            if key in debounced_targets:
                grouped_events = tuple(event_groups.get(key, []))
                if grouped_events:
                    actions.append(
                        DispatchAction(
                            state=state,
                            reason=DispatchReason.WAKE_EVENTS,
                            events=grouped_events,
                        )
                    )

        return actions

    async def _dispatch_action(self, action: DispatchAction) -> None:
        state = action.state
        if state.id in self._rt.dispatched:
            return

        if action.reason in (
            DispatchReason.WAKE_READY,
            DispatchReason.WAKE_EVENTS,
            DispatchReason.WAKE_TIMEOUT,
        ):
            rejection = await self._guard.check_wake(state)
            if rejection is not None:
                await self._save_state(state.with_failed(f"Wake rejected: {rejection}"))
                return

        self._rt.dispatched.add(state.id)
        task = asyncio.create_task(self._runner.run(action))
        self._track_active_task(task)

    async def _propagate_signals(self) -> None:
        candidates = await self._store.list_states(
            statuses=(AgentStateStatus.COMPLETED, AgentStateStatus.FAILED),
            signal_propagated=False,
            limit=1000,
        )
        for state in candidates:
            if not state.is_child or state.signal_propagated:
                continue

            parent = await self._store.get_state(state.parent_id or "")
            if parent is not None and parent.wake_condition is not None:
                completed_ids = list(parent.wake_condition.completed_ids)
                if state.id not in completed_ids:
                    completed_ids.append(state.id)
                    await self._save_state(
                        parent.with_updates(
                            wake_condition=parent.wake_condition.with_completed_ids(
                                completed_ids
                            )
                        )
                    )

            await self._save_state(state.with_signal_propagated())

    async def _cancel_subtree(self, state_id: str, reason: str) -> None:
        self._abort_runtime_state(state_id, reason)
        for child in await self._active_children(state_id):
            await self._cancel_subtree(child.id, reason)

        state = await self._store.get_state(state_id)
        if state is None:
            return
        await self._save_state(state.with_failed(reason))
        if state.is_root:
            await finish_stream_channel(self._rt.stream_channels, state.id)

    async def _shutdown_subtree(self, state_id: str) -> None:
        for child in await self._active_children(state_id):
            await self._shutdown_subtree(child.id)

        state = await self._store.get_state(state_id)
        if state is None:
            return

        self._abort_runtime_state(state_id, "Shutdown requested")
        if state.status == AgentStateStatus.RUNNING:
            await self._shutdown_running_state(state)
            return

        await self._shutdown_passive_state(state)

    async def _active_children(self, state_id: str) -> list[AgentState]:
        children = await self._store.list_states(parent_id=state_id, limit=1000)
        return [child for child in children if child.is_active()]

    async def _save_state(self, state: AgentState) -> None:
        await self._store.save_state(state)
        self._notify_state_change(state.id)
        self.nudge()

    async def _resolve_routable_state(
        self,
        state_id: str,
        deadline: float | None,
    ) -> AgentState | None:
        """Return a non-PENDING state, or None if the state vanishes."""
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

    def _notify_state_change(self, state_id: str) -> None:
        for waiter in self._rt.waiters.get(state_id, set()):
            waiter.set()

    def _track_active_task(self, task: asyncio.Task) -> None:
        self._rt.active_tasks.add(task)
        task.add_done_callback(self._rt.active_tasks.discard)

    def _build_stream(
        self,
        state_id: str,
        *,
        timeout: float | None,
        include_child_events: bool,
    ) -> AsyncIterator[AgentStreamItem]:
        async def iterator() -> AsyncIterator[AgentStreamItem]:
            open_stream_channel(
                self._rt.stream_channels,
                state_id,
                include_child_events=include_child_events,
            )
            try:
                async for item in consume_stream_channel(
                    self._rt.stream_channels,
                    state_id,
                    timeout=timeout,
                ):
                    yield item
            finally:
                await self._raise_stream_failure_if_needed(state_id)
                close_stream_channel(self._rt.stream_channels, state_id)

        return iterator()

    async def _route_with_stream(
        self,
        *,
        root_state_id: str,
        action: str,
        timeout: float | None,
        include_child_events: bool,
        operation: Callable[[], Awaitable[str]],
    ) -> RouteResult:
        if root_state_id in self._rt.stream_channels:
            raise RuntimeError(
                f"stream subscriber already active for root '{root_state_id}'"
            )
        state_id = await operation()
        return RouteResult(
            action=action,
            state_id=state_id,
            stream=self._build_stream(
                root_state_id,
                timeout=timeout,
                include_child_events=include_child_events,
            ),
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

    async def _prepare_root_agent(self, agent: Agent, state_id: str) -> Agent:
        """Prepare a root-level agent by injecting scheduling tools."""
        return await agent.create_child_agent(
            child_id=state_id,
            system_prompt_override=agent.config.system_prompt,
            exclude_tool_names={tool.get_name() for tool in agent.tools},
            extra_tools=list(self._scheduling_tools),
        )

    async def _rebind_registered_agent(self, state_id: str, agent: Agent) -> None:
        """Register prepared agent, closing any previous agent for the same state."""
        previous = self._rt.agents.get(state_id)
        self._rt.agents[state_id] = agent
        if previous is not None and previous is not agent:
            try:
                await previous.close()
            except Exception:  # noqa: BLE001
                logger.exception("scheduler_rebind_close_failed", state_id=state_id)

    def _abort_runtime_state(self, state_id: str, reason: str) -> None:
        signal = self._rt.abort_signals.get(state_id)
        if signal is None:
            signal = AbortSignal()
            self._rt.abort_signals[state_id] = signal
        if not signal.is_aborted():
            signal.abort(reason)

        handle = self._rt.execution_handles.get(state_id)
        if handle is not None:
            handle.cancel(reason)

    async def _shutdown_running_state(self, state: AgentState) -> None:
        if state.is_root and state.is_persistent:
            self._rt.shutdown_requested.add(state.id)
            await self._save_state(
                state.with_queued(pending_input=SHUTDOWN_SUMMARY_TASK)
            )
            return
        await self._save_state(state.with_failed("Shutdown before completion"))

    async def _shutdown_passive_state(self, state: AgentState) -> None:
        if state.is_root and state.status in (
            AgentStateStatus.WAITING,
            AgentStateStatus.IDLE,
        ):
            await self._save_state(
                state.with_queued(pending_input=SHUTDOWN_SUMMARY_TASK)
            )
            return

        if state.status in (
            AgentStateStatus.WAITING,
            AgentStateStatus.PENDING,
            AgentStateStatus.QUEUED,
        ):
            await self._save_state(state.with_failed("Shutdown before completion"))
            if state.is_root:
                await finish_stream_channel(self._rt.stream_channels, state.id)

    def _deadline_remaining(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0
        return remaining

    async def _raise_stream_failure_if_needed(self, state_id: str) -> None:
        state = await self._store.get_state(state_id)
        if state is not None and state.status == AgentStateStatus.FAILED:
            raise RuntimeError(state.result_summary or "scheduler stream failed")


__all__ = ["Scheduler"]
