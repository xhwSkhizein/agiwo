"""SchedulerEngine — the sole owner of scheduler orchestration semantics."""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator
from uuid import uuid4

from agiwo.agent.agent import Agent
from agiwo.agent.input import UserInput
from agiwo.agent.input_codec import extract_text
from agiwo.agent.runtime import RunOutput, TerminationReason
from agiwo.scheduler.control import SchedulerControl
from agiwo.scheduler.coordinator import SchedulerCoordinator
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    ChildAgentConfigOverrides,
    PendingEvent,
    SchedulerConfig,
    SchedulerEventType,
    SchedulerOutput,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
    to_seconds,
)
from agiwo.scheduler.runner import SchedulerRunner
from agiwo.scheduler.store import AgentStateStorage
from agiwo.tool.base import AgentProcessProbe
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SchedulerEngine(SchedulerControl):
    """Own scheduler state-machine semantics and tool-facing control operations."""

    def __init__(
        self,
        *,
        config: SchedulerConfig,
        store: AgentStateStorage,
        guard: TaskGuard,
        coordinator: SchedulerCoordinator,
        runner: SchedulerRunner,
    ) -> None:
        self._config = config
        self._store = store
        self._guard = guard
        self._coordinator = coordinator
        self._runner = runner
        self._scheduling_tools: list = []

    @property
    def runner(self) -> SchedulerRunner:
        return self._runner

    def get_registered_agent(self, state_id: str) -> Agent | None:
        return self._coordinator.get_registered_agent(state_id)

    def set_scheduling_tools(self, tools: list) -> None:
        """Register the scheduler-managed tool set for agent preparation."""
        self._scheduling_tools = list(tools)

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

        self.prepare_agent(agent)

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
            self._coordinator.set_abort_signal(state.id, abort_signal)

        task = asyncio.create_task(
            self._runner.run_root_agent(agent, user_input, resolved_session_id, state)
        )
        self._coordinator.track_active_task(task)
        return state.id

    async def submit_task(
        self,
        state_id: str,
        task: UserInput,
        *,
        agent: Agent | None = None,
    ) -> None:
        state = await self._store.get_state(state_id)
        if state is None:
            raise RuntimeError(f"Agent state '{state_id}' not found")
        if not state.is_persistent:
            raise RuntimeError(f"Agent '{state_id}' is not persistent. Use submit() instead.")

        if state.status in (AgentStateStatus.COMPLETED, AgentStateStatus.FAILED):
            await self._store.update_status(state_id, AgentStateStatus.SLEEPING)
            logger.info(
                "reset_terminal_to_sleeping",
                state_id=state_id,
                from_status=state.status.value,
            )
        elif state.status != AgentStateStatus.SLEEPING:
            raise RuntimeError(
                f"Agent '{state_id}' is {state.status.value}. "
                f"Cannot submit task (expected SLEEPING, COMPLETED, or FAILED)."
            )

        if agent is not None:
            self.prepare_agent(agent)

        wake_condition = WakeCondition(
            type=WakeType.TASK_SUBMITTED,
            submitted_task=task,
        )
        await self._update_status_and_notify(
            state_id,
            AgentStateStatus.SLEEPING,
            wake_condition=wake_condition,
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
        self._coordinator.open_output_channel(
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
            async for output in self._coordinator.consume_output_channel(agent.id, timeout):
                yield output
        finally:
            self._coordinator.close_output_channel(agent.id)

    async def submit_task_and_subscribe(
        self,
        state_id: str,
        task: UserInput,
        *,
        agent: Agent | None = None,
        timeout: float | None = None,
        include_child_outputs: bool = True,
    ) -> AsyncIterator[SchedulerOutput]:
        self._coordinator.open_output_channel(
            state_id,
            include_child_outputs=include_child_outputs,
        )

        try:
            await self.submit_task(state_id, task, agent=agent)
            async for output in self._coordinator.consume_output_channel(state_id, timeout):
                yield output
        finally:
            self._coordinator.close_output_channel(state_id)

    async def wait_for(
        self,
        state_id: str,
        timeout: float | None = None,
    ) -> RunOutput:
        start = time.time()
        initial_state = await self._store.get_state(state_id)
        is_persistent = initial_state.is_persistent if initial_state else False

        event = self._coordinator.get_or_create_state_event(state_id)

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
                        wake_condition = state.wake_condition
                        has_pending_task = (
                            wake_condition is not None
                            and wake_condition.type == WakeType.TASK_SUBMITTED
                            and wake_condition.submitted_task is not None
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
            self._coordinator.pop_state_event(state_id)

    async def get_state(self, state_id: str) -> AgentState | None:
        return await self._store.get_state(state_id)

    async def cancel(self, state_id: str, reason: str = "Cancelled by user") -> bool:
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

    async def steer(
        self,
        state_id: str,
        user_input: UserInput,
        *,
        urgent: bool = False,
    ) -> bool:
        message = extract_text(user_input)
        if not message:
            return False

        state = await self._store.get_state(state_id)
        if state is None:
            return False

        if state.status == AgentStateStatus.RUNNING:
            agent = self._coordinator.get_registered_agent(state_id)
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
            await self._try_urgent_wake(state)

        logger.info("steer_persisted", state_id=state_id, urgent=urgent)
        return True

    async def shutdown(self, state_id: str) -> bool:
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

    def prepare_agent(self, agent: Agent, scheduling_tools: list | None = None) -> None:
        """Inject scheduling tools into agent and register it in coordinator."""
        tools_to_inject = scheduling_tools if scheduling_tools is not None else self._scheduling_tools
        existing_names = {tool.get_name() for tool in agent.tools}
        for tool in tools_to_inject:
            if tool.get_name() not in existing_names:
                agent.tools.append(tool)
        if agent.options is not None:
            agent.options.enable_termination_summary = True
        self._coordinator.register_agent(agent)

    async def tick(self) -> None:
        await self._propagate_signals()
        await self._enforce_timeouts()
        await self._check_health()
        await self._process_pending_events()
        await self._start_pending()
        await self._wake_sleeping()

    async def _update_status_and_notify(
        self,
        state_id: str,
        status: AgentStateStatus,
        *,
        result_summary: str | None = ...,
        wake_condition: WakeCondition | None = ...,
    ) -> None:
        await self._store.update_status(
            state_id,
            status,
            result_summary=result_summary,
            wake_condition=wake_condition,
        )
        if status in (AgentStateStatus.COMPLETED, AgentStateStatus.FAILED):
            self._coordinator.notify_state_change(state_id)
        elif status == AgentStateStatus.SLEEPING and wake_condition is not ...:
            if wake_condition is not None and wake_condition.type == WakeType.TASK_SUBMITTED:
                self._coordinator.notify_state_change(state_id)

    async def _recursive_cancel(self, state_id: str, reason: str) -> None:
        signal = self._coordinator.get_abort_signal(state_id)
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

        await self._update_status_and_notify(
            state_id,
            AgentStateStatus.FAILED,
            result_summary=reason,
        )

    async def _recursive_shutdown(self, state_id: str) -> None:
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
            wake_condition = WakeCondition(
                type=WakeType.TASK_SUBMITTED,
                submitted_task=(
                    "System shutdown requested. Please produce a final summary "
                    "report of all work done so far."
                ),
            )
            await self._update_status_and_notify(
                state_id,
                AgentStateStatus.SLEEPING,
                wake_condition=wake_condition,
            )
        elif state.status == AgentStateStatus.PENDING:
            await self._update_status_and_notify(
                state_id,
                AgentStateStatus.FAILED,
                result_summary="Shutdown before execution",
            )

    async def _propagate_signals(self) -> None:
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
        now = datetime.now(timezone.utc)
        timed_out = await self._guard.find_timed_out(now)
        for state in timed_out:
            self._coordinator.dispatch_state_task(
                state,
                lambda: self._runner.wake_for_timeout(state),
            )

    async def _start_pending(self) -> None:
        pending = await self._store.find_pending()
        for state in pending:
            self._coordinator.dispatch_state_task(
                state,
                lambda: self._runner.run_pending_agent(state),
            )

    async def _check_health(self) -> None:
        now = datetime.now(timezone.utc)
        unhealthy = await self._guard.find_unhealthy(now)
        for state in unhealthy:
            if state.parent_id is None:
                continue
            parent_state = await self._store.get_state(state.parent_id)
            if parent_state is None:
                continue

            already_warned = await self._store.has_recent_health_warning(
                target_agent_id=state.parent_id,
                source_agent_id=state.id,
                within_seconds=self._config.task_limits.health_check_threshold_seconds,
                now=now,
            )
            if already_warned:
                continue

            event = PendingEvent(
                id=str(uuid4()),
                target_agent_id=state.parent_id,
                session_id=state.session_id,
                event_type=SchedulerEventType.HEALTH_WARNING,
                payload={
                    "child_agent_id": state.id,
                    "message": (
                        f"Agent '{state.id}' appears stuck — no activity for "
                        f">{self._config.task_limits.health_check_threshold_seconds:.0f}s. "
                        "Consider using cancel_agent to terminate it."
                    ),
                    "last_activity_at": (
                        state.last_activity_at.isoformat()
                        if state.last_activity_at
                        else None
                    ),
                },
                source_agent_id=state.id,
                created_at=now,
            )
            await self._store.save_event(event)
            logger.warning(
                "health_warning_emitted",
                stuck_agent_id=state.id,
                parent_id=state.parent_id,
            )

    async def _process_pending_events(self) -> None:
        now = datetime.now(timezone.utc)
        min_count = self._config.event_debounce_min_count
        max_wait = self._config.event_debounce_max_wait_seconds

        agent_session_pairs = await self._store.find_agents_with_debounced_events(
            min_count,
            max_wait,
            now,
        )
        for agent_id, session_id in agent_session_pairs:
            state = await self._store.get_state(agent_id)
            if state is None:
                await self._store.delete_events_by_agent(agent_id)
                continue

            if state.status != AgentStateStatus.SLEEPING:
                events = await self._store.get_pending_events(agent_id, session_id)
                if events:
                    await self._store.delete_events([event.id for event in events])
                continue

            rejection = await self._guard.check_wake(state)
            if rejection is not None:
                logger.warning(
                    "pending_events_wake_rejected",
                    state_id=agent_id,
                    reason=rejection,
                )
                continue

            if not self._coordinator.reserve_state_dispatch(state.id):
                continue

            events = await self._store.get_pending_events(agent_id, state.session_id)
            if not events:
                self._coordinator.release_state_dispatch(state.id)
                continue

            await self._store.delete_events([event.id for event in events])
            self._coordinator.track_active_task(
                asyncio.create_task(self._runner.wake_agent_for_events(state, events))
            )

    async def _wake_sleeping(self) -> None:
        now = datetime.now(timezone.utc)
        wakeable = await self._store.find_wakeable(now)
        for state in wakeable:
            rejection = await self._guard.check_wake(state)
            if rejection is not None:
                logger.warning(
                    "wake_rejected",
                    state_id=state.id,
                    reason=rejection,
                )
                await self._update_status_and_notify(
                    state.id,
                    AgentStateStatus.FAILED,
                    result_summary=f"Wake rejected: {rejection}",
                )
                continue

            self._coordinator.dispatch_state_task(
                state,
                lambda: self._runner.wake_agent(state),
            )

    async def _try_urgent_wake(self, state: AgentState) -> None:
        rejection = await self._guard.check_wake(state)
        if rejection is not None:
            logger.warning("urgent_wake_rejected", state_id=state.id, reason=rejection)
            return

        if not self._coordinator.reserve_state_dispatch(state.id):
            return

        events = await self._store.get_pending_events(state.id, state.session_id)
        if not events:
            self._coordinator.release_state_dispatch(state.id)
            return

        await self._store.delete_events([event.id for event in events])
        self._coordinator.track_active_task(
            asyncio.create_task(self._runner.wake_agent_for_events(state, events))
        )

    async def spawn_child(
        self,
        *,
        parent_agent_id: str,
        session_id: str,
        task: str,
        instruction: str | None,
        system_prompt: str | None,
        custom_child_id: str | None,
    ) -> AgentState:
        parent_state = await self._store.get_state(parent_agent_id)
        if parent_state is None:
            raise ValueError(f"Parent agent state '{parent_agent_id}' not found")

        rejection = await self._guard.check_spawn(parent_state)
        if rejection is not None:
            raise ValueError(f"Spawn rejected: {rejection}")

        child_id = custom_child_id or f"{parent_agent_id}_{uuid4().hex[:5]}"
        config_overrides = ChildAgentConfigOverrides(
            instruction=instruction,
            system_prompt=system_prompt,
        ).to_dict()
        state = AgentState(
            id=child_id,
            session_id=session_id,
            status=AgentStateStatus.PENDING,
            task=task,
            parent_id=parent_agent_id,
            config_overrides=config_overrides,
            depth=parent_state.depth + 1,
        )
        await self._store.save_state(state)
        return state

    async def sleep_current_agent(
        self,
        *,
        agent_id: str,
        session_id: str,
        wake_type: WakeType,
        wake_type_str: str,
        wait_mode_str: str,
        explicit_wait_for: list[str] | None,
        timeout: float | None,
        delay_seconds: float | int | None,
        time_unit_str: str,
        explain: str | None,
    ) -> tuple[WakeCondition, str]:
        wake_condition = await self._build_sleep_condition(
            agent_id=agent_id,
            session_id=session_id,
            wake_type=wake_type,
            wait_mode_str=wait_mode_str,
            explicit_wait_for=explicit_wait_for,
            timeout=timeout,
            delay_seconds=delay_seconds,
            time_unit_str=time_unit_str,
        )
        await self._store.update_status(
            agent_id,
            AgentStateStatus.SLEEPING,
            wake_condition=wake_condition,
            explain=explain,
        )
        summary = self._build_sleep_summary(
            agent_id=agent_id,
            wake_type=wake_type,
            wake_type_str=wake_type_str,
            wake_condition=wake_condition,
            delay_seconds=delay_seconds,
            time_unit_str=time_unit_str,
            explain=explain,
        )
        return wake_condition, summary

    async def get_child_state(self, target_id: str) -> AgentState | None:
        return await self._store.get_state(target_id)

    async def list_child_states(
        self,
        *,
        caller_id: str | None,
        session_id: str,
    ) -> list[AgentState]:
        children = await self._store.get_states_by_parent(caller_id)
        return [child for child in children if child.session_id == session_id]

    async def inspect_child_processes(
        self,
        target_id: str,
    ) -> list[dict[str, object]]:
        agent = self._coordinator.get_registered_agent(target_id)
        if agent is None:
            return []

        for tool in agent.tools:
            if not isinstance(tool, AgentProcessProbe):
                continue
            try:
                return await tool.list_agent_processes(target_id, state="running")
            except Exception:  # noqa: BLE001 - tool capability boundary
                return []
        return []

    async def cancel_child(
        self,
        *,
        caller_id: str | None,
        target_id: str,
        force: bool,
        reason: str,
    ) -> tuple[str, AgentState | None, list[dict[str, object]]]:
        target_state = await self._store.get_state(target_id)
        if target_state is None:
            return "missing", None, []

        if target_state.parent_id != caller_id:
            raise PermissionError(
                f"agent '{target_id}' is not a direct child of '{caller_id}'"
            )

        if target_state.status not in (
            AgentStateStatus.RUNNING,
            AgentStateStatus.SLEEPING,
            AgentStateStatus.PENDING,
        ):
            return "already_terminal", target_state, []

        if not force and target_state.status == AgentStateStatus.RUNNING:
            return (
                "requires_force",
                target_state,
                await self.inspect_child_processes(target_id),
            )

        await self._recursive_cancel(target_id, reason)
        return "cancelled", target_state, []

    def age_seconds(self, timestamp: datetime, *, now: datetime | None = None) -> int:
        current = now or datetime.now(timezone.utc)
        normalized = timestamp if timestamp.tzinfo else timestamp.replace(
            tzinfo=timezone.utc
        )
        return int((current - normalized).total_seconds())

    async def _build_sleep_condition(
        self,
        *,
        agent_id: str,
        session_id: str,
        wake_type: WakeType,
        wait_mode_str: str,
        explicit_wait_for: list[str] | None,
        timeout: float | None,
        delay_seconds: float | int | None,
        time_unit_str: str,
    ) -> WakeCondition:
        now = datetime.now(timezone.utc)
        wake_condition = WakeCondition(type=wake_type)
        if wake_type == WakeType.WAITSET:
            wake_condition.wait_for = await self._resolve_waitset_targets(
                agent_id=agent_id,
                session_id=session_id,
                explicit_wait_for=explicit_wait_for,
            )
            wake_condition.wait_mode = self._resolve_wait_mode(wait_mode_str)
            wake_condition.completed_ids = await self._collect_completed_child_ids(
                wake_condition.wait_for
            )
            effective_timeout = timeout or self._guard.limits.default_wait_timeout
            wake_condition.timeout_at = now + timedelta(seconds=effective_timeout)
            return wake_condition

        if delay_seconds is None:
            raise ValueError("delay_seconds is required for timer/periodic wake type")

        time_unit = self._resolve_time_unit(time_unit_str)
        wake_condition.time_value = delay_seconds
        wake_condition.time_unit = time_unit
        wake_condition.wakeup_at = now + timedelta(
            seconds=to_seconds(delay_seconds, time_unit)
        )
        if wake_type == WakeType.PERIODIC and timeout is not None:
            wake_condition.timeout_at = now + timedelta(seconds=timeout)
        return wake_condition

    def _build_sleep_summary(
        self,
        *,
        agent_id: str,
        wake_type: WakeType,
        wake_type_str: str,
        wake_condition: WakeCondition,
        delay_seconds: float | int | None,
        time_unit_str: str,
        explain: str | None,
    ) -> str:
        summary = f"Agent '{agent_id}' entering sleep. Wake condition: {wake_type_str}"
        if wake_type == WakeType.WAITSET:
            summary += (
                " "
                f"(waiting_for={len(wake_condition.wait_for)}, "
                f"mode={wake_condition.wait_mode.value}, "
                f"already_done={len(wake_condition.completed_ids)})"
            )
        elif delay_seconds is not None:
            summary += f" (delay={delay_seconds} {time_unit_str})"
        if explain:
            summary += f" | reason: {explain}"
        return summary

    async def _resolve_waitset_targets(
        self,
        *,
        agent_id: str,
        session_id: str,
        explicit_wait_for: list[str] | None,
    ) -> list[str]:
        if explicit_wait_for is not None:
            return explicit_wait_for

        children = await self._store.get_states_by_parent(agent_id)
        return [child.id for child in children if child.session_id == session_id]

    def _resolve_wait_mode(self, wait_mode_str: str) -> WaitMode:
        try:
            return WaitMode(wait_mode_str)
        except ValueError:
            return WaitMode.ALL

    async def _collect_completed_child_ids(
        self,
        child_ids: list[str],
    ) -> list[str]:
        completed_ids: list[str] = []
        for child_id in child_ids:
            child_state = await self._store.get_state(child_id)
            if child_state is not None and child_state.status in (
                AgentStateStatus.COMPLETED,
                AgentStateStatus.FAILED,
            ):
                completed_ids.append(child_id)
        return completed_ids

    def _resolve_time_unit(self, time_unit_str: str) -> TimeUnit:
        try:
            return TimeUnit(time_unit_str)
        except ValueError:
            return TimeUnit.SECONDS


__all__ = ["SchedulerEngine"]
