"""SchedulerEngine — the sole owner of scheduler orchestration semantics."""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator
from uuid import uuid4

from agiwo.agent.input import UserInput
from agiwo.agent.input_codec import extract_text
from agiwo.agent.runtime import AgentStreamItem, RunOutput, TerminationReason
from agiwo.agent.scheduler_port import SchedulerAgentPort
from agiwo.scheduler.control import (
    CancelChildRequest,
    CancelChildResult,
    SchedulerControl,
    SleepRequest,
    SleepResult,
    SpawnChildRequest,
)
from agiwo.scheduler.coordinator import SchedulerCoordinator
from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import (
    ACTIVE_AGENT_STATUSES,
    AgentState,
    AgentStateStatus,
    ChildAgentConfigOverrides,
    PendingEvent,
    SchedulerConfig,
    SchedulerEventType,
    WakeCondition,
    WakeType,
    to_seconds,
)
from agiwo.scheduler.runner import SchedulerRunner
from agiwo.scheduler.state_ops import SchedulerStateOps
from agiwo.scheduler.store import AgentStateStorage
from agiwo.scheduler.store.codec import serialize_child_agent_config_overrides
from agiwo.scheduler.tick_ops import SchedulerTickOps
from agiwo.scheduler.tree_ops import SchedulerTreeOps
from agiwo.tool.process import AgentProcessRegistry
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SchedulerEngine(SchedulerControl):
    """Own scheduler state-machine semantics and tool-facing control operations."""

    def __init__(
        self,
        *,
        store: AgentStateStorage,
        coordinator: SchedulerCoordinator,
        runner: SchedulerRunner,
        config: SchedulerConfig | None = None,
        guard: TaskGuard | None = None,
        state_ops: SchedulerStateOps | None = None,
        tree_ops: SchedulerTreeOps | None = None,
        tick_ops: SchedulerTickOps | None = None,
    ) -> None:
        self._store = store
        self._coordinator = coordinator
        self._runner = runner
        resolved_config = config or SchedulerConfig()
        resolved_state_ops = state_ops or SchedulerStateOps(
            store=store,
            coordinator=coordinator,
        )
        self._guard = guard or TaskGuard(resolved_config.task_limits, store)
        resolved_tree_ops = tree_ops or SchedulerTreeOps(
            store=store,
            coordinator=coordinator,
            state_ops=resolved_state_ops,
        )
        self._state_ops = resolved_state_ops
        self._tree_ops = resolved_tree_ops
        self._tick_ops = tick_ops or SchedulerTickOps(
            config=resolved_config,
            store=store,
            guard=self._guard,
            coordinator=coordinator,
            runner=runner,
            state_ops=resolved_state_ops,
        )
        self._scheduling_tools: list = []

    @property
    def runner(self) -> SchedulerRunner:
        return self._runner

    def get_registered_agent(self, state_id: str):
        agent = self._coordinator.get_registered_agent(state_id)
        if agent is None:
            return None
        return agent.unwrap_agent()

    def set_scheduling_tools(self, tools: list) -> None:
        """Register the scheduler-managed tool set for agent preparation."""
        self._scheduling_tools = list(tools)

    async def run(
        self,
        agent: SchedulerAgentPort,
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
        agent: SchedulerAgentPort,
        user_input: UserInput,
        *,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
        persistent: bool = False,
        agent_config_id: str | None = None,
    ) -> str:
        existing = await self._store.get_state(agent.id)
        if existing is not None and existing.status in ACTIVE_AGENT_STATUSES:
            raise RuntimeError(
                f"Agent '{agent.id}' is already active (status={existing.status.value}). "
                f"Cannot submit concurrently. Use a different agent_id or enqueue_input()."
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
            self._runner.run_root_agent(
                agent,
                user_input,
                resolved_session_id,
                state,
            )
        )
        self._coordinator.track_active_task(task)
        return state.id

    async def enqueue_input(
        self,
        state_id: str,
        user_input: UserInput,
        *,
        agent: SchedulerAgentPort | None = None,
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
            self.prepare_agent(agent)

        await self._state_ops.mark_queued(
            state,
            pending_input=user_input,
        )
        logger.info("input_enqueued_to_persistent_agent", state_id=state_id)

    async def stream(
        self,
        user_input: UserInput,
        *,
        agent: SchedulerAgentPort | None = None,
        state_id: str | None = None,
        session_id: str | None = None,
        abort_signal: AbortSignal | None = None,
        persistent: bool = False,
        agent_config_id: str | None = None,
        timeout: float | None = None,
        include_child_events: bool = True,
    ) -> AsyncIterator[AgentStreamItem]:
        root_state_id = state_id or (agent.id if agent is not None else None)
        if root_state_id is None:
            raise RuntimeError("scheduler.stream requires either agent or state_id")

        self._coordinator.open_stream_channel(
            root_state_id,
            include_child_events=include_child_events,
        )
        saw_root_terminal = False
        try:
            if state_id is None:
                if agent is None:
                    raise RuntimeError(
                        "scheduler.stream requires an agent when state_id is omitted"
                    )
                await self.submit(
                    agent,
                    user_input,
                    session_id=session_id,
                    abort_signal=abort_signal,
                    persistent=persistent,
                    agent_config_id=agent_config_id,
                )
            else:
                await self.enqueue_input(state_id, user_input, agent=agent)

            async for item in self._coordinator.consume_stream_channel(
                root_state_id,
                timeout,
            ):
                if item.depth == 0 and item.type in {"run_completed", "run_failed"}:
                    saw_root_terminal = True
                yield item

            if not saw_root_terminal:
                state = await self._store.get_state(root_state_id)
                if state is not None and state.status == AgentStateStatus.FAILED:
                    raise RuntimeError(
                        state.result_summary or "scheduler stream failed"
                    )
        finally:
            self._coordinator.close_stream_channel(root_state_id)

    async def wait_for(
        self,
        state_id: str,
        timeout: float | None = None,
    ) -> RunOutput:
        start = time.time()
        event = self._coordinator.get_or_create_state_event(state_id)

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
        if not state.is_active():
            return False

        await self._tree_ops.cancel_subtree(state_id, reason)
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
            handle = self._coordinator.get_execution_handle(state_id)
            if handle is None:
                return False
            if not await handle.steer(message):
                return False
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

        if urgent and state.status == AgentStateStatus.WAITING:
            await self._tick_ops.try_urgent_wake(state)

        logger.info("steer_persisted", state_id=state_id, urgent=urgent)
        return True

    async def shutdown(self, state_id: str) -> bool:
        state = await self._store.get_state(state_id)
        if state is None:
            return False
        if not state.is_active():
            return False

        await self._tree_ops.shutdown_subtree(state_id)
        logger.info("scheduler_shutdown", state_id=state_id)
        return True

    def prepare_agent(
        self,
        agent: SchedulerAgentPort,
        scheduling_tools: list | None = None,
    ) -> None:
        """Inject scheduling tools into agent and register it in coordinator."""
        tools_to_inject = (
            scheduling_tools if scheduling_tools is not None else self._scheduling_tools
        )
        agent.install_runtime_tools(list(tools_to_inject))
        agent.set_termination_summary_enabled(True)
        self._coordinator.register_agent(agent)

    async def tick(self) -> None:
        await self._tick_ops.propagate_signals()
        await self._tick_ops.enforce_timeouts()
        await self._tick_ops.process_pending_events()
        await self._tick_ops.start_pending()
        await self._tick_ops.start_queued_roots()
        await self._tick_ops.wake_waiting()

    async def spawn_child(self, request: SpawnChildRequest) -> AgentState:
        parent_state = await self._store.get_state(request.parent_agent_id)
        if parent_state is None:
            raise ValueError(
                f"Parent agent state '{request.parent_agent_id}' not found"
            )

        rejection = await self._guard.check_spawn(parent_state)
        if rejection is not None:
            raise ValueError(f"Spawn rejected: {rejection}")

        child_id = (
            request.custom_child_id or f"{request.parent_agent_id}_{uuid4().hex[:5]}"
        )
        state = AgentState(
            id=child_id,
            session_id=request.session_id,
            status=AgentStateStatus.PENDING,
            task=request.task,
            parent_id=request.parent_agent_id,
            config_overrides=serialize_child_agent_config_overrides(
                ChildAgentConfigOverrides(
                    instruction=request.instruction,
                    system_prompt=request.system_prompt,
                )
            ),
            depth=parent_state.depth + 1,
        )
        await self._store.save_state(state)
        return state

    async def sleep_current_agent(self, request: SleepRequest) -> SleepResult:
        wake_condition = await self._build_sleep_condition(request)
        state = await self._store.get_state(request.agent_id)
        if state is None:
            raise ValueError(f"Agent state '{request.agent_id}' not found")
        await self._state_ops.mark_waiting(
            state,
            wake_condition=wake_condition,
            explain=request.explain,
        )
        return SleepResult(
            wake_condition=wake_condition,
            summary=self._build_sleep_summary(request, wake_condition),
        )

    async def get_child_state(self, target_id: str) -> AgentState | None:
        return await self._store.get_state(target_id)

    async def list_child_states(
        self,
        *,
        caller_id: str | None,
        session_id: str,
    ) -> list[AgentState]:
        return await self._store.list_states(
            parent_id=caller_id,
            session_id=session_id,
            limit=1000,
        )

    async def inspect_child_processes(
        self,
        target_id: str,
    ) -> list[dict[str, object]]:
        agent = self._coordinator.get_registered_agent(target_id)
        if agent is None:
            return []

        for tool in agent.tools:
            if not isinstance(tool, AgentProcessRegistry):
                continue
            try:
                return await tool.list_agent_processes(target_id, state="running")
            except Exception:  # noqa: BLE001 - tool capability boundary
                return []
        return []

    async def cancel_child(self, request: CancelChildRequest) -> CancelChildResult:
        target_state = await self._store.get_state(request.target_id)
        if target_state is None:
            return CancelChildResult(outcome="missing")

        if target_state.parent_id != request.caller_id:
            raise PermissionError(
                f"agent '{request.target_id}' is not a direct child of '{request.caller_id}'"
            )

        if not target_state.is_active():
            return CancelChildResult(
                outcome="already_terminal",
                state=target_state,
            )

        if not request.force and target_state.status == AgentStateStatus.RUNNING:
            return CancelChildResult(
                outcome="requires_force",
                state=target_state,
                running_processes=await self.inspect_child_processes(request.target_id),
            )

        await self._tree_ops.cancel_subtree(request.target_id, request.reason)
        return CancelChildResult(
            outcome="cancelled",
            state=target_state,
        )

    def age_seconds(self, timestamp: datetime, *, now: datetime | None = None) -> int:
        current = now or datetime.now(timezone.utc)
        normalized = (
            timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        )
        return int((current - normalized).total_seconds())

    async def _build_sleep_condition(self, request: SleepRequest) -> WakeCondition:
        now = datetime.now(timezone.utc)
        if request.wake_type == WakeType.WAITSET:
            wait_for = await self._resolve_waitset_targets(request)
            return WakeCondition(
                type=request.wake_type,
                wait_for=wait_for,
                wait_mode=request.wait_mode,
                completed_ids=await self._collect_completed_child_ids(wait_for),
                timeout_at=now
                + timedelta(
                    seconds=request.timeout or self._guard.limits.default_wait_timeout
                ),
            )

        if request.delay_seconds is None:
            raise ValueError("delay_seconds is required for timer/periodic wake type")

        wake_condition = WakeCondition(
            type=request.wake_type,
            time_value=request.delay_seconds,
            time_unit=request.time_unit,
            wakeup_at=now
            + timedelta(seconds=to_seconds(request.delay_seconds, request.time_unit)),
        )
        if request.wake_type == WakeType.PERIODIC and request.timeout is not None:
            wake_condition.timeout_at = now + timedelta(seconds=request.timeout)
        return wake_condition

    def _build_sleep_summary(
        self,
        request: SleepRequest,
        wake_condition: WakeCondition,
    ) -> str:
        summary = (
            f"Agent '{request.agent_id}' entering sleep. "
            f"Wake condition: {request.wake_type.value}"
        )
        if request.wake_type == WakeType.WAITSET:
            summary += (
                " "
                f"(waiting_for={len(wake_condition.wait_for)}, "
                f"mode={wake_condition.wait_mode.value}, "
                f"already_done={len(wake_condition.completed_ids)})"
            )
        elif request.delay_seconds is not None:
            summary += f" (delay={request.delay_seconds} {request.time_unit.value})"
        if request.explain:
            summary += f" | reason: {request.explain}"
        return summary

    async def _resolve_waitset_targets(self, request: SleepRequest) -> list[str]:
        if request.wait_for is not None:
            return request.wait_for
        children = await self._store.list_states(
            parent_id=request.agent_id,
            session_id=request.session_id,
            limit=1000,
        )
        return [child.id for child in children]

    async def _collect_completed_child_ids(self, child_ids: list[str]) -> list[str]:
        completed_ids: list[str] = []
        for child_id in child_ids:
            child_state = await self._store.get_state(child_id)
            if child_state is not None and child_state.is_terminal():
                completed_ids.append(child_id)
        return completed_ids


__all__ = ["SchedulerEngine"]
