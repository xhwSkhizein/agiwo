"""
SchedulerRunner — execute a single scheduler-managed agent cycle.

This module owns agent execution, wake-message construction, and translating
RunOutput into persisted scheduler outcomes. It does not decide which states
should be dispatched; that remains the engine's job.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import uuid4

from agiwo.agent.execution import AgentExecutionHandlePort
from agiwo.agent.input import UserInput
from agiwo.agent.scheduler_port import ChildAgentOverrides, SchedulerAgentPort
from agiwo.agent.runtime import AgentStreamItem, RunOutput, TerminationReason
from agiwo.agent.streaming import consume_execution_stream
from agiwo.scheduler.coordinator import SchedulerCoordinator
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.state_ops import SchedulerStateOps
from agiwo.scheduler.store.codec import deserialize_child_agent_config_overrides
from agiwo.scheduler.store import AgentStateStorage
from agiwo.scheduler.wake_messages import WakeMessageBuilder
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


AgentRunMode = Literal[
    "root",
    "queued_root",
    "child_pending",
    "wake",
    "wake_events",
    "wake_timeout",
]


@dataclass(frozen=True)
class AgentRunSpec:
    transition_to_running: bool = False
    clear_wake_condition: bool = False
    increment_wake_count: bool = False
    create_abort_signal: bool = False
    enforce_parent_abort: bool = False
    emit_child_failed_event: bool = False


RUN_MODE_TO_SPEC: dict[AgentRunMode, AgentRunSpec] = {
    "root": AgentRunSpec(),
    "queued_root": AgentRunSpec(
        transition_to_running=True,
    ),
    "child_pending": AgentRunSpec(
        transition_to_running=True,
        create_abort_signal=True,
        enforce_parent_abort=True,
        emit_child_failed_event=True,
    ),
    "wake": AgentRunSpec(
        transition_to_running=True,
        increment_wake_count=True,
    ),
    "wake_events": AgentRunSpec(
        transition_to_running=True,
        increment_wake_count=True,
    ),
    "wake_timeout": AgentRunSpec(
        transition_to_running=True,
        clear_wake_condition=True,
        increment_wake_count=True,
    ),
}


class SchedulerRunner:
    """Execute one agent cycle and persist the resulting scheduler outcome."""

    def __init__(
        self,
        store: AgentStateStorage,
        coordinator: SchedulerCoordinator,
        semaphore: asyncio.Semaphore,
        state_ops: SchedulerStateOps | None = None,
        wake_message_builder: WakeMessageBuilder | None = None,
    ) -> None:
        self._store = store
        self._coordinator = coordinator
        self._semaphore = semaphore
        self._state_ops = state_ops or SchedulerStateOps(
            store=store,
            coordinator=coordinator,
        )
        self._wake_message_builder = wake_message_builder or WakeMessageBuilder(store)

    async def create_child_agent(self, state: AgentState) -> SchedulerAgentPort:
        """Create a child scheduler agent by copying the parent's template."""
        parent = self._coordinator.get_registered_agent(state.parent_id or "")
        if parent is None:
            raise RuntimeError(
                f"Parent agent '{state.parent_id}' not found in scheduler"
            )

        overrides = deserialize_child_agent_config_overrides(state.config_overrides)
        child = await parent.create_scheduler_child(
            child_id=state.id,
            overrides=ChildAgentOverrides(
                instruction=overrides.instruction,
                system_prompt=overrides.system_prompt,
                exclude_tool_names={"spawn_agent"},
            ),
        )
        self._coordinator.register_agent(child)
        return child

    async def run_root_agent(
        self,
        agent: SchedulerAgentPort,
        user_input: UserInput,
        session_id: str,
        state: AgentState,
    ) -> None:
        """Run the root agent submitted by the caller."""
        await self._execute_agent_run(
            state,
            mode="root",
            agent=agent,
            user_input=user_input,
            session_id=session_id,
            error_log="root_agent_failed",
            error_extra={
                "agent_id": agent.id,
                "state_id": state.id,
                "state_status": state.status.value,
            },
        )

    async def run_queued_root(self, state: AgentState) -> None:
        """Run a queued persistent root with its pending input."""
        agent = await self._load_registered_agent(
            state,
            missing_log="queued_root_agent_not_found",
            missing_message=f"Agent '{state.id}' not found in scheduler for queued run",
        )
        if agent is None:
            return
        user_input = await self._load_pending_input(state)
        await self._execute_agent_run(
            state,
            mode="queued_root",
            agent=agent,
            user_input=user_input,
            session_id=state.resolve_runtime_session_id(),
            error_log="queued_root_agent_failed",
            error_extra={
                "agent_id": state.id,
                "state_id": state.id,
                "state_status": state.status.value,
            },
        )

    async def run_pending_agent(self, state: AgentState) -> None:
        """Run a PENDING child agent."""
        child = await self.create_child_agent(state)
        await self._execute_agent_run(
            state,
            mode="child_pending",
            agent=child,
            user_input=state.task,
            session_id=state.resolve_runtime_session_id(),
            error_log="child_agent_failed",
            error_extra={
                "state_id": state.id,
                "parent_id": state.parent_id,
                "depth": state.depth,
            },
        )

    async def wake_agent(self, state: AgentState) -> None:
        """Wake a WAITING agent by running it with a wake message."""
        agent = await self._load_registered_agent(
            state,
            missing_log="wake_agent_not_found",
            missing_message=f"Agent '{state.id}' not found in scheduler for wake",
        )
        if agent is None:
            return
        wake_message = await self.build_wake_message(state)
        await self._execute_agent_run(
            state,
            mode="wake",
            agent=agent,
            user_input=wake_message,
            session_id=state.resolve_runtime_session_id(),
            error_log="wake_agent_failed",
            error_extra={
                "state_id": state.id,
                "wake_count": state.wake_count,
            },
        )

    async def wake_agent_for_events(
        self, state: AgentState, events: list[PendingEvent]
    ) -> None:
        """Wake a WAITING agent to process accumulated pending events."""
        agent = await self._load_registered_agent(
            state,
            missing_log="wake_agent_for_events_not_found",
        )
        if agent is None:
            return
        await self._execute_agent_run(
            state,
            mode="wake_events",
            agent=agent,
            user_input=self._wake_message_builder.build_from_events(events),
            session_id=state.resolve_runtime_session_id(),
            error_log="wake_agent_for_events_failed",
            error_extra={
                "state_id": state.id,
                "event_count": len(events),
            },
        )

    async def wake_for_timeout(self, state: AgentState) -> None:
        """Wake a timed-out WAITING agent so it can produce a summary report."""
        agent = await self._load_registered_agent(
            state,
            missing_log="timeout_wake_agent_not_found",
            missing_message=f"Agent '{state.id}' not found for timeout wake",
        )
        if agent is None:
            return
        timeout_message = await self._wake_message_builder.build_timeout(state)
        await self._execute_agent_run(
            state,
            mode="wake_timeout",
            agent=agent,
            user_input=timeout_message,
            session_id=state.resolve_runtime_session_id(),
            error_log="timeout_wake_failed",
            error_extra={"state_id": state.id},
        )

    async def _execute_agent_run(
        self,
        state: AgentState,
        mode: AgentRunMode,
        *,
        agent: SchedulerAgentPort,
        user_input: UserInput,
        session_id: str,
        error_log: str,
        error_extra: dict,
        override_spec: AgentRunSpec | None = None,
    ) -> None:
        spec = override_spec or RUN_MODE_TO_SPEC[mode]
        abort_signal = self._coordinator.get_abort_signal(state.id)
        if spec.create_abort_signal:
            abort_signal = AbortSignal()
            self._coordinator.set_abort_signal(state.id, abort_signal)

        try:
            async with self._semaphore:
                if await self._parent_aborted(state, spec):
                    return

                await self._prepare_state_for_run(state, spec, user_input=user_input)

                output = await self._run_agent_cycle(
                    state=state,
                    agent=agent,
                    user_input=user_input,
                    session_id=session_id,
                    abort_signal=abort_signal,
                )
                await self._handle_agent_output(state, output)
        except Exception as error:
            logger.exception(
                error_log,
                **error_extra,
                error=str(error),
                error_type=type(error).__name__,
            )
            await self._state_ops.mark_failed(state, str(error))
            if spec.emit_child_failed_event:
                await self._emit_event_to_parent(
                    state,
                    SchedulerEventType.CHILD_FAILED,
                    {"reason": str(error)},
                )
            if state.is_root:
                await self._coordinator.finish_stream_channel(state.id)
        finally:
            self._coordinator.pop_abort_signal(state.id)
            await self._maybe_cleanup_agent(state)

    async def _run_agent_cycle(
        self,
        *,
        state: AgentState,
        agent: SchedulerAgentPort,
        user_input: UserInput,
        session_id: str,
        abort_signal: AbortSignal | None,
    ) -> RunOutput:
        handle = agent.start(
            user_input,
            session_id=session_id,
            abort_signal=abort_signal,
        )
        self._coordinator.set_execution_handle(state.id, handle)
        try:
            return await self._observe_execution(
                state=state,
                handle=handle,
            )
        finally:
            self._coordinator.pop_execution_handle(state.id)

    async def _observe_execution(
        self,
        *,
        state: AgentState,
        handle: AgentExecutionHandlePort,
    ) -> RunOutput:
        result: RunOutput | None = None
        async for item in consume_execution_stream(
            handle,
            cancel_reason="scheduler event stream closed",
        ):
            await self._fanout_stream_item(state, item)
            result = self._maybe_build_run_output(item, fallback=result)

        final_result = await handle.wait()
        if result is None:
            return final_result
        return result

    def _maybe_build_run_output(
        self,
        item: AgentStreamItem,
        *,
        fallback: RunOutput | None,
    ) -> RunOutput | None:
        if item.type == "run_completed":
            return RunOutput(
                session_id=item.session_id,
                run_id=item.run_id,
                response=item.response,
                metrics=item.metrics,
                termination_reason=item.termination_reason,
            )
        if item.type == "run_failed":
            return RunOutput(
                session_id=item.session_id,
                run_id=item.run_id,
                error=item.error,
                termination_reason=TerminationReason.ERROR,
            )
        return fallback

    async def _fanout_stream_item(
        self,
        state: AgentState,
        item: AgentStreamItem,
    ) -> None:
        root_id = state.id if state.is_root else state.parent_id
        channel_state = self._coordinator.get_stream_channel(root_id)
        if channel_state is None:
            return
        if state.is_root or channel_state.include_child_events:
            await channel_state.queue.put(item)

    async def _parent_aborted(
        self,
        state: AgentState,
        spec: AgentRunSpec,
    ) -> bool:
        if not spec.enforce_parent_abort or state.is_root:
            return False

        parent_signal = self._coordinator.get_abort_signal(state.parent_id)
        if parent_signal is None or not parent_signal.is_aborted():
            return False

        await self._state_ops.mark_failed(state, "Parent cancelled")
        return True

    async def _prepare_state_for_run(
        self,
        state: AgentState,
        spec: AgentRunSpec,
        *,
        user_input: UserInput,
    ) -> None:
        if not spec.transition_to_running:
            return

        refreshed = await self._store.get_state(state.id)
        wake_count = state.wake_count
        if refreshed is not None:
            wake_count = refreshed.wake_count
        if spec.increment_wake_count:
            wake_count += 1

        target = refreshed if refreshed is not None else state
        await self._state_ops.mark_running(
            target,
            task=user_input,
            pending_input=None,
            wake_condition=None
            if spec.clear_wake_condition or state.is_waiting() or state.is_queued_root()
            else ...,
            wake_count=wake_count,
        )

    async def _load_pending_input(self, state: AgentState) -> UserInput:
        refreshed = await self._store.get_state(state.id)
        pending_input = (
            refreshed.pending_input if refreshed is not None else state.pending_input
        )
        if pending_input is None:
            raise RuntimeError(f"Queued input for state '{state.id}' is missing")
        return pending_input

    async def _load_registered_agent(
        self,
        state: AgentState,
        *,
        missing_log: str,
        missing_message: str | None = None,
    ) -> SchedulerAgentPort | None:
        agent = self._coordinator.get_registered_agent(state.id)
        if agent is not None:
            return agent

        logger.error(missing_log, state_id=state.id)
        if missing_message is not None:
            await self._state_ops.mark_failed(state, missing_message)
            if state.is_root:
                await self._coordinator.finish_stream_channel(state.id)
        return None

    async def _handle_agent_output(self, state: AgentState, output: RunOutput) -> None:
        """Persist the state transition implied by a completed agent cycle."""
        text = output.response

        if output.termination_reason == TerminationReason.SLEEPING:
            await self._emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_SLEEP_RESULT,
                {"result": text or "", "explain": state.explain},
            )
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
                refreshed_for_wait = await self._store.get_state(state.id)
                target_for_wait = (
                    refreshed_for_wait if refreshed_for_wait is not None else state
                )
                await self._state_ops.mark_waiting(
                    target_for_wait,
                    wake_condition=new_wc,
                    result_summary=text,
                )
                await self._emit_event_to_parent(
                    state,
                    SchedulerEventType.CHILD_SLEEP_RESULT,
                    {"result": text or "", "explain": state.explain, "periodic": True},
                )
                return

        if is_persistent:
            refreshed_for_idle = await self._store.get_state(state.id)
            target_for_idle = (
                refreshed_for_idle if refreshed_for_idle is not None else state
            )
            await self._state_ops.mark_idle(
                target_for_idle,
                result_summary=text,
            )
            return

        refreshed_for_complete = await self._store.get_state(state.id)
        target_for_complete = (
            refreshed_for_complete if refreshed_for_complete is not None else state
        )
        await self._state_ops.mark_completed(
            target_for_complete,
            result_summary=text,
        )
        await self._emit_event_to_parent(
            state,
            SchedulerEventType.CHILD_COMPLETED,
            {"result": text or ""},
        )

    async def _emit_event_to_parent(
        self,
        state: AgentState,
        event_type: SchedulerEventType,
        payload: dict,
    ) -> None:
        """Create a PendingEvent for the parent agent if this is a child agent."""
        if state.is_root:
            return
        parent_state = await self._store.get_state(state.parent_id)
        if parent_state is None:
            return

        event = PendingEvent(
            id=str(uuid4()),
            target_agent_id=state.parent_id,
            session_id=state.session_id,
            event_type=event_type,
            payload={
                **payload,
                "child_agent_id": state.id,
                "child_task": state.task
                if isinstance(state.task, str)
                else str(state.task),
            },
            source_agent_id=state.id,
            created_at=datetime.now(timezone.utc),
        )
        await self._store.save_event(event)
        logger.info(
            "pending_event_created",
            event_type=event_type.value,
            target_agent_id=state.parent_id,
            source_agent_id=state.id,
        )

    async def _maybe_cleanup_agent(self, state: AgentState) -> None:
        """Release dispatch reservation and clean finished child agents."""
        self._coordinator.release_state_dispatch(state.id)
        if state.is_root:
            return
        refreshed = await self._store.get_state(state.id)
        if refreshed is None or refreshed.status in (
            AgentStateStatus.COMPLETED,
            AgentStateStatus.FAILED,
        ):
            agent = self._coordinator.get_registered_agent(state.id)
            self._coordinator.unregister_agent(state.id)
            if agent is not None:
                await agent.close()

    async def build_wake_message(self, state: AgentState) -> UserInput:
        """Build a wake message with auto-injected child results."""
        return await self._wake_message_builder.build(state)


__all__ = ["AgentRunSpec", "SchedulerRunner"]
