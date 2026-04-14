"""
SchedulerRunner — execute one scheduler dispatch action.

The runner owns only a single execution cycle. It does not decide *which*
states should run next; that remains the engine's job.
"""

import asyncio
import dataclasses
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections.abc import Callable
from uuid import uuid4

from agiwo.agent import Agent, AgentStreamItem, RunOutput, TerminationReason, UserInput
from agiwo.scheduler.commands import DispatchAction, DispatchReason
from agiwo.scheduler.formatting import (
    SHUTDOWN_SUMMARY_TASK,
    build_events_message,
    build_fork_task_notice,
    build_timeout_message,
    build_wake_message,
)
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerRunResult,
    SchedulerEventType,
    WakeType,
)
from agiwo.scheduler.runtime_state import ExecutionHandleLike, RuntimeState
from agiwo.scheduler.store.base import AgentStateStorage
from agiwo.scheduler.store.codec import deserialize_child_agent_config_overrides
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

_CHILD_EXCLUDED_SYSTEM_TOOLS: frozenset[str] = frozenset({"spawn_agent"})

_FAILED_TERMINATIONS = frozenset(
    {
        TerminationReason.CANCELLED,
        TerminationReason.ERROR,
        TerminationReason.ERROR_WITH_CONTEXT,
        TerminationReason.TIMEOUT,
    }
)


@dataclass(frozen=True, slots=True)
class RunnerContext:
    """All external runner dependencies, nothing more."""

    store: AgentStateStorage
    rt: RuntimeState
    notify_state_change: Callable[[str], None]
    nudge: Callable[[], None]
    semaphore: asyncio.Semaphore


class SchedulerRunner:
    """Execute one agent cycle and translate it back into scheduler state."""

    def __init__(self, context: RunnerContext) -> None:
        self._ctx = context

    async def run(self, action: DispatchAction) -> None:
        state = action.state
        abort_signal = self._ctx.rt.abort_signals.get(state.id)
        if action.reason == DispatchReason.CHILD_PENDING and abort_signal is None:
            abort_signal = AbortSignal()
            self._ctx.rt.abort_signals[state.id] = abort_signal

        try:
            async with self._ctx.semaphore:
                if abort_signal is not None and abort_signal.is_aborted():
                    return
                if action.reason == DispatchReason.CHILD_PENDING:
                    if await self._parent_aborted(state):
                        return

                agent = await self._resolve_agent(action)
                if agent is None:
                    return

                user_input = await self._prepare_state_for_run(action)
                output = await self._run_agent_cycle(
                    action=action,
                    state=state,
                    agent=agent,
                    user_input=user_input,
                    session_id=state.resolve_runtime_session_id(),
                    abort_signal=abort_signal,
                )
                await self._handle_agent_output(action, output)
        except Exception as error:  # noqa: BLE001
            logger.exception(
                "scheduler_dispatch_failed",
                state_id=state.id,
                reason=action.reason.value,
                error=str(error),
                error_type=type(error).__name__,
            )
            await self._fail_state(state.id, str(error))
            if state.is_child:
                await self._emit_event_to_parent(
                    state,
                    SchedulerEventType.CHILD_FAILED,
                    {"reason": str(error)},
                )
        finally:
            self._ctx.rt.abort_signals.pop(state.id, None)
            await self._cleanup_after_run(state)

    async def create_child_agent(self, state: AgentState) -> Agent:
        """Create a child agent for a pending scheduler state.

        System tools (scheduler runtime tools) are passed separately from
        functional/user tools.  ``spawn_agent`` is excluded for non-fork
        children; fork children inherit ALL system tools for KV cache reuse
        (the gate check on ``SpawnAgentTool`` still blocks actual spawning).
        """
        parent = self._ctx.rt.agents.get(state.parent_id or "")
        if parent is None:
            raise RuntimeError(f"Parent agent '{state.parent_id}' not found in runtime")

        overrides = deserialize_child_agent_config_overrides(state.config_overrides)

        if overrides.fork:
            child_system_tools = list(parent.system_tools)
        else:
            child_system_tools = [
                t
                for t in parent.system_tools
                if t.name not in _CHILD_EXCLUDED_SYSTEM_TOOLS
            ]

        child = await parent.create_child_agent(
            child_id=state.id,
            instruction=overrides.instruction,
            system_prompt_override=overrides.system_prompt,
            child_allowed_tools=list(overrides.allowed_tools)
            if overrides.allowed_tools is not None
            else None,
            child_allowed_skills=list(overrides.allowed_skills)
            if overrides.allowed_skills is not None
            else None,
            inherit_all_extra_tools=overrides.fork,
            system_tools=child_system_tools or None,
        )

        if overrides.fork:
            await self._copy_session_steps_for_fork(parent, child, state)

        self._ctx.rt.agents[state.id] = child
        return child

    async def _copy_session_steps_for_fork(
        self,
        parent: Agent,
        child: Agent,
        state: AgentState,
    ) -> None:
        """Copy parent's session steps into the child's session for fork mode."""
        parent_state = await self._ctx.store.get_state(state.parent_id or "")
        if parent_state is None:
            return
        parent_session_id = parent_state.resolve_runtime_session_id()
        child_session_id = state.resolve_runtime_session_id()

        parent_steps = await parent.run_step_storage.get_steps(
            session_id=parent_session_id,
            agent_id=state.parent_id,
        )
        if not parent_steps:
            return

        forked_steps = [
            dataclasses.replace(
                step,
                id=str(uuid4()),
                session_id=child_session_id,
                agent_id=state.id,
            )
            for step in parent_steps
        ]
        await child.run_step_storage.save_steps_batch(forked_steps)
        logger.info(
            "fork_steps_copied",
            parent_id=state.parent_id,
            child_id=state.id,
            steps_count=len(forked_steps),
        )

    async def _build_wake_message(self, state: AgentState) -> UserInput:
        succeeded, failed = await self._collect_child_results(state)
        return build_wake_message(state.wake_condition, succeeded, failed)

    async def _build_timeout_message(self, state: AgentState) -> str:
        succeeded, failed = await self._collect_child_results(state)
        return build_timeout_message(state.wake_condition, succeeded, failed)

    async def _resolve_agent(
        self,
        action: DispatchAction,
    ) -> Agent | None:
        state = action.state
        if action.reason == DispatchReason.CHILD_PENDING:
            return await self.create_child_agent(state)

        agent = self._ctx.rt.agents.get(state.id)
        if agent is not None:
            return agent

        await self._fail_state(state.id, f"Agent '{state.id}' not found in scheduler")
        return None

    async def _prepare_state_for_run(self, action: DispatchAction) -> UserInput:
        state = action.state
        if action.reason == DispatchReason.ROOT_SUBMIT:
            return (
                action.input_override
                if action.input_override is not None
                else state.task
            )

        if action.reason == DispatchReason.ROOT_QUEUED_INPUT:
            if action.input_override is None:
                raise RuntimeError(f"Queued input for state '{state.id}' is missing")
            await self._save_state(
                state.with_running(
                    task=action.input_override,
                    pending_input=None,
                    wake_condition=None,
                    explain=None,
                )
            )
            return action.input_override

        if action.reason == DispatchReason.CHILD_PENDING:
            user_input = (
                action.input_override
                if action.input_override is not None
                else state.task
            )
            overrides = deserialize_child_agent_config_overrides(state.config_overrides)
            if overrides.fork:
                user_input = build_fork_task_notice(user_input)
            await self._save_state(
                state.with_running(
                    task=user_input,
                    pending_input=None,
                    wake_condition=None,
                )
            )
            return user_input

        if action.reason == DispatchReason.WAKE_EVENTS:
            user_input = (
                action.input_override
                if action.input_override is not None
                else build_events_message(action.events)
            )
        elif action.reason == DispatchReason.WAKE_TIMEOUT:
            user_input = (
                action.input_override
                if action.input_override is not None
                else await self._build_timeout_message(state)
            )
        else:
            user_input = (
                action.input_override
                if action.input_override is not None
                else await self._build_wake_message(state)
            )

        await self._save_state(
            state.with_running(
                task=user_input,
                pending_input=None,
                wake_condition=None,
                wake_count=state.wake_count + 1,
            )
        )
        return user_input

    async def _run_agent_cycle(
        self,
        *,
        action: DispatchAction,
        state: AgentState,
        agent: Agent,
        user_input: UserInput,
        session_id: str,
        abort_signal: AbortSignal | None,
    ) -> RunOutput:
        handle = agent.start(
            user_input,
            session_id=session_id,
            abort_signal=abort_signal,
        )
        self._ctx.rt.execution_handles[state.id] = handle
        try:
            await self._ack_action_events(action)
            return await self._observe_execution(state=state, handle=handle)
        finally:
            self._ctx.rt.execution_handles.pop(state.id, None)

    async def _observe_execution(
        self,
        *,
        state: AgentState,
        handle: ExecutionHandleLike,
    ) -> RunOutput:
        result: RunOutput | None = None
        completed = False
        try:
            async for item in handle.stream():
                await self._fanout_stream_item(state, item)
                result = self._maybe_build_run_output(item, fallback=result)
            await handle.wait()
            completed = True
        finally:
            if not completed:
                handle.cancel("scheduler event stream closed")
                try:
                    await handle.wait()
                except asyncio.CancelledError:
                    pass
        if result is None:
            raise RuntimeError(
                f"Agent '{state.id}' execution stream ended without a terminal result"
            )
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

    @staticmethod
    def _build_last_run_result(
        *,
        termination_reason: TerminationReason,
        run_id: str | None,
        summary: str | None = None,
        error: str | None = None,
    ) -> SchedulerRunResult:
        return SchedulerRunResult(
            run_id=run_id,
            termination_reason=termination_reason,
            summary=summary,
            error=error,
        )

    async def _fanout_stream_item(
        self, state: AgentState, item: AgentStreamItem
    ) -> None:
        root_id = state.id if state.is_root else state.parent_id
        if root_id is None:
            return
        channel = self._ctx.rt.stream_channels.get(root_id)
        if channel is None:
            return
        if state.is_root or channel.include_child_events:
            await channel.queue.put(item)

    async def _parent_aborted(self, state: AgentState) -> bool:
        parent_signal = self._ctx.rt.abort_signals.get(state.parent_id)
        if parent_signal is None or not parent_signal.is_aborted():
            return False

        await self._fail_state(
            state.id,
            "Parent cancelled",
            termination_reason=TerminationReason.CANCELLED,
        )
        await self._emit_event_to_parent(
            state,
            SchedulerEventType.CHILD_FAILED,
            {"reason": "Parent cancelled"},
        )
        return True

    async def _handle_agent_output(
        self,
        action: DispatchAction,
        output: RunOutput,
    ) -> None:
        state = action.state
        current_state = await self._ctx.store.get_state(state.id)
        if current_state is None:
            return

        if self._should_preserve_terminal_abort(current_state):
            return

        text = output.response

        if await self._handle_shutdown_requested(current_state, text):
            return
        if await self._handle_failed_output(current_state, output, text):
            return
        if output.termination_reason == TerminationReason.SLEEPING:
            await self._emit_event_to_parent(
                current_state,
                SchedulerEventType.CHILD_SLEEP_RESULT,
                {"result": text or "", "explain": current_state.explain},
            )
            return

        if await self._handle_periodic_output(action, current_state, text, output):
            return

        await self._complete_state(current_state, output, text)

    async def _emit_event_to_parent(
        self,
        state: AgentState,
        event_type: SchedulerEventType,
        payload: dict[str, object],
    ) -> None:
        if state.is_root:
            return

        parent_state = await self._ctx.store.get_state(state.parent_id or "")
        if parent_state is None:
            return

        event = PendingEvent(
            id=str(uuid4()),
            target_agent_id=state.parent_id or "",
            session_id=state.session_id,
            event_type=event_type,
            payload={
                **payload,
                "child_agent_id": state.id,
                "child_task": (
                    state.task if isinstance(state.task, str) else str(state.task)
                ),
            },
            source_agent_id=state.id,
            created_at=datetime.now(timezone.utc),
        )
        await self._ctx.store.save_event(event)
        self._ctx.nudge()

    async def _ack_action_events(self, action: DispatchAction) -> None:
        if not action.events:
            return
        try:
            event_ids = [event.id for event in action.events]
            await self._ctx.store.delete_events(event_ids)
            logger.info(
                "scheduler_events_acknowledged",
                state_id=action.state.id,
                reason=action.reason.value,
                event_ids=event_ids,
                source_agent_ids=[
                    event.source_agent_id
                    for event in action.events
                    if event.source_agent_id is not None
                ],
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "scheduler_event_ack_failed",
                state_id=action.state.id,
                reason=action.reason.value,
            )

    async def _save_state(self, state: AgentState) -> None:
        await self._ctx.store.save_state(state)
        self._ctx.notify_state_change(state.id)
        self._ctx.nudge()

    async def _fail_state(
        self,
        state_id: str,
        reason: str,
        *,
        termination_reason: TerminationReason = TerminationReason.ERROR,
    ) -> None:
        current_state = await self._ctx.store.get_state(state_id)
        if current_state is None:
            return
        await self._save_state(
            current_state.with_failed(reason).with_updates(
                last_run_result=self._build_last_run_result(
                    termination_reason=termination_reason,
                    run_id=None,
                    error=reason,
                )
            )
        )

    async def _cleanup_after_run(self, state: AgentState) -> None:
        self._ctx.rt.dispatched.discard(state.id)
        if state.is_root:
            refreshed = await self._ctx.store.get_state(state.id)
            if self._should_finish_root_stream_channel(state.id, refreshed):
                await self._finish_stream_channel(state.id)
            return

        refreshed = await self._ctx.store.get_state(state.id)
        if refreshed is None or refreshed.status in (
            AgentStateStatus.COMPLETED,
            AgentStateStatus.FAILED,
        ):
            agent = self._ctx.rt.agents.pop(state.id, None)
            self._ctx.rt.execution_handles.pop(state.id, None)
            if agent is not None:
                await agent.close()

    async def _finish_stream_channel(self, state_id: str) -> None:
        channel = self._ctx.rt.stream_channels.get(state_id)
        if channel is not None:
            await channel.queue.put(None)

    def _should_finish_root_stream_channel(
        self,
        state_id: str,
        state: AgentState | None,
    ) -> bool:
        channel = self._ctx.rt.stream_channels.get(state_id)
        if channel is None:
            return False
        if channel.close_on_root_run_end:
            return True
        if state is None:
            return True
        return state.status in (
            AgentStateStatus.IDLE,
            AgentStateStatus.COMPLETED,
            AgentStateStatus.FAILED,
        )

    async def _collect_child_results(
        self,
        state: AgentState,
    ) -> tuple[dict[str, str], dict[str, str]]:
        wc = state.wake_condition
        target_ids = set(wc.wait_for) if wc and wc.wait_for else None

        children = await self._ctx.store.list_states(
            parent_id=state.id,
            limit=1000,
        )
        children_by_id = {child.id: child for child in children}

        if target_ids is None:
            target_ids = set(children_by_id.keys())

        succeeded: dict[str, str] = {}
        failed: dict[str, str] = {}
        for child_id in target_ids:
            child = children_by_id.get(child_id)
            if child is None:
                failed[child_id] = "Agent state not found"
            elif child.status == AgentStateStatus.FAILED:
                failed[child_id] = child.result_summary or "Unknown failure"
            elif child.status == AgentStateStatus.COMPLETED:
                succeeded[child_id] = child.result_summary or "Completed"
            else:
                failed[child_id] = f"Not finished: status={child.status.value}"
        return succeeded, failed

    def _should_preserve_terminal_abort(self, state: AgentState) -> bool:
        abort_signal = self._ctx.rt.abort_signals.get(state.id)
        return (
            state.is_terminal()
            and abort_signal is not None
            and abort_signal.is_aborted()
        )

    async def _handle_shutdown_requested(
        self,
        state: AgentState,
        text: str | None,
    ) -> bool:
        if state.id not in self._ctx.rt.shutdown_requested:
            return False

        self._ctx.rt.shutdown_requested.discard(state.id)
        if state.is_root and state.is_persistent:
            await self._save_state(
                state.with_queued(
                    pending_input=SHUTDOWN_SUMMARY_TASK,
                ).with_updates(result_summary=text)
            )
            return True

        error = "Shutdown before completion"
        await self._save_state(
            state.with_failed(error).with_updates(
                last_run_result=self._build_last_run_result(
                    termination_reason=TerminationReason.ERROR,
                    run_id=None,
                    summary=text,
                    error=error,
                )
            )
        )
        if state.is_child:
            await self._emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_FAILED,
                {"reason": error},
            )
        return True

    async def _handle_failed_output(
        self,
        state: AgentState,
        output: RunOutput,
        text: str | None,
    ) -> bool:
        if output.termination_reason not in _FAILED_TERMINATIONS:
            return False

        reason = output.error or text or output.termination_reason.value
        await self._save_state(
            state.with_failed(reason).with_updates(
                last_run_result=self._build_last_run_result(
                    termination_reason=output.termination_reason,
                    run_id=output.run_id,
                    summary=text,
                    error=reason,
                )
            )
        )
        if state.is_child:
            await self._emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_FAILED,
                {"reason": reason},
            )
        return True

    async def _handle_periodic_output(
        self,
        action: DispatchAction,
        state: AgentState,
        text: str | None,
        output: RunOutput,
    ) -> bool:
        original_wc = action.state.wake_condition
        if original_wc is None or original_wc.type != WakeType.PERIODIC:
            return False

        secs = original_wc.to_seconds()
        if secs is None:
            return False

        should_rollback = False
        if state.no_progress:
            agent = self._ctx.rt.agents.get(state.id)
            should_rollback = (
                agent is not None and agent.config.options.enable_context_rollback
            )
            if should_rollback:
                await self._rollback_run_steps(state, output)

        next_wakeup = datetime.now(timezone.utc) + timedelta(seconds=secs)
        new_state = state.with_waiting(
            wake_condition=original_wc.with_next_wakeup(next_wakeup),
            result_summary=text if not should_rollback else state.result_summary,
        )
        if should_rollback:
            new_state = new_state.with_updates(
                rollback_count=state.rollback_count + 1,
            )
        await self._save_state(new_state)
        if not should_rollback:
            await self._emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_SLEEP_RESULT,
                {
                    "result": text or "",
                    "explain": state.explain,
                    "periodic": True,
                },
            )
        return True

    async def _rollback_run_steps(
        self,
        state: AgentState,
        output: RunOutput,
    ) -> None:
        run_start_seq = output.metadata.get("run_start_seq")
        if run_start_seq is None:
            return
        agent = self._ctx.rt.agents.get(state.id)
        if agent is None:
            return
        session_id = state.resolve_runtime_session_id()
        storage = agent.run_step_storage
        deleted = await storage.delete_steps(session_id, start_seq=run_start_seq)
        logger.info(
            "context_rollback_executed",
            state_id=state.id,
            session_id=session_id,
            run_start_seq=run_start_seq,
            deleted_steps=deleted,
            rollback_count=state.rollback_count + 1,
        )

    async def _complete_state(
        self,
        state: AgentState,
        output: RunOutput,
        text: str | None,
    ) -> None:
        last_run_result = self._build_last_run_result(
            termination_reason=TerminationReason.COMPLETED,
            run_id=output.run_id,
            summary=text,
        )
        if state.is_root and state.is_persistent:
            await self._save_state(
                state.with_idle(result_summary=text).with_updates(
                    last_run_result=last_run_result
                )
            )
            return

        await self._save_state(
            state.with_completed(result_summary=text).with_updates(
                last_run_result=last_run_result
            )
        )
        if state.is_child:
            await self._emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_COMPLETED,
                {"result": text or ""},
            )


__all__ = ["RunnerContext", "SchedulerRunner"]
