"""
SchedulerRunner — execute one scheduler dispatch action.

The runner owns only a single execution cycle. It does not decide *which*
states should run next; that remains the engine's job.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections.abc import Callable
from uuid import uuid4

from agiwo.agent import Agent
from agiwo.agent import UserInput
from agiwo.agent import AgentStreamItem, RunOutput, TerminationReason
from agiwo.scheduler.commands import DispatchAction, DispatchReason
from agiwo.scheduler.formatting import (
    build_child_result_detail_lines,
    format_child_results_summary,
)
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
    WakeType,
)
from agiwo.scheduler.runtime_state import ExecutionHandleLike, RuntimeState
from agiwo.scheduler.store import AgentStateStorage
from agiwo.scheduler.store.codec import deserialize_child_agent_config_overrides
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

_SHUTDOWN_SUMMARY_TASK = (
    "System shutdown requested. Please produce a final summary "
    "report of all work done so far."
)
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
    scheduling_tools: tuple[object, ...]


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
        parent = self._ctx.rt.agents.get(state.parent_id or "")
        if parent is None:
            raise RuntimeError(f"Parent agent '{state.parent_id}' not found in runtime")

        overrides = deserialize_child_agent_config_overrides(state.config_overrides)
        child = await parent.create_child_agent(
            child_id=state.id,
            instruction=overrides.instruction,
            system_prompt_override=overrides.system_prompt,
            exclude_tool_names={"spawn_agent"},
            extra_tools=[
                tool
                for tool in self._ctx.scheduling_tools
                if tool.get_name() != "spawn_agent"
            ],
        )
        self._ctx.rt.agents[state.id] = child
        return child

    async def build_wake_message(self, state: AgentState) -> UserInput:
        wc = state.wake_condition
        if wc is None:
            return "You have been woken up. Please continue your task."

        if wc.type == WakeType.WAITSET:
            succeeded, failed = await self._collect_child_results(state)
            done = len(succeeded) + len(failed)
            return format_child_results_summary(
                header=f"Child agents completed ({done}/{len(wc.wait_for)}).",
                succeeded=succeeded,
                failed=failed,
                closing_instruction=(
                    "Please synthesize a final response based on the successful "
                    "results above."
                ),
            )
        if wc.type == WakeType.TIMER:
            return "The scheduled delay has elapsed. Please continue your task."
        if wc.type == WakeType.PERIODIC:
            return (
                "A scheduled periodic check has triggered. "
                "Please check progress and decide whether to continue waiting "
                "or produce a final result."
            )
        return "You have been woken up. Please continue your task."

    async def build_timeout_message(self, state: AgentState) -> str:
        succeeded, failed = await self._collect_child_results(state)
        wc = state.wake_condition
        done = len(succeeded) + len(failed)
        return format_child_results_summary(
            header="Wait timeout reached.",
            succeeded=succeeded,
            failed=failed,
            progress_line=f"Completed children: {done}/{len(wc.wait_for) if wc else 0}",
            closing_instruction=(
                "Please produce a summary report with whatever results are available."
            ),
        )

    def build_events_message(self, events: tuple[PendingEvent, ...]) -> str:
        lines = [f"You have {len(events)} new notification(s):\n"]
        for event in events:
            event_label = event.event_type.value.replace("_", " ").title()
            child_id = event.payload.get(
                "child_agent_id", event.source_agent_id or "unknown"
            )
            lines.append(f"### {event_label} - Agent: {child_id}")
            if event.event_type == SchedulerEventType.CHILD_SLEEP_RESULT:
                lines.extend(
                    build_child_result_detail_lines(
                        result=event.payload.get("result", ""),
                        explain=event.payload.get("explain"),
                        periodic=event.payload.get("periodic", False),
                        result_as_block=True,
                    )
                )
            elif event.event_type == SchedulerEventType.CHILD_COMPLETED:
                lines.extend(
                    build_child_result_detail_lines(
                        result=event.payload.get("result", ""),
                        result_as_block=True,
                    )
                )
            elif event.event_type == SchedulerEventType.CHILD_FAILED:
                lines.extend(
                    build_child_result_detail_lines(
                        failure_reason=event.payload.get("reason", "Unknown failure")
                    )
                )
            elif event.event_type == SchedulerEventType.USER_HINT:
                hint = event.payload.get("hint", "")
                if hint:
                    lines.append(f"User hint: {hint}")
            lines.append("")
        lines.append(
            "Please review these notifications and take appropriate action "
            "(e.g., summarize results for the user, cancel stuck agents, etc.)."
        )
        return "\n".join(lines)

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
                else self.build_events_message(action.events)
            )
        elif action.reason == DispatchReason.WAKE_TIMEOUT:
            user_input = (
                action.input_override
                if action.input_override is not None
                else await self.build_timeout_message(state)
            )
        else:
            user_input = (
                action.input_override
                if action.input_override is not None
                else await self.build_wake_message(state)
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

        await self._fail_state(state.id, "Parent cancelled")
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

        if await self._handle_periodic_output(action, current_state, text):
            return

        await self._complete_state(current_state, text)

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
            await self._ctx.store.delete_events([event.id for event in action.events])
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

    async def _fail_state(self, state_id: str, reason: str) -> None:
        current_state = await self._ctx.store.get_state(state_id)
        if current_state is None:
            return
        await self._save_state(current_state.with_failed(reason))

    async def _cleanup_after_run(self, state: AgentState) -> None:
        self._ctx.rt.dispatched.discard(state.id)
        if state.is_root:
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

    async def _collect_child_results(
        self,
        state: AgentState,
    ) -> tuple[dict[str, str], dict[str, str]]:
        wc = state.wake_condition
        child_ids = wc.wait_for if wc else []
        if not child_ids:
            child_ids = [
                child.id
                for child in await self._ctx.store.list_states(
                    parent_id=state.id,
                    limit=1000,
                )
            ]

        succeeded: dict[str, str] = {}
        failed: dict[str, str] = {}
        for child_id in child_ids:
            child = await self._ctx.store.get_state(child_id)
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
                    pending_input=_SHUTDOWN_SUMMARY_TASK,
                ).with_updates(result_summary=text)
            )
            return True

        await self._save_state(state.with_failed("Shutdown before completion"))
        if state.is_child:
            await self._emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_FAILED,
                {"reason": "Shutdown before completion"},
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
        await self._save_state(state.with_failed(reason))
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
    ) -> bool:
        original_wc = action.state.wake_condition
        if original_wc is None or original_wc.type != WakeType.PERIODIC:
            return False

        secs = original_wc.to_seconds()
        if secs is None:
            return False

        next_wakeup = datetime.now(timezone.utc) + timedelta(seconds=secs)
        await self._save_state(
            state.with_waiting(
                wake_condition=original_wc.with_next_wakeup(next_wakeup),
                result_summary=text,
            )
        )
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

    async def _complete_state(
        self,
        state: AgentState,
        text: str | None,
    ) -> None:
        if state.is_root and state.is_persistent:
            await self._save_state(state.with_idle(result_summary=text))
            return

        await self._save_state(state.with_completed(result_summary=text))
        if state.is_child:
            await self._emit_event_to_parent(
                state,
                SchedulerEventType.CHILD_COMPLETED,
                {"result": text or ""},
            )


__all__ = ["RunnerContext", "SchedulerRunner"]
