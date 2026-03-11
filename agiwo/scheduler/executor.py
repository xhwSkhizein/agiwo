"""
SchedulerExecutor — Agent execution logic for the Scheduler.

Handles running, waking, and output processing for scheduled agents.
Extracted from scheduler.py to reduce file size and improve maintainability.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Literal
from uuid import uuid4

from agiwo.agent.agent import Agent
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.schema import (
    RunOutput,
    StepRecord,
    TerminationReason,
    UserInput,
    normalize_to_message,
)
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    ChildAgentConfigOverrides,
    PendingEvent,
    SchedulerEventType,
    SchedulerOutput,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.formatting import (
    build_child_result_detail_lines,
    format_child_results_summary,
)
from agiwo.scheduler.runtime import SchedulerRuntime
from agiwo.scheduler.store import AgentStateStorage
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


AgentRunMode = Literal[
    "root",
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
    emit_error_output: bool = False
    enforce_parent_abort: bool = False
    emit_child_failed_event: bool = False


RUN_MODE_TO_SPEC: dict[AgentRunMode, AgentRunSpec] = {
    "root": AgentRunSpec(emit_error_output=True),
    "child_pending": AgentRunSpec(
        transition_to_running=True,
        create_abort_signal=True,
        enforce_parent_abort=True,
        emit_child_failed_event=True,
    ),
    "wake": AgentRunSpec(
        transition_to_running=True,
        increment_wake_count=True,
        emit_error_output=True,
    ),
    "wake_events": AgentRunSpec(
        transition_to_running=True,
        increment_wake_count=True,
        emit_error_output=True,
    ),
    "wake_timeout": AgentRunSpec(
        transition_to_running=True,
        clear_wake_condition=True,
        increment_wake_count=True,
        emit_error_output=True,
    ),
}


class SchedulerExecutor:
    """
    Handles agent execution within the Scheduler.

    Responsible for:
    - Running root and child agents
    - Waking sleeping agents
    - Handling agent output (completion, sleep, periodic)
    - Building wake messages with child results
    - Generating pending events for parent agents
    - Syncing step activity to AgentState via on_step hook
    """

    def __init__(
        self,
        store: AgentStateStorage,
        runtime: SchedulerRuntime,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self._store = store
        self._runtime = runtime
        self._semaphore = semaphore

    async def create_child_agent(self, state: AgentState) -> Agent:
        """Create a child Agent by copying the parent's configuration."""
        parent = self._runtime.get_registered_agent(state.parent_id or "")
        if parent is None:
            raise RuntimeError(
                f"Parent agent '{state.parent_id}' not found in scheduler"
            )

        overrides = ChildAgentConfigOverrides.from_dict(state.config_overrides)
        child = await parent.derive_child(
            child_id=state.id,
            instruction=overrides.instruction,
            system_prompt_override=overrides.system_prompt,
            exclude_tool_names={"spawn_agent"},
        )
        self._runtime.register_agent(child)
        self._wrap_on_step_hook(child, state.id)
        return child

    def _wrap_on_step_hook(self, agent: Agent, state_id: str) -> None:
        """Wrap agent's on_step hook to sync last_activity_at and recent_steps to AgentState."""
        if agent.hooks is None:
            agent.hooks = AgentHooks()

        original_hook = agent.hooks.on_step

        async def scheduler_on_step(step: StepRecord) -> None:
            if original_hook:
                await original_hook(step)
            await self._sync_step_to_state(state_id, step)

        agent.hooks.on_step = scheduler_on_step

    async def _sync_step_to_state(self, state_id: str, step: StepRecord) -> None:
        """Update last_activity_at and rolling recent_steps in AgentState."""
        try:
            now = datetime.now(timezone.utc)
            step_summary: dict = {
                "role": step.role.value
                if hasattr(step.role, "value")
                else str(step.role),
                "timestamp": now.isoformat(),
            }
            if step.tool_calls:
                step_summary["tool_calls"] = [
                    tc.get("function", {}).get("name", "unknown")
                    for tc in step.tool_calls
                ]
            if step.tool_name:
                step_summary["tool_name"] = step.tool_name

            await self._store.append_recent_step(state_id, step_summary)
        except Exception:  # noqa: BLE001 - scheduler hook boundary
            # Non-critical — don't propagate hook errors
            pass

    async def run_root_agent(
        self,
        agent: Agent,
        user_input: UserInput,
        session_id: str,
        state: AgentState,
    ) -> None:
        """Run the root agent (submitted by user)."""
        self._wrap_on_step_hook(agent, state.id)
        await self._execute_agent_run(
            state,
            mode="root",
            agent_loader=self._constant_agent_loader(agent),
            user_input_loader=self._constant_input_loader(user_input),
            session_id=session_id,
            error_log="root_agent_failed",
            error_extra={
                "agent_id": agent.id,
                "state_id": state.id,
                "state_status": state.status.value,
            },
        )

    async def run_agent(self, state: AgentState) -> None:
        """Run a PENDING child agent."""
        await self._execute_agent_run(
            state,
            mode="child_pending",
            agent_loader=lambda: self.create_child_agent(state),
            user_input_loader=self._constant_input_loader(state.task),
            session_id=state.id,
            error_log="child_agent_failed",
            error_extra={
                "state_id": state.id,
                "parent_id": state.parent_id,
                "depth": state.depth,
            },
        )

    async def wake_agent(self, state: AgentState) -> None:
        """Wake a SLEEPING agent by running it with a wake message."""
        await self._execute_agent_run(
            state,
            mode="wake",
            agent_loader=lambda: self._load_registered_agent(
                state,
                missing_log="wake_agent_not_found",
                missing_message=f"Agent '{state.id}' not found in scheduler for wake",
            ),
            user_input_loader=lambda: self._build_wake_message(state),
            session_id=state.session_id,
            error_log="wake_agent_failed",
            error_extra={
                "state_id": state.id,
                "wake_count": state.wake_count,
            },
        )

    async def wake_agent_for_events(
        self, state: AgentState, events: list[PendingEvent]
    ) -> None:
        """Wake a SLEEPING agent to process accumulated pending events."""
        await self._execute_agent_run(
            state,
            mode="wake_events",
            agent_loader=lambda: self._load_registered_agent(
                state,
                missing_log="wake_agent_for_events_not_found",
            ),
            user_input_loader=self._constant_input_loader(
                self._build_events_wake_message(events)
            ),
            session_id=state.session_id,
            error_log="wake_agent_for_events_failed",
            error_extra={
                "state_id": state.id,
                "event_count": len(events),
            },
        )

    async def wake_for_timeout(self, state: AgentState) -> None:
        """Wake a timed-out SLEEPING agent so it can produce a summary report."""
        await self._execute_agent_run(
            state,
            mode="wake_timeout",
            agent_loader=lambda: self._load_registered_agent(
                state,
                missing_log="timeout_wake_agent_not_found",
                missing_message=f"Agent '{state.id}' not found for timeout wake",
            ),
            user_input_loader=lambda: self._build_timeout_wake_message(state),
            session_id=state.session_id,
            error_log="timeout_wake_failed",
            error_extra={"state_id": state.id},
        )

    async def _execute_agent_run(
        self,
        state: AgentState,
        mode: AgentRunMode,
        *,
        agent_loader: Callable[[], Awaitable[Agent | None]],
        user_input_loader: Callable[[], Awaitable[UserInput] | UserInput],
        session_id: str,
        error_log: str,
        error_extra: dict,
        override_spec: AgentRunSpec | None = None,
    ) -> None:
        spec = override_spec or RUN_MODE_TO_SPEC[mode]
        abort_signal = self._runtime.get_abort_signal(state.id)
        if spec.create_abort_signal:
            abort_signal = AbortSignal()
            self._runtime.set_abort_signal(state.id, abort_signal)

        try:
            async with self._semaphore:
                if await self._parent_aborted(state, spec):
                    return

                agent = await agent_loader()
                if agent is None:
                    return

                await self._prepare_state_for_run(state, spec)
                user_input = await self._resolve_user_input(user_input_loader)

                output = await agent.run(
                    user_input,
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
            await self._runtime.update_status_and_notify(
                state.id,
                AgentStateStatus.FAILED,
                result_summary=str(error),
            )
            if spec.emit_child_failed_event:
                await self._emit_event_to_parent(
                    state,
                    SchedulerEventType.CHILD_FAILED,
                    {"reason": str(error)},
                )
            elif spec.emit_error_output:
                await self._emit_output(state, str(error), is_final=state.parent_id is None)
        finally:
            self._runtime.pop_abort_signal(state.id)
            await self._maybe_cleanup_agent(state)

    async def _parent_aborted(
        self,
        state: AgentState,
        spec: AgentRunSpec,
    ) -> bool:
        if not spec.enforce_parent_abort or state.parent_id is None:
            return False

        parent_signal = self._runtime.get_abort_signal(state.parent_id)
        if parent_signal is None or not parent_signal.is_aborted():
            return False

        await self._runtime.update_status_and_notify(
            state.id,
            AgentStateStatus.FAILED,
            result_summary="Parent cancelled",
        )
        return True

    async def _prepare_state_for_run(
        self,
        state: AgentState,
        spec: AgentRunSpec,
    ) -> None:
        if spec.increment_wake_count:
            await self._store.increment_wake_count(state.id)

        if not spec.transition_to_running:
            return

        update_kwargs: dict[str, WakeCondition | None] = {}
        if spec.clear_wake_condition:
            update_kwargs["wake_condition"] = None
        await self._store.update_status(
            state.id,
            AgentStateStatus.RUNNING,
            **update_kwargs,
        )

    async def _resolve_user_input(
        self,
        user_input_loader: Callable[[], Awaitable[UserInput] | UserInput],
    ) -> UserInput:
        user_input = user_input_loader()
        if asyncio.iscoroutine(user_input):
            return await user_input
        return user_input

    def _constant_agent_loader(
        self, agent: Agent
    ) -> Callable[[], Awaitable[Agent]]:
        async def _loader() -> Agent:
            return agent

        return _loader

    def _constant_input_loader(
        self, user_input: UserInput
    ) -> Callable[[], UserInput]:
        def _loader() -> UserInput:
            return user_input

        return _loader

    async def _load_registered_agent(
        self,
        state: AgentState,
        *,
        missing_log: str,
        missing_message: str | None = None,
    ) -> Agent | None:
        agent = self._runtime.get_registered_agent(state.id)
        if agent is not None:
            return agent

        logger.error(missing_log, state_id=state.id)
        if missing_message is not None:
            await self._runtime.update_status_and_notify(
                state.id,
                AgentStateStatus.FAILED,
                result_summary=missing_message,
            )
            await self._emit_output(
                state,
                missing_message,
                is_final=state.parent_id is None,
            )
        return None

    async def _build_timeout_wake_message(self, state: AgentState) -> str:
        succeeded, failed = await self._collect_child_results(state)
        wc = state.wake_condition
        total = len(wc.wait_for) if wc else 0
        done = len(succeeded) + len(failed)
        return format_child_results_summary(
            header="Wait timeout reached.",
            succeeded=succeeded,
            failed=failed,
            progress_line=f"Completed children: {done}/{total}",
            closing_instruction="Please produce a summary report with whatever results are available.",
        )

    async def _handle_agent_output(self, state: AgentState, output: RunOutput) -> None:
        """Handle agent output: SLEEPING, persistent idle, PERIODIC reschedule, or COMPLETED.

        Each branch emits text to the output channel when the agent produced content.
        Also generates pending events for parent agents when appropriate.
        """
        text = output.response

        if output.termination_reason == TerminationReason.SLEEPING:
            await self._emit_output(state, text, is_final=False)
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
                await self._store.update_status(
                    state.id,
                    AgentStateStatus.SLEEPING,
                    wake_condition=new_wc,
                    result_summary=text,
                )
                await self._emit_output(state, text, is_final=False)
                # Notify parent about periodic child result
                await self._emit_event_to_parent(
                    state,
                    SchedulerEventType.CHILD_SLEEP_RESULT,
                    {"result": text or "", "explain": state.explain, "periodic": True},
                )
                return

        is_root = state.parent_id is None

        if is_persistent:
            await self._runtime.update_status_and_notify(
                state.id,
                AgentStateStatus.SLEEPING,
                wake_condition=WakeCondition(type=WakeType.TASK_SUBMITTED),
                result_summary=text,
            )
            await self._emit_output(state, text, is_final=is_root)
            return

        await self._runtime.update_status_and_notify(
            state.id, AgentStateStatus.COMPLETED, result_summary=text
        )
        await self._emit_output(state, text, is_final=is_root)
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
        if state.parent_id is None:
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

    async def _emit_output(
        self, state: AgentState, text: str | None, *, is_final: bool
    ) -> None:
        """Push agent text output to the root agent's output channel.

        For child agents, the output is routed to the root's channel.
        If ``include_child_outputs`` is False on the channel, child outputs are skipped.
        When ``is_final`` is True and there is no text, a ``None`` sentinel is pushed
        so that consumers unblock.
        Child agent output is prefixed with a notice tag indicating the source.
        """
        root_id = state.id if state.parent_id is None else state.parent_id
        channel_state = self._runtime.get_output_channel(root_id)
        if channel_state is None:
            return

        is_child = state.parent_id is not None
        if is_child and not channel_state.include_child_outputs:
            return

        # Prefix child output with source notice
        if is_child and text:
            refreshed = await self._store.get_state(state.id)
            status_val = refreshed.status.value if refreshed else "unknown"
            text = f"<notice>agent_id={state.id}, status={status_val}</notice>\n{text}"

        if text:
            await channel_state.queue.put(
                SchedulerOutput(state_id=state.id, text=text, is_final=is_final)
            )
        elif is_final:
            await channel_state.queue.put(None)

    async def _maybe_cleanup_agent(self, state: AgentState) -> None:
        """Remove agent from registry only when truly done (COMPLETED or FAILED, non-root only).

        SLEEPING agents must remain in the registry so wake_agent() can find them.
        When done, also remove from dispatched set so no stale entries remain.
        """
        self._runtime.release_state_dispatch(state.id)
        if state.parent_id is None:
            return
        refreshed = await self._store.get_state(state.id)
        if refreshed is None or refreshed.status in (
            AgentStateStatus.COMPLETED,
            AgentStateStatus.FAILED,
        ):
            self._runtime.unregister_agent(state.id)

    async def _build_wake_message(self, state: AgentState) -> UserInput:
        """Build a wake message with auto-injected child results."""
        wc = state.wake_condition
        if wc is None:
            return "You have been woken up. Please continue your task."

        if wc.type == WakeType.WAITSET:
            succeeded, failed = await self._collect_child_results(state)
            total = len(wc.wait_for)
            done = len(succeeded) + len(failed)
            return format_child_results_summary(
                header=f"Child agents completed ({done}/{total}).",
                succeeded=succeeded,
                failed=failed,
                closing_instruction="Please synthesize a final response based on the successful results above.",
            )

        if wc.type == WakeType.TIMER:
            return "The scheduled delay has elapsed. Please continue your task."

        if wc.type == WakeType.PERIODIC:
            return (
                "A scheduled periodic check has triggered. "
                "Please check progress and decide whether to continue waiting "
                "or produce a final result."
            )

        if wc.type == WakeType.TASK_SUBMITTED:
            return normalize_to_message(wc.submitted_task or "")

        return "You have been woken up. Please continue your task."

    def _build_events_wake_message(self, events: list[PendingEvent]) -> str:
        """Build a wake message from a list of pending events."""
        lines = [f"You have {len(events)} new notification(s):\n"]
        for event in events:
            event_label = event.event_type.value.replace("_", " ").title()
            child_id = event.payload.get(
                "child_agent_id", event.source_agent_id or "unknown"
            )
            lines.append(f"### {event_label} — Agent: {child_id}")
            if event.event_type == SchedulerEventType.CHILD_SLEEP_RESULT:
                result = event.payload.get("result", "")
                explain = event.payload.get("explain")
                periodic = event.payload.get("periodic", False)
                lines.extend(
                    build_child_result_detail_lines(
                        result=result,
                        explain=explain,
                        periodic=periodic,
                        result_as_block=True,
                    )
                )
            elif event.event_type == SchedulerEventType.CHILD_COMPLETED:
                result = event.payload.get("result", "")
                lines.extend(
                    build_child_result_detail_lines(
                        result=result,
                        result_as_block=True,
                    )
                )
            elif event.event_type == SchedulerEventType.CHILD_FAILED:
                reason = event.payload.get("reason", "Unknown failure")
                lines.extend(build_child_result_detail_lines(failure_reason=reason))
            elif event.event_type == SchedulerEventType.HEALTH_WARNING:
                lines.append(
                    event.payload.get("message", "Agent may be stuck or unhealthy")
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

    async def _collect_child_results(
        self, state: AgentState
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Collect results from child agents, separated into succeeded and failed.

        Returns:
            (succeeded, failed) — dicts of {child_id: summary/reason}
        """
        wc = state.wake_condition
        child_ids = wc.wait_for if wc else []
        if not child_ids:
            children = await self._store.get_states_by_parent(state.id)
            child_ids = [c.id for c in children]

        succeeded: dict[str, str] = {}
        failed: dict[str, str] = {}
        for cid in child_ids:
            child = await self._store.get_state(cid)
            if child is None:
                failed[cid] = "Agent state not found"
            elif child.status == AgentStateStatus.FAILED:
                failed[cid] = child.result_summary or "Unknown failure"
            else:
                succeeded[cid] = child.result_summary or f"status={child.status.value}"
        return succeeded, failed
