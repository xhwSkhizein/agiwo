"""
SchedulerExecutor — Agent execution logic for the Scheduler.

Handles running, waking, and output processing for scheduled agents.
Extracted from scheduler.py to reduce file size and improve maintainability.
"""

import asyncio
import copy
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
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
    OutputChannelState,
    PendingEvent,
    SchedulerEventType,
    SchedulerOutput,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store import AgentStateStorage
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

if TYPE_CHECKING:
    from agiwo.scheduler.scheduler import Scheduler

logger = get_logger(__name__)


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
        scheduler: "Scheduler",
        store: AgentStateStorage,
        agents: dict[str, Agent],
        abort_signals: dict[str, AbortSignal],
        state_events: dict[str, asyncio.Event],
        semaphore: asyncio.Semaphore,
        dispatched_state_ids: set[str],
        output_channels: dict[str, OutputChannelState],
    ) -> None:
        self._scheduler = scheduler
        self._store = store
        self._agents = agents
        self._abort_signals = abort_signals
        self._state_events = state_events
        self._semaphore = semaphore
        self._dispatched_state_ids = dispatched_state_ids
        self._output_channels = output_channels

    async def create_child_agent(self, state: AgentState) -> Agent:
        """Create a child Agent by copying the parent's configuration."""
        parent = self._agents.get(state.parent_id)
        if parent is None:
            raise RuntimeError(
                f"Parent agent '{state.parent_id}' not found in scheduler"
            )

        overrides = state.config_overrides or {}
        child_options = copy.deepcopy(parent.options) if parent.options else None
        if child_options is not None:
            child_options.enable_termination_summary = True

        # Get parent's fully built system prompt
        parent_system_prompt = await parent.get_effective_system_prompt()

        # Instruction is a task-specific directive — append to system prompt (static per child)
        instruction = overrides.get("instruction")
        if instruction:
            parent_system_prompt += (
                f"\n\n<task-instruction>\n{instruction}\n</task-instruction>"
            )

        child = Agent(
            name=parent.name,
            description=parent.description,
            model=parent.model,
            id=state.id,
            tools=[t for t in parent.tools if t.get_name() != "spawn_agent"],
            system_prompt=parent_system_prompt,
            options=child_options,
            hooks=parent.hooks,
        )
        self._agents[child.id] = child
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
        except Exception:
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
            await self._update_status_and_notify(
                state.id, AgentStateStatus.FAILED, result_summary=str(e)
            )
            await self._emit_output(state, str(e), is_final=True)
        finally:
            self._abort_signals.pop(state.id, None)
            await self._maybe_cleanup_agent(state)

    async def run_agent(self, state: AgentState) -> None:
        """Run a PENDING child agent."""
        abort_signal = AbortSignal()
        parent_signal = self._abort_signals.get(state.parent_id or "")
        self._abort_signals[state.id] = abort_signal

        async with self._semaphore:
            if parent_signal is not None and parent_signal.is_aborted():
                await self._update_status_and_notify(
                    state.id, AgentStateStatus.FAILED, result_summary="Parent cancelled"
                )
                return

            await self._store.update_status(state.id, AgentStateStatus.RUNNING)
            try:
                child = await self.create_child_agent(state)
                child_session_id = state.id

                output = await child.run(
                    state.task, session_id=child_session_id, abort_signal=abort_signal
                )
                await self._handle_agent_output(state, output)
            except Exception as e:
                logger.exception(
                    "child_agent_failed",
                    state_id=state.id,
                    parent_id=state.parent_id,
                    depth=state.depth,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await self._update_status_and_notify(
                    state.id, AgentStateStatus.FAILED, result_summary=str(e)
                )
                await self._emit_event_to_parent(
                    state,
                    SchedulerEventType.CHILD_FAILED,
                    {"reason": str(e)},
                )
            finally:
                self._abort_signals.pop(state.id, None)
                await self._maybe_cleanup_agent(state)

    async def wake_agent(self, state: AgentState) -> None:
        """Wake a SLEEPING agent by running it with a wake message."""
        agent = self._agents.get(state.id)
        if agent is None:
            logger.error("wake_agent_not_found", state_id=state.id)
            error_msg = f"Agent '{state.id}' not found in scheduler for wake"
            await self._update_status_and_notify(
                state.id,
                AgentStateStatus.FAILED,
                result_summary=error_msg,
            )
            await self._emit_output(state, error_msg, is_final=state.parent_id is None)
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
                    state_id=state.id,
                    wake_count=state.wake_count,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await self._update_status_and_notify(
                    state.id, AgentStateStatus.FAILED, result_summary=str(e)
                )
                await self._emit_output(state, str(e), is_final=state.parent_id is None)
            finally:
                self._abort_signals.pop(state.id, None)
                await self._maybe_cleanup_agent(state)

    async def wake_agent_for_events(
        self, state: AgentState, events: list[PendingEvent]
    ) -> None:
        """Wake a SLEEPING agent to process accumulated pending events."""
        agent = self._agents.get(state.id)
        if agent is None:
            logger.error("wake_agent_for_events_not_found", state_id=state.id)
            return

        await self._store.increment_wake_count(state.id)
        abort_signal = self._abort_signals.get(state.id)
        async with self._semaphore:
            await self._store.update_status(state.id, AgentStateStatus.RUNNING)
            try:
                wake_message = self._build_events_wake_message(events)
                output = await agent.run(
                    wake_message, session_id=state.session_id, abort_signal=abort_signal
                )
                await self._handle_agent_output(state, output)
            except Exception as e:
                logger.exception(
                    "wake_agent_for_events_failed",
                    state_id=state.id,
                    event_count=len(events),
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await self._update_status_and_notify(
                    state.id, AgentStateStatus.FAILED, result_summary=str(e)
                )
                await self._emit_output(state, str(e), is_final=state.parent_id is None)
            finally:
                self._abort_signals.pop(state.id, None)
                await self._maybe_cleanup_agent(state)

    async def wake_for_timeout(self, state: AgentState) -> None:
        """Wake a timed-out SLEEPING agent so it can produce a summary report."""
        agent = self._agents.get(state.id)
        if agent is None:
            logger.error("timeout_wake_agent_not_found", state_id=state.id)
            error_msg = f"Agent '{state.id}' not found for timeout wake"
            await self._update_status_and_notify(
                state.id,
                AgentStateStatus.FAILED,
                result_summary=error_msg,
            )
            await self._emit_output(state, error_msg, is_final=state.parent_id is None)
            return

        succeeded, failed = await self._collect_child_results(state)
        wc = state.wake_condition
        total = len(wc.wait_for) if wc else 0
        done = len(succeeded) + len(failed)

        wake_msg = (
            f"Wait timeout reached.\n"
            f"Completed children: {done}/{total}\n\n"
        )
        if succeeded:
            wake_msg += "## Successful Results\n"
            for cid, summary in succeeded.items():
                wake_msg += f"- [{cid}] {summary}\n"
            wake_msg += "\n"
        if failed:
            wake_msg += "## Failed Agents\n"
            for cid, reason in failed.items():
                wake_msg += f"- [{cid}] FAILED: {reason}\n"
            wake_msg += "\n"
        wake_msg += "Please produce a summary report with whatever results are available."

        await self._store.increment_wake_count(state.id)
        abort_signal = self._abort_signals.get(state.id)
        async with self._semaphore:
            await self._store.update_status(state.id, AgentStateStatus.RUNNING, wake_condition=None)
            try:
                output = await agent.run(
                    wake_msg, session_id=state.session_id, abort_signal=abort_signal
                )
                await self._handle_agent_output(state, output)
            except Exception as e:
                logger.exception(
                    "timeout_wake_failed",
                    state_id=state.id,
                    error=str(e),
                )
                await self._update_status_and_notify(
                    state.id, AgentStateStatus.FAILED, result_summary=str(e)
                )
                await self._emit_output(state, str(e), is_final=state.parent_id is None)

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
            await self._update_status_and_notify(
                state.id,
                AgentStateStatus.SLEEPING,
                wake_condition=WakeCondition(type=WakeType.TASK_SUBMITTED),
                result_summary=text,
            )
            await self._emit_output(state, text, is_final=is_root)
            return

        await self._update_status_and_notify(
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
        channel_state = self._output_channels.get(root_id)
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

    def _close_output_channel(self, root_state_id: str) -> None:
        """Remove the output channel for a root agent (called by Scheduler after consume)."""
        self._output_channels.pop(root_state_id, None)

    async def _maybe_cleanup_agent(self, state: AgentState) -> None:
        """Remove agent from registry only when truly done (COMPLETED or FAILED, non-root only).

        SLEEPING agents must remain in the registry so wake_agent() can find them.
        When done, also remove from dispatched set so no stale entries remain.
        """
        self._dispatched_state_ids.discard(state.id)
        if state.parent_id is None:
            return
        refreshed = await self._store.get_state(state.id)
        if refreshed is None or refreshed.status in (
            AgentStateStatus.COMPLETED,
            AgentStateStatus.FAILED,
        ):
            self._agents.pop(state.id, None)

    def _notify_state_change(self, state_id: str) -> None:
        """Notify waiters that a state has changed."""
        event = self._state_events.get(state_id)
        if event is not None:
            event.set()

    async def _update_status_and_notify(
        self,
        state_id: str,
        status: AgentStateStatus,
        *,
        result_summary: str | None = ...,
        wake_condition: WakeCondition | None = ...,
    ) -> None:
        """Update state status and notify waiters if it's a terminal state."""
        await self._store.update_status(
            state_id, status, result_summary=result_summary, wake_condition=wake_condition
        )
        if status in (AgentStateStatus.COMPLETED, AgentStateStatus.FAILED):
            self._notify_state_change(state_id)
        elif status == AgentStateStatus.SLEEPING and wake_condition is not ...:
            if (
                wake_condition is not None
                and wake_condition.type == WakeType.TASK_SUBMITTED
            ):
                self._notify_state_change(state_id)

    async def _build_wake_message(self, state: AgentState) -> UserInput:
        """Build a wake message with auto-injected child results."""
        wc = state.wake_condition
        if wc is None:
            return "You have been woken up. Please continue your task."

        if wc.type == WakeType.WAITSET:
            succeeded, failed = await self._collect_child_results(state)
            total = len(wc.wait_for)
            done = len(succeeded) + len(failed)
            msg = f"Child agents completed ({done}/{total}).\n\n"
            if succeeded:
                msg += "## Successful Results\n"
                for cid, summary in succeeded.items():
                    msg += f"- [{cid}] {summary}\n"
                msg += "\n"
            if failed:
                msg += "## Failed Agents\n"
                for cid, reason in failed.items():
                    msg += f"- [{cid}] FAILED: {reason}\n"
                msg += "\n"
            msg += "Please synthesize a final response based on the successful results above."
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
                if periodic:
                    lines.append("(Periodic check completed)")
                if explain:
                    lines.append(f"Sleep reason: {explain}")
                if result:
                    lines.append(f"Result:\n{result}")
            elif event.event_type == SchedulerEventType.CHILD_COMPLETED:
                result = event.payload.get("result", "")
                if result:
                    lines.append(f"Result:\n{result}")
            elif event.event_type == SchedulerEventType.CHILD_FAILED:
                reason = event.payload.get("reason", "Unknown failure")
                lines.append(f"Failure reason: {reason}")
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
