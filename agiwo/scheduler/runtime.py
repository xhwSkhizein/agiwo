"""Scheduler runtime state and coordination helpers."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from agiwo.agent.agent import Agent
from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    OutputChannelState,
    SchedulerOutput,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store import AgentStateStorage
from agiwo.utils.abort_signal import AbortSignal


class SchedulerRuntime:
    """Own in-memory scheduler runtime state and the public coordination API."""

    def __init__(self, store: AgentStateStorage) -> None:
        self._store = store
        self._active_tasks: set[asyncio.Task] = set()
        self._agents: dict[str, Agent] = {}
        self._abort_signals: dict[str, AbortSignal] = {}
        self._state_events: dict[str, asyncio.Event] = {}
        self._dispatched_state_ids: set[str] = set()
        self._output_channels: dict[str, OutputChannelState] = {}

    @property
    def active_tasks(self) -> set[asyncio.Task]:
        return self._active_tasks

    @property
    def dispatched_state_ids(self) -> set[str]:
        return self._dispatched_state_ids

    def register_agent(self, agent: Agent) -> None:
        self._agents[agent.id] = agent

    def unregister_agent(self, state_id: str) -> None:
        self._agents.pop(state_id, None)

    def get_registered_agent(self, state_id: str) -> Agent | None:
        return self._agents.get(state_id)

    def set_abort_signal(self, state_id: str, signal: AbortSignal) -> None:
        self._abort_signals[state_id] = signal

    def get_abort_signal(self, state_id: str) -> AbortSignal | None:
        return self._abort_signals.get(state_id)

    def pop_abort_signal(self, state_id: str) -> AbortSignal | None:
        return self._abort_signals.pop(state_id, None)

    def get_or_create_state_event(self, state_id: str) -> asyncio.Event:
        return self._state_events.setdefault(state_id, asyncio.Event())

    def pop_state_event(self, state_id: str) -> None:
        self._state_events.pop(state_id, None)

    def notify_state_change(self, state_id: str) -> None:
        event = self._state_events.get(state_id)
        if event is not None:
            event.set()

    def track_active_task(self, task: asyncio.Task) -> None:
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    def reserve_state_dispatch(self, state_id: str) -> bool:
        if state_id in self._dispatched_state_ids:
            return False
        self._dispatched_state_ids.add(state_id)
        return True

    def release_state_dispatch(self, state_id: str) -> None:
        self._dispatched_state_ids.discard(state_id)

    def dispatch_state_task(
        self,
        state: AgentState,
        runner: Callable[[], Awaitable[None]],
    ) -> bool:
        if not self.reserve_state_dispatch(state.id):
            return False
        self.track_active_task(asyncio.create_task(runner()))
        return True

    def open_output_channel(
        self,
        state_id: str,
        *,
        include_child_outputs: bool,
    ) -> OutputChannelState:
        channel_state = OutputChannelState(
            queue=asyncio.Queue(),
            include_child_outputs=include_child_outputs,
        )
        self._output_channels[state_id] = channel_state
        return channel_state

    def get_output_channel(self, state_id: str) -> OutputChannelState | None:
        return self._output_channels.get(state_id)

    def close_output_channel(self, state_id: str) -> None:
        self._output_channels.pop(state_id, None)

    async def consume_output_channel(
        self,
        state_id: str,
        timeout: float | None,
    ) -> AsyncIterator[SchedulerOutput]:
        channel_state = self._output_channels.get(state_id)
        if channel_state is None:
            return

        start = time.time()
        while True:
            remaining = None
            if timeout is not None:
                elapsed = time.time() - start
                if elapsed >= timeout:
                    return
                remaining = timeout - elapsed

            try:
                item = await asyncio.wait_for(
                    channel_state.queue.get(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                return

            if item is None:
                return
            yield item
            if item.is_final:
                return

    async def update_status_and_notify(
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
            self.notify_state_change(state_id)
        elif status == AgentStateStatus.SLEEPING and wake_condition is not ...:
            if wake_condition is not None and wake_condition.type == WakeType.TASK_SUBMITTED:
                self.notify_state_change(state_id)

    async def recursive_cancel(self, state_id: str, reason: str) -> None:
        signal = self.get_abort_signal(state_id)
        if signal is not None:
            signal.abort(reason)

        children = await self._store.get_states_by_parent(state_id)
        for child in children:
            if child.status in (
                AgentStateStatus.RUNNING,
                AgentStateStatus.SLEEPING,
                AgentStateStatus.PENDING,
            ):
                await self.recursive_cancel(child.id, reason)

        await self.update_status_and_notify(
            state_id,
            AgentStateStatus.FAILED,
            result_summary=reason,
        )

    async def recursive_shutdown(self, state_id: str) -> None:
        children = await self._store.get_states_by_parent(state_id)
        for child in children:
            if child.status in (
                AgentStateStatus.RUNNING,
                AgentStateStatus.SLEEPING,
                AgentStateStatus.PENDING,
            ):
                await self.recursive_shutdown(child.id)

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
            await self.update_status_and_notify(
                state_id,
                AgentStateStatus.SLEEPING,
                wake_condition=wake_condition,
            )
        elif state.status == AgentStateStatus.PENDING:
            await self.update_status_and_notify(
                state_id,
                AgentStateStatus.FAILED,
                result_summary="Shutdown before execution",
            )


__all__ = ["SchedulerRuntime"]
