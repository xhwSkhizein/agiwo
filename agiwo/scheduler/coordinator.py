"""Pure in-memory runtime coordination for the scheduler."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from agiwo.agent.execution import AgentExecutionHandlePort
from agiwo.agent.runtime import AgentStreamItem
from agiwo.agent.scheduler_port import SchedulerAgentPort
from agiwo.utils.abort_signal import AbortSignal


@dataclass
class StreamChannelState:
    queue: asyncio.Queue
    include_child_events: bool = True


class SchedulerCoordinator:
    """Own only process-memory runtime state for scheduler execution."""

    def __init__(self) -> None:
        self._active_tasks: set[asyncio.Task] = set()
        self._agents: dict[str, SchedulerAgentPort] = {}
        self._execution_handles: dict[str, AgentExecutionHandlePort] = {}
        self._abort_signals: dict[str, AbortSignal] = {}
        self._state_events: dict[str, asyncio.Event] = {}
        self._dispatched_state_ids: set[str] = set()
        self._stream_channels: dict[str, StreamChannelState] = {}

    @property
    def active_tasks(self) -> set[asyncio.Task]:
        return self._active_tasks

    @property
    def dispatched_state_ids(self) -> set[str]:
        return self._dispatched_state_ids

    def register_agent(self, agent: SchedulerAgentPort) -> None:
        self._agents[agent.id] = agent

    def unregister_agent(self, state_id: str) -> None:
        self._agents.pop(state_id, None)
        self._execution_handles.pop(state_id, None)

    def get_registered_agent(self, state_id: str) -> SchedulerAgentPort | None:
        return self._agents.get(state_id)

    def set_execution_handle(
        self,
        state_id: str,
        handle: AgentExecutionHandlePort,
    ) -> None:
        self._execution_handles[state_id] = handle

    def get_execution_handle(self, state_id: str) -> AgentExecutionHandlePort | None:
        return self._execution_handles.get(state_id)

    def pop_execution_handle(self, state_id: str) -> AgentExecutionHandlePort | None:
        return self._execution_handles.pop(state_id, None)

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
        state_id: str,
        run_factory: Callable[[], Awaitable[None]],
    ) -> bool:
        if not self.reserve_state_dispatch(state_id):
            return False
        self.track_active_task(asyncio.create_task(run_factory()))
        return True

    def open_stream_channel(
        self,
        state_id: str,
        *,
        include_child_events: bool,
    ) -> StreamChannelState:
        channel_state = StreamChannelState(
            queue=asyncio.Queue(),
            include_child_events=include_child_events,
        )
        self._stream_channels[state_id] = channel_state
        return channel_state

    def get_stream_channel(self, state_id: str) -> StreamChannelState | None:
        return self._stream_channels.get(state_id)

    def close_stream_channel(self, state_id: str) -> None:
        self._stream_channels.pop(state_id, None)

    async def finish_stream_channel(self, state_id: str) -> None:
        channel = self._stream_channels.get(state_id)
        if channel is not None:
            await channel.queue.put(None)

    async def consume_stream_channel(
        self,
        state_id: str,
        timeout: float | None,
    ) -> AsyncIterator[AgentStreamItem]:
        channel = self._stream_channels.get(state_id)
        if channel is None:
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
                item = await asyncio.wait_for(channel.queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return

            if item is None:
                return
            yield item
            if item.depth == 0 and item.type in {"run_completed", "run_failed"}:
                return


__all__ = ["SchedulerCoordinator"]
