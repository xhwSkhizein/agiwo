"""Stream construction and routing extracted from Scheduler."""

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING

from agiwo.agent import AgentStreamItem
from agiwo.scheduler.commands import RouteResult
from agiwo.scheduler.models import AgentStateStatus
from agiwo.scheduler.stream import (
    close_stream_channel,
    consume_stream_channel,
    open_stream_channel,
)

if TYPE_CHECKING:
    from agiwo.scheduler.engine import Scheduler


def build_stream(
    sched: "Scheduler",
    state_id: str,
    *,
    timeout: float | None,
    include_child_events: bool,
    close_on_root_run_end: bool,
) -> AsyncIterator[AgentStreamItem]:
    async def iterator() -> AsyncIterator[AgentStreamItem]:
        try:
            async for item in consume_stream_channel(
                sched._rt.stream_channels,
                state_id,
                timeout=timeout,
            ):
                yield item
        finally:
            await raise_stream_failure_if_needed(sched, state_id)
            close_stream_channel(sched._rt.stream_channels, state_id)

    return iterator()


async def route_with_stream(
    sched: "Scheduler",
    *,
    root_state_id: str,
    action: str,
    timeout: float | None,
    include_child_events: bool,
    close_on_root_run_end: bool,
    operation: Callable[[], Awaitable[str]],
) -> RouteResult:
    if root_state_id in sched._rt.stream_channels:
        raise RuntimeError(
            f"stream subscriber already active for root '{root_state_id}'"
        )
    open_stream_channel(
        sched._rt.stream_channels,
        root_state_id,
        include_child_events=include_child_events,
        close_on_root_run_end=close_on_root_run_end,
    )
    try:
        state_id = await operation()
    except Exception:
        close_stream_channel(sched._rt.stream_channels, root_state_id)
        raise
    return RouteResult(
        action=action,
        state_id=state_id,
        stream=build_stream(
            sched,
            root_state_id,
            timeout=timeout,
            include_child_events=include_child_events,
            close_on_root_run_end=close_on_root_run_end,
        ),
    )


async def raise_stream_failure_if_needed(sched: "Scheduler", state_id: str) -> None:
    state = await sched._store.get_state(state_id)
    if state is not None and state.status == AgentStateStatus.FAILED:
        raise RuntimeError(state.result_summary or "scheduler stream failed")
