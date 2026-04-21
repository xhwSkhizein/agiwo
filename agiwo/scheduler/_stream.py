"""Stream construction and routing extracted from Scheduler."""

from collections.abc import AsyncIterator, Awaitable, Callable

from agiwo.agent import AgentStreamItem
from agiwo.scheduler.commands import RouteResult
from agiwo.scheduler.engine_context import EngineContext
from agiwo.scheduler.models import AgentStateStatus
from agiwo.scheduler.stream import (
    close_stream_channel,
    consume_stream_channel,
    open_stream_channel,
)


def build_stream(
    ctx: EngineContext,
    state_id: str,
    *,
    timeout: float | None,
) -> AsyncIterator[AgentStreamItem]:
    async def iterator() -> AsyncIterator[AgentStreamItem]:
        try:
            async for item in consume_stream_channel(
                ctx.rt.stream_channels,
                state_id,
                timeout=timeout,
            ):
                yield item
        finally:
            await raise_stream_failure_if_needed(ctx, state_id)
            close_stream_channel(ctx.rt.stream_channels, state_id)

    return iterator()


async def route_with_stream(
    ctx: EngineContext,
    *,
    root_state_id: str,
    action: str,
    timeout: float | None,
    include_child_events: bool,
    close_on_root_run_end: bool,
    operation: Callable[[], Awaitable[str]],
) -> RouteResult:
    if root_state_id in ctx.rt.stream_channels:
        raise RuntimeError(
            f"stream subscriber already active for root '{root_state_id}'"
        )
    open_stream_channel(
        ctx.rt.stream_channels,
        root_state_id,
        include_child_events=include_child_events,
        close_on_root_run_end=close_on_root_run_end,
    )
    try:
        state_id = await operation()
    except Exception:
        close_stream_channel(ctx.rt.stream_channels, root_state_id)
        raise
    return RouteResult(
        action=action,
        state_id=state_id,
        stream=build_stream(
            ctx,
            root_state_id,
            timeout=timeout,
        ),
    )


async def raise_stream_failure_if_needed(ctx: EngineContext, state_id: str) -> None:
    state = await ctx.store.get_state(state_id)
    if state is not None and state.status == AgentStateStatus.FAILED:
        raise RuntimeError(state.result_summary or "scheduler stream failed")
