"""Stream channel lifecycle helpers for the scheduler."""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from agiwo.agent import AgentStreamItem


@dataclass
class StreamChannelState:
    queue: asyncio.Queue
    include_child_events: bool = True


async def finish_stream_channel(
    channels: dict[str, StreamChannelState], state_id: str
) -> None:
    channel = channels.get(state_id)
    if channel is not None:
        await channel.queue.put(None)


def open_stream_channel(
    channels: dict[str, StreamChannelState],
    state_id: str,
    *,
    include_child_events: bool,
) -> None:
    if state_id in channels:
        raise RuntimeError(f"stream subscriber already active for root '{state_id}'")
    channels[state_id] = StreamChannelState(
        queue=asyncio.Queue(),
        include_child_events=include_child_events,
    )


def close_stream_channel(
    channels: dict[str, StreamChannelState], state_id: str
) -> None:
    channels.pop(state_id, None)


async def consume_stream_channel(
    channels: dict[str, StreamChannelState],
    state_id: str,
    *,
    timeout: float | None,
) -> AsyncIterator[AgentStreamItem]:
    start = time.monotonic()
    while True:
        remaining = stream_remaining(timeout, start)
        if timeout is not None and remaining == 0:
            return

        try:
            item = await asyncio.wait_for(
                channels[state_id].queue.get(),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            return

        if item is None:
            return
        yield item
        if item.depth == 0 and item.type in {"run_completed", "run_failed"}:
            return


def stream_remaining(timeout: float | None, start: float) -> float | None:
    if timeout is None:
        return None
    elapsed = time.monotonic() - start
    if elapsed >= timeout:
        return 0
    return timeout - elapsed


__all__ = [
    "StreamChannelState",
    "close_stream_channel",
    "consume_stream_channel",
    "finish_stream_channel",
    "open_stream_channel",
    "stream_remaining",
]
