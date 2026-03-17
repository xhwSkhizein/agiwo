import asyncio
from collections.abc import AsyncIterator

from agiwo.agent.execution import AgentExecutionHandlePort
from agiwo.agent.runtime import AgentStreamItem


async def consume_execution_stream(
    handle: AgentExecutionHandlePort,
    *,
    cancel_reason: str,
) -> AsyncIterator[AgentStreamItem]:
    completed = False
    try:
        async for item in handle.stream():
            yield item
        await handle.wait()
        completed = True
    finally:
        if not completed:
            handle.cancel(cancel_reason)
            try:
                await handle.wait()
            except asyncio.CancelledError:
                pass


__all__ = ["consume_execution_stream"]
