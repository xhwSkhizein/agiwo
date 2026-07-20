"""SSE event stream for a single Objective (P5-02).

``stream_objective_events`` replays committed timeline facts after a cursor
and then live-follows new commits via ``ObjectiveStore.commit_notifier``.
Callers own reconnection: resolve ``Last-Event-ID`` / ``after_sequence`` into
a single starting cursor before calling this generator.
"""

import asyncio
from collections.abc import AsyncIterator

from agiwo.objective import ObjectiveStore
from agiwo.objective.projection import project_objective, project_timeline_page
from agiwo.utils.logging import get_logger

from server.services.objective_serialization import sse_message_from_node

logger = get_logger(__name__)

DEFAULT_MAX_PENDING = 500
DEFAULT_WAIT_TIMEOUT_SECONDS = 15.0


async def stream_objective_events(
    store: ObjectiveStore,
    objective_id: str,
    after_sequence: int = 0,
    *,
    max_pending: int = DEFAULT_MAX_PENDING,
    wait_timeout: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
) -> AsyncIterator[dict[str, str]]:
    """Yield SSE message dicts for facts with ``sequence > after_sequence``.

    Replays existing history first, then blocks on new commits until the
    Objective reaches a terminal state and every fact has been replayed, or
    the client falls too far behind (``max_pending``), in which case the
    generator stops so the caller can force a fresh reconnect/resync.
    """
    cursor = after_sequence
    while True:
        facts = await store.list_facts(objective_id=objective_id)
        if not facts:
            logger.info("objective_sse_not_found", objective_id=objective_id)
            return

        pending = project_timeline_page(
            facts,
            after_sequence=cursor,
            limit=max_pending + 1,
        )
        overflow = len(pending) > max_pending
        if overflow:
            pending = pending[:max_pending]

        for node in pending:
            yield sse_message_from_node(node)
            cursor = node.sequence

        if overflow:
            logger.warning(
                "objective_sse_client_lag_exceeded",
                objective_id=objective_id,
                max_pending=max_pending,
            )
            return

        view = project_objective(facts, objective_id=objective_id)
        if view is not None and view.is_terminal and cursor >= view.last_sequence:
            return

        await store.commit_notifier.wait(objective_id, timeout=wait_timeout)


async def bounded(
    stream: AsyncIterator[dict[str, str]],
    timeout_seconds: float,
) -> AsyncIterator[dict[str, str]]:
    """Consume ``stream`` until it ends or ``timeout_seconds`` elapses in total."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_seconds
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                item = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
            except asyncio.TimeoutError:
                return
            except StopAsyncIteration:
                return
            yield item
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


__all__ = ["bounded", "stream_objective_events"]
