"""Process-local commit wakeups for Objective SSE (not a truth source)."""

import asyncio
from collections import defaultdict


class CommitNotifier:
    """Wake SSE waiters after ObjectiveStore commits. Events must re-read store."""

    def __init__(self) -> None:
        self._waiters: dict[str, set[asyncio.Event]] = defaultdict(set)

    def notify(self, objective_id: str) -> None:
        for event in list(self._waiters.get(objective_id, ())):
            event.set()

    async def wait(
        self,
        objective_id: str,
        *,
        timeout: float | None = 15.0,
    ) -> bool:
        event = asyncio.Event()
        self._waiters[objective_id].add(event)
        try:
            if timeout is None:
                await event.wait()
                return True
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
                return True
            except asyncio.TimeoutError:
                return False
        finally:
            waiters = self._waiters.get(objective_id)
            if waiters is not None:
                waiters.discard(event)
                if not waiters:
                    self._waiters.pop(objective_id, None)


__all__ = ["CommitNotifier"]
