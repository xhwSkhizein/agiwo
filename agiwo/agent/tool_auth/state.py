import asyncio

from agiwo.tool.authz import ConsentDecision
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class ConsentWaiter:
    """Coordinator for waiting user consent decisions."""

    def __init__(self, default_timeout: float = 300.0) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._decisions: dict[str, ConsentDecision] = {}
        self._timeouts: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._default_timeout = default_timeout

    async def wait_for_consent(
        self,
        tool_call_id: str,
        timeout: float | None = None,
    ) -> ConsentDecision:
        wait_timeout = timeout or self._default_timeout
        async with self._lock:
            if tool_call_id in self._decisions:
                return self._decisions[tool_call_id]
            event = asyncio.Event()
            self._events[tool_call_id] = event
            self._timeouts[tool_call_id] = wait_timeout

        try:
            await asyncio.wait_for(event.wait(), timeout=wait_timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                self._decisions[tool_call_id] = ConsentDecision(
                    decision="deny",
                    patterns=[],
                )
            logger.warning(
                "consent_wait_timeout",
                tool_call_id=tool_call_id,
                timeout=wait_timeout,
            )
            raise
        finally:
            async with self._lock:
                self._events.pop(tool_call_id, None)
                self._timeouts.pop(tool_call_id, None)

        async with self._lock:
            return self._decisions.setdefault(
                tool_call_id,
                ConsentDecision(decision="deny", patterns=[]),
            )

    async def resolve(self, tool_call_id: str, decision: ConsentDecision) -> None:
        async with self._lock:
            self._decisions[tool_call_id] = decision
            event = self._events.get(tool_call_id)
            if event is not None:
                event.set()

    async def cancel(self, tool_call_id: str) -> None:
        async with self._lock:
            self._events.pop(tool_call_id, None)
            self._timeouts.pop(tool_call_id, None)
            self._decisions[tool_call_id] = ConsentDecision(
                decision="deny",
                patterns=[],
            )
