"""
ConsentWaiter - Authorization wait coordinator.

Coordinates waiting for user consent decisions using asyncio.Event.
"""

import asyncio
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class ConsentDecision(BaseModel):
    """User consent decision"""

    decision: Literal["allow", "deny"]
    patterns: list[str] = []  # User-selected patterns
    expires_at: datetime | None = None


class ConsentWaiter:
    """
    Coordinator for waiting user consent decisions.

    Uses asyncio.Event to suspend execution until user makes a decision.
    """

    def __init__(self, default_timeout: float = 300.0) -> None:
        """
        Initialize consent waiter.

        Args:
            default_timeout: Default timeout in seconds (default: 5 minutes)
        """
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
        """
        Wait for user consent decision.

        Args:
            tool_call_id: Tool call unique identifier
            timeout: Timeout in seconds (None uses default)

        Returns:
            ConsentDecision: User's decision

        Raises:
            asyncio.TimeoutError: If timeout occurs
        """
        timeout = timeout or self._default_timeout

        async with self._lock:
            # Check if decision already exists (idempotent)
            if tool_call_id in self._decisions:
                return self._decisions[tool_call_id]

            # Create Event
            event = asyncio.Event()
            self._events[tool_call_id] = event
            self._timeouts[tool_call_id] = timeout

        try:
            # Wait for decision or timeout
            await asyncio.wait_for(event.wait(), timeout=timeout)

            async with self._lock:
                decision = self._decisions.get(tool_call_id)
                if decision is None:
                    # Timeout occurred, create deny decision
                    decision = ConsentDecision(decision="deny", patterns=[])
                    self._decisions[tool_call_id] = decision
                return decision

        except asyncio.TimeoutError:
            # Handle timeout
            async with self._lock:
                decision = ConsentDecision(decision="deny", patterns=[])
                self._decisions[tool_call_id] = decision
                # Clean up Event
                self._events.pop(tool_call_id, None)
                self._timeouts.pop(tool_call_id, None)

            logger.warning(
                "consent_wait_timeout",
                tool_call_id=tool_call_id,
                timeout=timeout,
            )
            raise

        finally:
            # Clean up resources
            async with self._lock:
                self._events.pop(tool_call_id, None)
                self._timeouts.pop(tool_call_id, None)

    async def resolve(
        self,
        tool_call_id: str,
        decision: ConsentDecision,
    ) -> None:
        """
        Resolve consent decision and wake up waiting tasks.

        Args:
            tool_call_id: Tool call unique identifier
            decision: User's consent decision

        Note: Repeated resolves are idempotent (uses last decision)
        """
        async with self._lock:
            # Save decision
            self._decisions[tool_call_id] = decision

            # Wake up waiting tasks
            event = self._events.get(tool_call_id)
            if event:
                event.set()

        logger.info(
            "consent_resolved",
            tool_call_id=tool_call_id,
            decision=decision.decision,
        )

    async def cancel(self, tool_call_id: str) -> None:
        """
        Cancel waiting (for execution interruption).

        Args:
            tool_call_id: Tool call unique identifier
        """
        async with self._lock:
            if tool_call_id in self._events:
                self._events.pop(tool_call_id, None)
                self._timeouts.pop(tool_call_id, None)
                # Create deny decision
                self._decisions[tool_call_id] = ConsentDecision(
                    decision="deny", patterns=[]
                )

        logger.info("consent_cancelled", tool_call_id=tool_call_id)


__all__ = ["ConsentWaiter", "ConsentDecision"]
