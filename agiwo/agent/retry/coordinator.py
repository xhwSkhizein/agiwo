"""Agent-owned retry coordinator at model/tool attempt boundaries."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agiwo.agent.retry.faults import (
    ExecutionFault,
    FaultDisposition,
    IdempotencyKind,
    RunBlockingFaultError,
    may_auto_retry,
)

Sleeper = Callable[[float], Awaitable[None]]
ContinueGate = Callable[[], Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    min_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 10.0


async def _default_sleeper(seconds: float) -> None:
    await asyncio.sleep(seconds)


async def _always_continue() -> bool:
    return True


class RetryCoordinator:
    """Decide and pace safe automatic retries."""

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        *,
        sleeper: Sleeper | None = None,
        should_continue: ContinueGate | None = None,
    ) -> None:
        self.policy = policy or RetryPolicy()
        self._sleeper = sleeper or _default_sleeper
        self._should_continue = should_continue or _always_continue

    def can_retry(
        self,
        fault: ExecutionFault,
        *,
        attempt_no: int,
        idempotency: IdempotencyKind,
        idempotency_key: str | None = None,
    ) -> bool:
        if attempt_no >= self.policy.max_attempts:
            return False
        return may_auto_retry(
            fault, idempotency=idempotency, idempotency_key=idempotency_key
        )

    async def wait_before_retry(self, attempt_no: int) -> None:
        await self._sleeper(self.backoff_seconds(attempt_no))

    def backoff_seconds(self, attempt_no: int) -> float:
        """Exponential backoff for the given 1-based attempt that just failed."""
        return min(
            self.policy.max_backoff_seconds,
            self.policy.min_backoff_seconds * (2 ** max(0, attempt_no - 1)),
        )

    async def ensure_progress_allowed(self) -> None:
        if not await self._should_continue():
            raise RunBlockingFaultError(
                ExecutionFault(
                    operation="retry_gate",
                    disposition=FaultDisposition.NON_RETRYABLE,
                    run_blocking=True,
                    message="retry blocked by pause or progress gate",
                    provenance={"gate": "should_continue"},
                )
            )

    def raise_boundary(
        self,
        fault: ExecutionFault,
        *,
        attempts: list[ExecutionFault],
        exhausted: bool,
    ) -> None:
        if fault.run_blocking or exhausted:
            raise RunBlockingFaultError(fault, attempts=attempts, exhausted=exhausted)


__all__ = ["RetryCoordinator", "RetryPolicy"]
