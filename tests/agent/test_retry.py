"""P4-01: structured retry disposition and coordinator."""

import pytest

from agiwo.agent.retry import (
    ExecutionFault,
    FaultDisposition,
    IdempotencyKind,
    RetryCoordinator,
    RetryPolicy,
    RunBlockingFaultError,
    finalization_for_blocking_fault,
    map_provider_exception,
    may_auto_retry,
)


def test_may_auto_retry_requires_idempotency() -> None:
    fault = ExecutionFault(
        operation="llm",
        disposition=FaultDisposition.RETRYABLE,
        run_blocking=True,
    )
    assert may_auto_retry(fault, idempotency=IdempotencyKind.GUARANTEED)
    assert not may_auto_retry(fault, idempotency=IdempotencyKind.NOT_IDEMPOTENT)
    assert not may_auto_retry(
        fault, idempotency=IdempotencyKind.CONDITIONAL, idempotency_key=None
    )
    assert may_auto_retry(
        fault, idempotency=IdempotencyKind.CONDITIONAL, idempotency_key="k1"
    )


def test_non_retryable_never_auto_retries() -> None:
    fault = ExecutionFault(
        operation="llm",
        disposition=FaultDisposition.NON_RETRYABLE,
        run_blocking=True,
    )
    assert not may_auto_retry(fault, idempotency=IdempotencyKind.GUARANTEED)


class _AuthError(Exception):
    status_code = 401


class _RateError(Exception):
    status_code = 429


def test_map_provider_uses_status_not_message() -> None:
    auth = map_provider_exception(_AuthError("anything rate limit 429"))
    assert auth.disposition is FaultDisposition.NON_RETRYABLE
    rate = map_provider_exception(_RateError("forbidden looking text"))
    assert rate.disposition is FaultDisposition.RETRYABLE


@pytest.mark.asyncio
async def test_coordinator_backoff_and_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    allowed = True

    async def _gate() -> bool:
        return allowed

    coord = RetryCoordinator(
        RetryPolicy(max_attempts=3, min_backoff_seconds=1.0, max_backoff_seconds=4.0),
        sleeper=_sleep,
        should_continue=_gate,
    )
    fault = ExecutionFault(
        operation="llm",
        disposition=FaultDisposition.RETRYABLE,
        run_blocking=True,
    )
    assert coord.can_retry(fault, attempt_no=1, idempotency=IdempotencyKind.GUARANTEED)
    await coord.wait_before_retry(1)
    assert sleeps == [1.0]
    allowed = False
    with pytest.raises(RunBlockingFaultError):
        await coord.ensure_progress_allowed()


def test_retry_exhausted_finalization_targets_agent() -> None:
    fault = ExecutionFault(
        operation="llm",
        disposition=FaultDisposition.RETRYABLE,
        run_blocking=True,
        message="boom",
    )
    err = RunBlockingFaultError(fault, attempts=[fault, fault], exhausted=True)
    result = finalization_for_blocking_fault(err, carry_forward=[])
    assert result.decision["target"] == "agent"
    assert "system_retry_exhausted" in result.report


def test_outcome_unknown_finalization_targets_user() -> None:
    fault = ExecutionFault(
        operation="tool:pay",
        disposition=FaultDisposition.OUTCOME_UNKNOWN,
        run_blocking=True,
        external_effect_may_have_started=True,
    )
    err = RunBlockingFaultError(fault, attempts=[fault], exhausted=False)
    result = finalization_for_blocking_fault(err)
    assert result.decision["target"] == "user"
    assert result.decision["expects_reply"] is True
    assert "system_outcome_unknown" in result.report
