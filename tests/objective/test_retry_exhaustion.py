"""P4-02: retry exhaustion produces system agent handoff finalization."""

from agiwo.agent.retry import (
    ExecutionFault,
    FaultDisposition,
    RunBlockingFaultError,
    finalization_for_blocking_fault,
)


def test_system_report_has_provenance_and_attempts() -> None:
    attempts = [
        ExecutionFault(
            operation="llm",
            disposition=FaultDisposition.RETRYABLE,
            run_blocking=True,
            provider_code="RateLimitError:429",
            attempt_no=1,
        ),
        ExecutionFault(
            operation="llm",
            disposition=FaultDisposition.RETRYABLE,
            run_blocking=True,
            provider_code="RateLimitError:429",
            attempt_no=2,
        ),
        ExecutionFault(
            operation="llm",
            disposition=FaultDisposition.RETRYABLE,
            run_blocking=True,
            provider_code="RateLimitError:429",
            attempt_no=3,
        ),
    ]
    err = RunBlockingFaultError(attempts[-1], attempts=attempts, exhausted=True)
    result = finalization_for_blocking_fault(
        err,
        carry_forward=[{"id": "m1", "description": "continue", "status": "active"}],
    )
    assert result.decision["target"] == "agent"
    assert result.parse_error == "system_retry_exhausted"
    assert "attempts=3" in result.report
    assert "system_retry_exhausted" in result.report
    assert result.carry_forward[0]["id"] == "m1"
