"""Structured execution faults for infrastructure retry decisions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FaultDisposition(str, Enum):
    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    OUTCOME_UNKNOWN = "outcome_unknown"


class IdempotencyKind(str, Enum):
    GUARANTEED = "guaranteed"
    CONDITIONAL = "conditional"
    NOT_IDEMPOTENT = "not_idempotent"


@dataclass(frozen=True, slots=True)
class ExecutionFault:
    """Provider/tool fault with structured disposition (not error-text heuristics)."""

    operation: str
    disposition: FaultDisposition
    run_blocking: bool
    response_observed: bool = False
    external_effect_may_have_started: bool = False
    provider_code: str | None = None
    tool_code: str | None = None
    message: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    attempt_no: int = 1
    logical_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "disposition": self.disposition.value,
            "run_blocking": self.run_blocking,
            "response_observed": self.response_observed,
            "external_effect_may_have_started": self.external_effect_may_have_started,
            "provider_code": self.provider_code,
            "tool_code": self.tool_code,
            "message": self.message,
            "provenance": dict(self.provenance),
            "attempt_no": self.attempt_no,
            "logical_call_id": self.logical_call_id,
        }


class RunBlockingFaultError(Exception):
    """Fault that must end the current Run with a system report snapshot."""

    def __init__(
        self,
        fault: ExecutionFault,
        *,
        attempts: list[ExecutionFault] | None = None,
        exhausted: bool = False,
    ) -> None:
        self.fault = fault
        self.attempts = list(attempts or [fault])
        self.exhausted = exhausted
        super().__init__(fault.message or fault.disposition.value)


def may_auto_retry(
    fault: ExecutionFault,
    *,
    idempotency: IdempotencyKind,
    idempotency_key: str | None = None,
) -> bool:
    """Auto-retry only when disposition is retryable and idempotency is proven."""
    if fault.disposition is not FaultDisposition.RETRYABLE:
        return False
    if idempotency is IdempotencyKind.GUARANTEED:
        return True
    if idempotency is IdempotencyKind.CONDITIONAL:
        return bool(idempotency_key)
    return False


__all__ = [
    "ExecutionFault",
    "FaultDisposition",
    "IdempotencyKind",
    "RunBlockingFaultError",
    "may_auto_retry",
]
