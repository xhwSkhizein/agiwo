"""Optional LLM budget admission protocol for a Run.

Callers may inject an ``LlmBudgetGate`` on the Agent. When a gate is present
it is consulted before each provider attempt. The retired Objective plane used
``objective_id`` on admit/cost events; those fields remain as opaque run-scope
tags when a gate is wired.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class LlmBudgetDenied(Exception):
    """Raised when budget admission refuses a provider attempt."""

    def __init__(
        self,
        *,
        reason: str,
        used: float | None = None,
        limit: float | None = None,
        call_cost_ceiling: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.reason = reason
        self.used = used
        self.limit = limit
        self.call_cost_ceiling = call_cost_ceiling
        self.details = details or {}
        super().__init__(reason)


class MissingBudgetGateError(Exception):
    """A budget scope tag is set but no LlmBudgetGate was injected."""

    def __init__(self, scope_id: str) -> None:
        self.objective_id = scope_id  # legacy attribute name for callers/tests
        self.scope_id = scope_id
        super().__init__(
            f"budget-scoped run {scope_id!r} requires an LlmBudgetGate (fail-closed)"
        )


@dataclass(frozen=True, slots=True)
class LlmAttemptAdmitRequest:
    objective_id: str
    run_id: str
    logical_call_id: str
    phase: str
    attempt_no: int
    call_ordinal: int
    request_tokens: int
    max_output_tokens: int
    call_cost_ceiling: float
    price_snapshot: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LlmAttemptCostEvent:
    objective_id: str
    run_id: str
    logical_call_id: str
    phase: str
    attempt_no: int
    call_ordinal: int
    request_tokens: int
    accepted_output_tokens: int
    call_cost_ceiling: float
    cost_usd: float
    response_observed: bool
    source: str
    price_snapshot: dict[str, float] = field(default_factory=dict)
    retry_reason: str | None = None


@runtime_checkable
class LlmBudgetGate(Protocol):
    async def check_before_attempt(self, request: LlmAttemptAdmitRequest) -> None:
        """Allow the attempt or raise LlmBudgetDenied (no provider call)."""

    async def record_attempt_cost(self, event: LlmAttemptCostEvent) -> None:
        """Idempotently append actual LLM cost for this attempt."""


class PermissiveLlmBudgetGate:
    """No-op gate for tests that inject a budget scope without a real ledger."""

    async def check_before_attempt(self, request: LlmAttemptAdmitRequest) -> None:
        del request

    async def record_attempt_cost(self, event: LlmAttemptCostEvent) -> None:
        del event


__all__ = [
    "LlmAttemptAdmitRequest",
    "LlmAttemptCostEvent",
    "LlmBudgetDenied",
    "LlmBudgetGate",
    "MissingBudgetGateError",
    "PermissiveLlmBudgetGate",
]
