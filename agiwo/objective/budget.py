"""ObjectiveBudget ledger helpers (handoff / verification quotas)."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agiwo.objective.errors import BudgetBoundaryHit, ValidationError
from agiwo.objective.log import FactDraft, fact_budget_usage_recorded
from agiwo.objective.models import (
    HandoffDecision,
    HandoffTarget,
    ObjectiveBudget,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class BudgetConsumption:
    """Planned append-only usage deltas for one Decision."""

    handoffs_delta: float = 0.0
    verification_attempts_delta: float = 0.0

    @property
    def is_empty(self) -> bool:
        return self.handoffs_delta == 0.0 and self.verification_attempts_delta == 0.0


def consumption_for_decision(decision: HandoffDecision) -> BudgetConsumption:
    """Count only agent/verifier handoffs; user targets do not consume quotas."""
    if decision.target is HandoffTarget.AGENT:
        return BudgetConsumption(handoffs_delta=1.0)
    if decision.target is HandoffTarget.VERIFIER:
        return BudgetConsumption(
            handoffs_delta=1.0,
            verification_attempts_delta=1.0,
        )
    return BudgetConsumption()


def check_and_plan_consumption(
    budget: ObjectiveBudget,
    decision: HandoffDecision,
    *,
    checked_at: datetime | None = None,
    pending_action: str = "create_next_assignment",
) -> BudgetConsumption:
    """Return planned deltas or raise BudgetBoundaryHit (no partial consume)."""
    plan = consumption_for_decision(decision)
    if plan.is_empty:
        return plan
    at = checked_at or utc_now()

    if plan.handoffs_delta > 0 and budget.handoffs.remaining < plan.handoffs_delta:
        raise BudgetBoundaryHit(
            dimension="handoffs",
            checked_at=at,
            used=budget.handoffs.used,
            limit=budget.handoffs.limit,
            pending_action=pending_action,
            required=plan.handoffs_delta,
        )
    if (
        plan.verification_attempts_delta > 0
        and budget.verification_attempts.remaining < plan.verification_attempts_delta
    ):
        raise BudgetBoundaryHit(
            dimension="verification_attempts",
            checked_at=at,
            used=budget.verification_attempts.used,
            limit=budget.verification_attempts.limit,
            pending_action=pending_action,
            required=plan.verification_attempts_delta,
        )
    return plan


def apply_consumption(
    budget: ObjectiveBudget,
    plan: BudgetConsumption,
) -> ObjectiveBudget:
    """Return a new budget snapshot after applying planned deltas."""
    handoffs = budget.handoffs
    verification = budget.verification_attempts
    if plan.handoffs_delta:
        handoffs = handoffs.with_used(handoffs.used + plan.handoffs_delta)
    if plan.verification_attempts_delta:
        verification = verification.with_used(
            verification.used + plan.verification_attempts_delta
        )
    return ObjectiveBudget(
        handoffs=handoffs,
        verification_attempts=verification,
        llm_cost_usd=budget.llm_cost_usd,
        active_seconds=budget.active_seconds,
    )


def usage_facts_for_consumption(
    *,
    budget_before: ObjectiveBudget,
    plan: BudgetConsumption,
    occurred_at: datetime | None = None,
) -> list[FactDraft]:
    """Emit BudgetUsageRecorded drafts; verifier emits both dims in one batch."""
    if plan.is_empty:
        return []
    after = apply_consumption(budget_before, plan)
    drafts: list[FactDraft] = []
    if plan.handoffs_delta:
        draft = fact_budget_usage_recorded(
            dimension="handoffs",
            delta=plan.handoffs_delta,
            used_after=after.handoffs.used,
        )
        drafts.append(
            FactDraft(
                kind=draft.kind,
                payload=draft.payload,
                occurred_at=occurred_at,
            )
        )
    if plan.verification_attempts_delta:
        draft = fact_budget_usage_recorded(
            dimension="verification_attempts",
            delta=plan.verification_attempts_delta,
            used_after=after.verification_attempts.used,
        )
        drafts.append(
            FactDraft(
                kind=draft.kind,
                payload=draft.payload,
                occurred_at=occurred_at,
            )
        )
    return drafts


def validate_limit_not_below_used(
    budget: ObjectiveBudget,
    *,
    handoffs: float | None = None,
    verification_attempts: float | None = None,
    llm_cost_usd: float | None = None,
    active_seconds: float | None = None,
) -> None:
    checks = (
        ("handoffs", handoffs, budget.handoffs.used),
        (
            "verification_attempts",
            verification_attempts,
            budget.verification_attempts.used,
        ),
        ("llm_cost_usd", llm_cost_usd, budget.llm_cost_usd.used),
        ("active_seconds", active_seconds, budget.active_seconds.used),
    )
    for name, new_limit, used in checks:
        if new_limit is not None and new_limit < used:
            raise ValidationError(
                f"budget {name} limit cannot be below used",
                dimension=name,
                limit=new_limit,
                used=used,
            )


def budget_agent_projection(
    budget: ObjectiveBudget,
    *,
    current_window_used_seconds: float | None = None,
) -> dict[str, Any]:
    """Read-only limit/used/remaining for agent-facing ObjectiveView consumers.

    Active time remaining is based on the current open window (not historical
    audit totals stored in ``budget.active_seconds.used``).
    """
    window_used = (
        current_window_used_seconds
        if current_window_used_seconds is not None
        else budget.active_seconds.used
    )
    return {
        "max_handoffs": budget.handoffs.limit,
        "used_handoffs": budget.handoffs.used,
        "remaining_handoffs": budget.handoffs.remaining,
        "max_verification_attempts": budget.verification_attempts.limit,
        "used_verification_attempts": budget.verification_attempts.used,
        "remaining_verification_attempts": budget.verification_attempts.remaining,
        "max_llm_cost_usd": budget.llm_cost_usd.limit,
        "used_llm_cost_usd": budget.llm_cost_usd.used,
        "remaining_llm_cost_usd": budget.llm_cost_usd.remaining,
        "max_active_seconds": budget.active_seconds.limit,
        "used_active_seconds": window_used,
        "remaining_active_seconds": max(0.0, budget.active_seconds.limit - window_used),
        "historical_active_seconds": budget.active_seconds.used,
    }


__all__ = [
    "BudgetConsumption",
    "apply_consumption",
    "budget_agent_projection",
    "check_and_plan_consumption",
    "consumption_for_decision",
    "usage_facts_for_consumption",
    "validate_limit_not_below_used",
]
