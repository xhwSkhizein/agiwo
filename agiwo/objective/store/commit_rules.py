"""Shared commit invariants for ObjectiveStore backends."""

from collections.abc import Mapping, Sequence

from agiwo.objective.errors import StoreError
from agiwo.objective.log import ObjectiveFactKind, ObjectiveLogEntry
from agiwo.objective.models import (
    OBJECTIVE_TERMINAL,
    ObjectiveBudget,
    ObjectiveStatus,
)
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.projection import project_objective
from agiwo.objective.store.base import SlotMutation


def derive_objective_status(
    facts: Sequence[ObjectiveLogEntry],
) -> ObjectiveStatus | None:
    """Replay status-affecting facts in sequence order (no full projection)."""
    status: ObjectiveStatus | None = None
    for fact in sorted(facts, key=lambda f: f.sequence):
        kind = fact.kind
        payload = fact.payload
        if kind == ObjectiveFactKind.OBJECTIVE_CREATED:
            status = ObjectiveStatus.CREATED
        elif kind == ObjectiveFactKind.OBJECTIVE_STATUS_CHANGED:
            status = ObjectiveStatus(payload["to_status"])
        elif kind == ObjectiveFactKind.CONTEXT_CAPACITY_EXCEEDED:
            if status not in OBJECTIVE_TERMINAL:
                status = ObjectiveStatus.WAITING_USER
        elif kind == ObjectiveFactKind.ROOT_RUN_STARTED:
            if status == ObjectiveStatus.CREATED:
                status = ObjectiveStatus.RUNNING
        elif kind == ObjectiveFactKind.DRAIN_STARTED:
            if status not in OBJECTIVE_TERMINAL:
                status = ObjectiveStatus.DRAINING
        elif kind == ObjectiveFactKind.DRAIN_COMPLETED:
            status = ObjectiveStatus(payload["next_status"])
        elif kind == ObjectiveFactKind.OBJECTIVE_DELIVERED:
            if status not in OBJECTIVE_TERMINAL:
                status = ObjectiveStatus.COMPLETED
    return status


def validate_commit_invariants(
    *,
    existing_facts_by_objective: Mapping[str, Sequence[ObjectiveLogEntry]],
    facts: Sequence[ObjectiveLogEntry],
    slot_mutation: SlotMutation | None,
    outbox_records: Sequence[DispatchRequested],
) -> None:
    """Enforce Session slot release and RootRunRequested↔outbox coupling."""
    _validate_root_run_requested_has_outbox(facts, outbox_records)
    _validate_budget_usage_facts(existing_facts_by_objective, facts)
    if slot_mutation is not None and slot_mutation.action == "release":
        _validate_slot_release_requires_terminal(
            existing_facts_by_objective=existing_facts_by_objective,
            facts=facts,
            slot_mutation=slot_mutation,
        )


def _validate_budget_usage_facts(
    existing_facts_by_objective: Mapping[str, Sequence[ObjectiveLogEntry]],
    facts: Sequence[ObjectiveLogEntry],
) -> None:
    """Reject usage facts that disagree with the projected used baseline."""
    usage_facts = [
        fact for fact in facts if fact.kind == ObjectiveFactKind.BUDGET_USAGE_RECORDED
    ]
    if not usage_facts:
        return
    by_objective: dict[str, list[ObjectiveLogEntry]] = {}
    for fact in usage_facts:
        by_objective.setdefault(fact.objective_id, []).append(fact)

    for objective_id, batch in by_objective.items():
        prior = list(existing_facts_by_objective.get(objective_id, ()))
        view = project_objective(prior, objective_id=objective_id)
        if view is None or view.budget is None:
            raise StoreError(
                "budget usage requires an existing Objective budget",
                objective_id=objective_id,
            )
        budget: ObjectiveBudget = view.budget
        for fact in sorted(batch, key=lambda f: f.sequence):
            dimension = fact.payload["dimension"]
            delta = float(fact.payload["delta"])
            used_after = float(fact.payload["used_after"])
            current = _dimension_used(budget, dimension)
            expected = current + delta
            if abs(used_after - expected) > 1e-9:
                raise StoreError(
                    "budget usage used_after disagrees with projected used",
                    objective_id=objective_id,
                    dimension=dimension,
                    projected_used=current,
                    delta=delta,
                    used_after=used_after,
                    expected=expected,
                )
            dim = _dimension(budget, dimension)
            if used_after > dim.limit + 1e-9:
                raise StoreError(
                    "budget usage exceeds limit",
                    objective_id=objective_id,
                    dimension=dimension,
                    used_after=used_after,
                    limit=dim.limit,
                )
            budget = _with_dimension_used(budget, dimension, used_after)


def _dimension(budget: ObjectiveBudget, dimension: str):
    mapping = {
        "handoffs": budget.handoffs,
        "verification_attempts": budget.verification_attempts,
        "llm_cost_usd": budget.llm_cost_usd,
        "active_seconds": budget.active_seconds,
    }
    if dimension not in mapping:
        raise StoreError("unknown budget dimension", dimension=dimension)
    return mapping[dimension]


def _dimension_used(budget: ObjectiveBudget, dimension: str) -> float:
    return _dimension(budget, dimension).used


def _with_dimension_used(
    budget: ObjectiveBudget,
    dimension: str,
    used: float,
) -> ObjectiveBudget:
    handoffs = budget.handoffs
    verification = budget.verification_attempts
    llm = budget.llm_cost_usd
    active = budget.active_seconds
    if dimension == "handoffs":
        handoffs = handoffs.with_used(used)
    elif dimension == "verification_attempts":
        verification = verification.with_used(used)
    elif dimension == "llm_cost_usd":
        llm = llm.with_used(used)
    elif dimension == "active_seconds":
        active = active.with_used(used)
    else:
        raise StoreError("unknown budget dimension", dimension=dimension)
    return ObjectiveBudget(
        handoffs=handoffs,
        verification_attempts=verification,
        llm_cost_usd=llm,
        active_seconds=active,
    )


def _validate_root_run_requested_has_outbox(
    facts: Sequence[ObjectiveLogEntry],
    outbox_records: Sequence[DispatchRequested],
) -> None:
    requested_ids = {
        fact.payload["run_id"]
        for fact in facts
        if fact.kind == ObjectiveFactKind.ROOT_RUN_REQUESTED
    }
    if not requested_ids:
        return
    outbox_ids = {record.run_id for record in outbox_records}
    missing = sorted(requested_ids - outbox_ids)
    if missing:
        raise StoreError(
            "RootRunRequested requires DispatchRequested in the same commit",
            missing_run_ids=missing,
        )


def _validate_slot_release_requires_terminal(
    *,
    existing_facts_by_objective: Mapping[str, Sequence[ObjectiveLogEntry]],
    facts: Sequence[ObjectiveLogEntry],
    slot_mutation: SlotMutation,
) -> None:
    objective_id = slot_mutation.objective_id
    combined = list(existing_facts_by_objective.get(objective_id, ()))
    combined.extend(f for f in facts if f.objective_id == objective_id)
    status = derive_objective_status(combined)
    if status is None or status not in OBJECTIVE_TERMINAL:
        raise StoreError(
            "cannot release session slot for a non-terminal Objective",
            session_id=slot_mutation.session_id,
            objective_id=objective_id,
            status=None if status is None else status.value,
        )


__all__ = [
    "derive_objective_status",
    "validate_commit_invariants",
]
