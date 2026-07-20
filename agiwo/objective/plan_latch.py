"""Set verification_required when a root RunPlan has ever had milestones.

Authority lives at the Decision boundary: Objective reads RunLog plan facts
before mapping finalization, then commits VerificationRequiredSet if needed.
Live Agent hooks are not used.
"""

from collections.abc import Sequence

from agiwo.agent.models.log import RunPlanUpdated
from agiwo.objective.log import FactBatch, fact_verification_required_set
from agiwo.objective.models import CommandReceiptStatus, utc_now
from agiwo.objective.projection import project_objective
from agiwo.objective.store.base import CommandReceipt, ObjectiveStore, command_scope
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def run_plan_ever_had_milestone(entries: Sequence[object]) -> tuple[bool, str | None]:
    """Return (True, active_or_first_id) when any RunPlanUpdated has ≥1 milestone."""
    for entry in entries:
        if not isinstance(entry, RunPlanUpdated):
            continue
        if len(entry.milestones) < 1:
            continue
        milestone_id = entry.active_milestone_id or entry.milestones[0].id
        return True, milestone_id
    return False, None


async def maybe_commit_verification_required(
    store: ObjectiveStore,
    *,
    objective_id: str,
    run_id: str,
    milestone_id: str | None = None,
) -> bool:
    """Commit VerificationRequiredSet once when latch is not yet set."""
    facts = await store.list_facts(objective_id=objective_id)
    view = project_objective(facts, objective_id=objective_id)
    if view is None or view.verification_required:
        return False
    now = utc_now()
    seq = await store.get_max_sequence(objective_id) + 1
    receipt = CommandReceipt(
        scope=command_scope(objective_id, "verification_required"),
        idempotency_key=f"verification_required:{objective_id}",
        request_hash=run_id,
        status=CommandReceiptStatus.COMPLETED,
        response_payload={"run_id": run_id, "milestone_id": milestone_id},
        created_at=now,
        completed_at=now,
    )
    fact = FactBatch(
        objective_id=objective_id,
        start_sequence=seq,
        now=now,
    ).add(
        fact_verification_required_set(
            run_id=run_id,
            milestone_id=milestone_id,
        )
    )
    await store.commit_command(receipt=receipt, facts=[fact])
    logger.info(
        "objective_verification_required_set",
        objective_id=objective_id,
        run_id=run_id,
        milestone_id=milestone_id,
    )
    return True


async def ensure_verification_required_from_run_plan(
    store: ObjectiveStore,
    *,
    objective_id: str,
    run_id: str,
    plan_entries: Sequence[object],
) -> bool:
    """Commit latch when RunLog shows this root run ever had ≥1 milestone."""
    had, milestone_id = run_plan_ever_had_milestone(plan_entries)
    if not had:
        return False
    return await maybe_commit_verification_required(
        store,
        objective_id=objective_id,
        run_id=run_id,
        milestone_id=milestone_id,
    )


__all__ = [
    "ensure_verification_required_from_run_plan",
    "maybe_commit_verification_required",
    "run_plan_ever_had_milestone",
]
