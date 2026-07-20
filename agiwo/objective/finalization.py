"""Map Agent RunFinalizationResult into Objective domain facts/outcomes."""

from agiwo.agent.models.finalization import RunFinalizationResult
from agiwo.agent.models.run import RunStatus
from agiwo.objective.errors import ValidationError
from agiwo.objective.models import (
    ContributionAnnotationSpec,
    HandoffDecision,
    HandoffTarget,
    NewContributionSpec,
    ObjectiveUpdateSpec,
    RunOutcome,
    RunRole,
    new_id,
)
from agiwo.objective.projection import ObjectiveView


def map_finalization_to_outcome(
    *,
    view: ObjectiveView,
    run_id: str,
    role: RunRole,
    result: RunFinalizationResult,
    terminal_status: RunStatus = RunStatus.COMPLETED,
) -> tuple[RunOutcome, HandoffDecision]:
    active = view.active_root_run
    if active is None:
        raise ValidationError(
            "no active root run for finalization",
            objective_id=view.objective_id,
        )
    if active.run_id != run_id:
        raise ValidationError(
            "finalization run mismatch",
            expected=active.run_id,
            actual=run_id,
        )

    decision_raw = dict(result.decision)
    target = HandoffTarget(decision_raw["target"])
    if result.mechanical_handoff:
        raw_target = decision_raw.get("target", "agent")
        if raw_target not in {"agent", "user", "verifier"}:
            raw_target = "agent"
        target = HandoffTarget(raw_target)
        decision_raw = {"target": target.value}
        if target is HandoffTarget.USER:
            decision_raw["expects_reply"] = True
    decision = HandoffDecision(
        target=target,
        expects_reply=(
            bool(decision_raw.get("expects_reply"))
            if target is HandoffTarget.USER
            else None
        ),
        reason=result.parse_error,
    )

    # When verification_required, work cannot deliver directly.
    if (
        role is RunRole.WORK
        and view.verification_required
        and target is HandoffTarget.USER
        and not decision.expects_reply
    ):
        decision = HandoffDecision(
            target=HandoffTarget.VERIFIER,
            expects_reply=None,
            reason="verification_required_coerced_to_verifier",
        )
        target = HandoffTarget.VERIFIER

    update_spec = None
    if result.objective_update is not None:
        update_spec = ObjectiveUpdateSpec.from_dict(result.objective_update)

    report = (result.report or "").strip()
    if not report and terminal_status is RunStatus.COMPLETED:
        report = "(empty report)"

    outcome = RunOutcome(
        outcome_id=new_id("out_"),
        run_id=run_id,
        role=role,
        terminal_status=terminal_status,
        reason=result.parse_error or "finalized",
        report=report,
        new_contributions=tuple(
            NewContributionSpec(
                content=item["content"],
                summary=item.get("summary"),
            )
            for item in result.new_contributions
        ),
        contribution_annotations=tuple(
            ContributionAnnotationSpec(
                contribution_id=item["contribution_id"],
                annotation=item["annotation"],
                deactivate=bool(item.get("deactivate", False)),
            )
            for item in result.contribution_annotations
        ),
        decision=decision,
        objective_update=update_spec,
        carry_forward=tuple(
            str(item.get("id") or item.get("description") or "")
            for item in result.carry_forward
            if item
        ),
        provenance={
            "mechanical_handoff": result.mechanical_handoff,
            "parse_error": result.parse_error,
            "carry_forward_items": list(result.carry_forward),
        },
    )
    return outcome, decision


def next_run_role_for_decision(
    *,
    decision: HandoffDecision,
) -> RunRole | None:
    """Return next peer root Run role, or None when waiting/delivering."""
    if decision.target is HandoffTarget.AGENT:
        return RunRole.WORK
    if decision.target is HandoffTarget.VERIFIER:
        return RunRole.VERIFICATION
    return None


def should_deliver(
    decision: HandoffDecision,
    *,
    role: RunRole,
    verification_required: bool,
) -> bool:
    """Deliver on user/no-reply when work is allowed or the run is verification."""
    if decision.target is not HandoffTarget.USER or decision.expects_reply is not False:
        return False
    if role is RunRole.VERIFICATION:
        return True
    if role is RunRole.WORK:
        return not verification_required
    return False


__all__ = [
    "map_finalization_to_outcome",
    "next_run_role_for_decision",
    "should_deliver",
]
