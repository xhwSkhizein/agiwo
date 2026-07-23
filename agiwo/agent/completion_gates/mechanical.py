"""Mechanical completion gates: Workers + open RunPlan milestones."""

from agiwo.agent.completion_gates.models import AllowComplete, Continue, GateDecision
from agiwo.agent.models.plan import Milestone, RunPlan

_OPEN_MILESTONE_STATUSES = frozenset({"pending", "active"})


def open_milestones(plan: RunPlan) -> list[Milestone]:
    """Return milestones that still block Run completion."""
    return [
        milestone
        for milestone in plan.milestones
        if milestone.status in _OPEN_MILESTONE_STATUSES
    ]


def build_mechanical_feedback(
    *,
    open_items: list[Milestone],
    active_worker_ids: frozenset[str],
) -> str:
    """Build actionable feedback for the model when stop is blocked."""
    parts: list[str] = []
    if active_worker_ids:
        workers = ", ".join(sorted(active_worker_ids))
        parts.append(
            "Unfinished Workers remain "
            f"({workers}). Wait for their reports or cancel them before ending the Run."
        )
    if open_items:
        items = "\n".join(
            f"- [{item.status}] {item.id}: {item.description}" for item in open_items
        )
        parts.append(
            "The run plan still has unfinished milestones. Continue the work and use "
            "update_plan to complete, abandon, or revise them before ending:\n"
            f"{items}"
        )
    return "\n\n".join(parts)


def evaluate_mechanical(
    *,
    plan: RunPlan,
    active_worker_ids: frozenset[str],
) -> GateDecision:
    """Pure mechanical gate decision from RunPlan + Worker registry snapshot."""
    remaining = open_milestones(plan)
    if not remaining and not active_worker_ids:
        return AllowComplete()
    return Continue(
        feedback_text=build_mechanical_feedback(
            open_items=remaining,
            active_worker_ids=active_worker_ids,
        )
    )


__all__ = [
    "build_mechanical_feedback",
    "evaluate_mechanical",
    "open_milestones",
]
