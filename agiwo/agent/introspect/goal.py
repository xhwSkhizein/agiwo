"""Goal and milestone rules for agent introspection."""

from typing import cast

from agiwo.agent.introspect.models import (
    GoalState,
    GoalUpdate,
    GoalUpdateReason,
    Milestone,
    MilestoneStatus,
)
from agiwo.tool.base import ToolResult

_VALID_STATUSES: set[str] = {"pending", "active", "completed", "abandoned"}


class GoalValidationError(ValueError):
    """Raised when a milestone declaration is not a valid goal contract."""


def parse_declared_milestones(output: object) -> list[Milestone]:
    if not isinstance(output, dict):
        return []
    raw_milestones = output.get("milestones")
    if not isinstance(raw_milestones, list):
        return []

    milestones: list[Milestone] = []
    seen_ids: set[str] = set()
    active_count = 0
    for raw in raw_milestones:
        if not isinstance(raw, dict):
            raise GoalValidationError("milestones must contain objects")
        milestone_id = raw.get("id")
        description = raw.get("description")
        if not isinstance(milestone_id, str) or not milestone_id.strip():
            raise GoalValidationError("milestone id must be a non-empty string")
        if not isinstance(description, str) or not description.strip():
            raise GoalValidationError(
                "milestone description must be a non-empty string"
            )
        normalized_id = milestone_id.strip()
        if normalized_id in seen_ids:
            raise GoalValidationError(f"duplicate milestone id: {normalized_id}")
        seen_ids.add(normalized_id)
        status = raw.get("status", "pending")
        if not isinstance(status, str) or status not in _VALID_STATUSES:
            raise GoalValidationError(f"invalid milestone status: {status}")
        if status == "active":
            active_count += 1
        milestones.append(
            Milestone(
                id=normalized_id,
                description=description.strip(),
                status=cast(MilestoneStatus, status),
            )
        )
    if active_count > 1:
        raise GoalValidationError("milestones may contain at most one active item")
    return milestones


def active_milestone_id_from_milestones(milestones: list[Milestone]) -> str | None:
    for milestone in milestones:
        if milestone.status == "active":
            return milestone.id
    return None


def milestone_transition_requires_introspection(
    *,
    previous: list[Milestone],
    current: list[Milestone],
    reason: str,
    active_milestone_id: str | None,
) -> bool:
    previous_active_id = active_milestone_id_from_milestones(previous)
    current_active_id = active_milestone_id or active_milestone_id_from_milestones(
        current
    )
    if previous_active_id is not None and current_active_id != previous_active_id:
        return True
    if reason in {"completed", "activated"}:
        return True

    previous_status_by_id = {milestone.id: milestone.status for milestone in previous}
    for milestone in current:
        if (
            previous_status_by_id.get(milestone.id) != "completed"
            and milestone.status == "completed"
        ):
            return True
    return False


def _validate_milestones(milestones: list[Milestone]) -> None:
    seen_ids: set[str] = set()
    active_count = 0
    for milestone in milestones:
        if not milestone.id.strip():
            raise GoalValidationError("milestone id must be a non-empty string")
        if not milestone.description.strip():
            raise GoalValidationError(
                "milestone description must be a non-empty string"
            )
        if milestone.id in seen_ids:
            raise GoalValidationError(f"duplicate milestone id: {milestone.id}")
        seen_ids.add(milestone.id)
        if milestone.status == "active":
            active_count += 1
    if active_count > 1:
        raise GoalValidationError("milestones may contain at most one active item")


def update_goal_milestones(
    state: GoalState,
    milestones: list[Milestone],
    *,
    current_seq: int,
    source_tool_call_id: str | None,
    reason: GoalUpdateReason = "declared",
) -> GoalUpdate:
    _validate_milestones(milestones)
    previous_milestones = list(state.milestones)
    existing_by_id = {milestone.id: milestone for milestone in state.milestones}
    next_by_id = dict(existing_by_id)

    for milestone in milestones:
        existing = existing_by_id.get(milestone.id)
        if existing is not None:
            milestone.declared_at_seq = existing.declared_at_seq
            milestone.completed_at_seq = existing.completed_at_seq
        else:
            milestone.declared_at_seq = current_seq
        if milestone.status == "completed" and milestone.completed_at_seq is None:
            milestone.completed_at_seq = current_seq
        next_by_id[milestone.id] = milestone

    next_milestones = list(next_by_id.values())
    if active_milestone_id_from_milestones(next_milestones) is None:
        for milestone in next_milestones:
            if milestone.status == "pending":
                milestone.status = "active"
                break

    active_id = active_milestone_id_from_milestones(next_milestones)
    state.milestones = next_milestones
    state.active_milestone_id = active_id
    return GoalUpdate(
        milestones=list(state.milestones),
        active_milestone_id=active_id,
        source_tool_call_id=source_tool_call_id,
        reason=reason,
        milestone_switch=milestone_transition_requires_introspection(
            previous=previous_milestones,
            current=next_milestones,
            reason=reason,
            active_milestone_id=active_id,
        ),
    )


def handle_goal_tool_result(
    result: ToolResult,
    state: GoalState,
    *,
    current_seq: int,
) -> GoalUpdate | None:
    if result.tool_name != "declare_milestones" or not result.is_success:
        return None
    milestones = parse_declared_milestones(result.output)
    if not milestones:
        return None
    return update_goal_milestones(
        state,
        milestones,
        current_seq=current_seq,
        source_tool_call_id=result.tool_call_id or None,
    )


__all__ = [
    "GoalValidationError",
    "active_milestone_id_from_milestones",
    "handle_goal_tool_result",
    "milestone_transition_requires_introspection",
    "parse_declared_milestones",
    "update_goal_milestones",
]
