"""Atomic RunPlan change application and tool-result handling."""

from copy import deepcopy
from typing import Any, cast

from agiwo.agent.models.plan import (
    Milestone,
    MilestoneStatus,
    RunPlan,
    RunPlanUpdate,
    RunPlanUpdateReason,
)
from agiwo.tool.base import ToolResult

_VALID_STATUSES: set[str] = {"pending", "active", "completed", "abandoned"}
_RESOLVED_STATUSES: set[str] = {"completed", "abandoned"}


class PlanValidationError(ValueError):
    """Raised when an update_plan change batch is illegal."""


def parse_plan_changes(output: object) -> list[dict[str, Any]]:
    """Extract normalized change dicts from a successful update_plan tool output."""
    if not isinstance(output, dict):
        return []
    raw_changes = output.get("changes")
    if not isinstance(raw_changes, list):
        return []
    return [item for item in raw_changes if isinstance(item, dict)]


def apply_plan_changes(
    plan: RunPlan,
    changes: list[dict[str, Any]],
    *,
    current_seq: int,
    source_tool_call_id: str | None,
    reason: RunPlanUpdateReason = "updated",
) -> RunPlanUpdate:
    """Atomically apply *changes* to *plan*. Mutates *plan* on success only."""
    if not changes:
        raise PlanValidationError("changes must be a non-empty array")

    previous_active_id = plan.active_milestone_id
    working = deepcopy(plan.milestones)
    by_id = {milestone.id: milestone for milestone in working}
    explicit_active_ids: list[str] = []

    for index, change in enumerate(changes):
        milestone_id = _require_non_empty_str(
            change.get("id"), field_name=f"changes[{index}].id"
        )
        existing = by_id.get(milestone_id)
        if existing is None:
            description = change.get("description")
            if not isinstance(description, str) or not description.strip():
                raise PlanValidationError(
                    f"changes[{index}]: new milestone id {milestone_id!r} "
                    "requires a non-empty description"
                )
            status_raw = change.get("status", "pending")
            status = _parse_status(status_raw, field_name=f"changes[{index}].status")
            milestone = Milestone(
                id=milestone_id,
                description=description.strip(),
                status=status,
                declared_at_seq=current_seq,
            )
            if status == "completed":
                milestone.completed_at_seq = current_seq
            working.append(milestone)
            by_id[milestone_id] = milestone
            if status == "active":
                explicit_active_ids.append(milestone_id)
            continue

        if "description" in change:
            description = change["description"]
            if not isinstance(description, str) or not description.strip():
                raise PlanValidationError(
                    f"changes[{index}].description must be a non-empty string"
                )
            existing.description = description.strip()

        if "status" in change:
            status = _parse_status(
                change["status"], field_name=f"changes[{index}].status"
            )
            existing.status = status
            if status == "completed" and existing.completed_at_seq is None:
                existing.completed_at_seq = current_seq
            if status in {"pending", "active"}:
                existing.completed_at_seq = None
            if status == "active":
                explicit_active_ids.append(milestone_id)

    if len(explicit_active_ids) > 1:
        raise PlanValidationError("changes may contain at most one active item")

    if len(explicit_active_ids) == 1:
        new_active_id = explicit_active_ids[0]
        for milestone in working:
            if milestone.id == new_active_id:
                milestone.status = "active"
            elif milestone.status == "active":
                # Explicit activation demotes unresolved previous active to pending.
                milestone.status = "pending"

    _normalize_active(working)
    _validate_milestones(working)

    active_id = _active_milestone_id(working)
    plan.milestones = working
    plan.revision += 1
    return RunPlanUpdate(
        milestones=list(plan.milestones),
        active_milestone_id=active_id,
        source_tool_call_id=source_tool_call_id,
        reason=_infer_reason(reason, previous_active_id, active_id, changes),
        revision=plan.revision,
        milestone_switch=previous_active_id is not None
        and active_id != previous_active_id,
    )


def handle_plan_tool_result(
    result: ToolResult,
    plan: RunPlan,
    *,
    current_seq: int,
) -> RunPlanUpdate | None:
    if result.tool_name != "update_plan" or not result.is_success:
        return None
    changes = parse_plan_changes(result.output)
    if not changes:
        return None
    return apply_plan_changes(
        plan,
        changes,
        current_seq=current_seq,
        source_tool_call_id=result.tool_call_id or None,
    )


def format_plan_update_content(update: RunPlanUpdate) -> str:
    counts = {
        "pending": 0,
        "active": 0,
        "completed": 0,
        "abandoned": 0,
    }
    for milestone in update.milestones:
        counts[milestone.status] = counts.get(milestone.status, 0) + 1
    return (
        f"Plan updated (revision={update.revision}): "
        f"pending={counts['pending']}, active={counts['active']}, "
        f"completed={counts['completed']}, abandoned={counts['abandoned']}"
    )


def _normalize_active(milestones: list[Milestone]) -> None:
    active_ids = [m.id for m in milestones if m.status == "active"]
    if len(active_ids) > 1:
        raise PlanValidationError("milestones may contain at most one active item")
    if active_ids:
        return
    unresolved = [m for m in milestones if m.status not in _RESOLVED_STATUSES]
    if not unresolved:
        return
    for milestone in milestones:
        if milestone.status == "pending":
            milestone.status = "active"
            return


def _validate_milestones(milestones: list[Milestone]) -> None:
    seen_ids: set[str] = set()
    active_count = 0
    for milestone in milestones:
        if not milestone.id.strip():
            raise PlanValidationError("milestone id must be a non-empty string")
        if not milestone.description.strip():
            raise PlanValidationError(
                "milestone description must be a non-empty string"
            )
        if milestone.id in seen_ids:
            raise PlanValidationError(f"duplicate milestone id: {milestone.id}")
        seen_ids.add(milestone.id)
        if milestone.status == "active":
            active_count += 1
    if active_count > 1:
        raise PlanValidationError("milestones may contain at most one active item")


def _active_milestone_id(milestones: list[Milestone]) -> str | None:
    for milestone in milestones:
        if milestone.status == "active":
            return milestone.id
    return None


def _parse_status(value: object, *, field_name: str) -> MilestoneStatus:
    if not isinstance(value, str) or value not in _VALID_STATUSES:
        raise PlanValidationError(f"invalid milestone status at {field_name}: {value}")
    return cast(MilestoneStatus, value)


def _require_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _infer_reason(
    default: RunPlanUpdateReason,
    previous_active_id: str | None,
    active_id: str | None,
    changes: list[dict[str, Any]],
) -> RunPlanUpdateReason:
    if previous_active_id is None and active_id is not None:
        return "declared"
    statuses = {
        change.get("status")
        for change in changes
        if isinstance(change, dict) and "status" in change
    }
    if "completed" in statuses:
        return "completed"
    if "active" in statuses and previous_active_id != active_id:
        return "activated"
    return default


__all__ = [
    "PlanValidationError",
    "apply_plan_changes",
    "format_plan_update_content",
    "handle_plan_tool_result",
    "parse_plan_changes",
]
