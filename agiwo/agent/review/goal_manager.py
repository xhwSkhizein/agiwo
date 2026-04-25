"""Milestone declaration, activation, completion, and querying."""

from agiwo.agent.models.review import Milestone, ReviewState


def _active_milestone_id(milestones: list[Milestone]) -> str | None:
    for milestone in milestones:
        if milestone.status == "active":
            return milestone.id
    return None


def declare_milestones(
    state: ReviewState,
    milestones: list[Milestone],
    *,
    current_seq: int = 0,
) -> list[str]:
    """Declare or update milestones while preserving existing milestone metadata."""

    previous_active_id = _active_milestone_id(state.milestones)
    existing_ids = {milestone.id for milestone in state.milestones}

    for milestone in milestones:
        if milestone.id in existing_ids:
            for index, existing in enumerate(state.milestones):
                if existing.id != milestone.id:
                    continue
                milestone.declared_at_seq = existing.declared_at_seq
                milestone.completed_at_seq = existing.completed_at_seq
                if (
                    milestone.status == "completed"
                    and milestone.completed_at_seq is None
                ):
                    milestone.completed_at_seq = current_seq
                state.milestones[index] = milestone
                break
            continue
        milestone.declared_at_seq = current_seq
        if milestone.status == "completed" and milestone.completed_at_seq is None:
            milestone.completed_at_seq = current_seq
        state.milestones.append(milestone)

    if _active_milestone_id(state.milestones) is None and state.milestones:
        for milestone in state.milestones:
            if milestone.status == "pending":
                milestone.status = "active"
                break

    current_active_id = _active_milestone_id(state.milestones)
    if previous_active_id is not None and current_active_id != previous_active_id:
        state.pending_review_reason = "milestone_switch"

    return [milestone.id for milestone in milestones]


def complete_active_milestone(state: ReviewState, *, seq: int) -> bool:
    """Mark the active milestone as completed. Returns True if one was completed."""

    for milestone in state.milestones:
        if milestone.status != "active":
            continue
        milestone.status = "completed"
        milestone.completed_at_seq = seq
        state.pending_review_reason = "milestone_switch"
        return True
    return False


def activate_next_milestone(state: ReviewState) -> Milestone | None:
    """Activate the first pending milestone. Returns it, or None."""

    had_active_milestone = _active_milestone_id(state.milestones) is not None
    for milestone in state.milestones:
        if milestone.status != "pending":
            continue
        milestone.status = "active"
        if had_active_milestone:
            state.pending_review_reason = "milestone_switch"
        return milestone
    return None


def get_active_milestone(state: ReviewState) -> Milestone | None:
    """Return the currently active milestone, or None."""

    for milestone in state.milestones:
        if milestone.status == "active":
            return milestone
    return None


__all__ = [
    "activate_next_milestone",
    "complete_active_milestone",
    "declare_milestones",
    "get_active_milestone",
]
