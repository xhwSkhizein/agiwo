"""Goal Manager — milestone declaration, activation, completion, and querying."""

from agiwo.agent.models.review import Milestone, ReviewState


class GoalManager:
    """Thin OO wrapper around module-level milestone helpers."""

    def declare_milestones(
        self,
        state: ReviewState,
        milestones: list[Milestone],
        *,
        current_seq: int = 0,
    ) -> list[str]:
        return declare_milestones(state, milestones, current_seq=current_seq)

    def complete_active_milestone(self, state: ReviewState, *, seq: int) -> bool:
        return complete_active_milestone(state, seq=seq)

    def activate_next_milestone(self, state: ReviewState) -> Milestone | None:
        return activate_next_milestone(state)

    def get_active_milestone(self, state: ReviewState) -> Milestone | None:
        return get_active_milestone(state)


def declare_milestones(
    state: ReviewState,
    milestones: list[Milestone],
    *,
    current_seq: int = 0,
) -> list[str]:
    """Declare or update milestones. First pending becomes active if none is active.

    Returns the list of milestone ids that were declared.
    """
    has_active = any(m.status == "active" for m in state.milestones)

    new_ids = [m.id for m in milestones]
    existing_ids = {m.id for m in state.milestones}

    # Update existing or add new
    for m in milestones:
        if m.id in existing_ids:
            for i, existing in enumerate(state.milestones):
                if existing.id == m.id:
                    m.declared_at_seq = existing.declared_at_seq
                    state.milestones[i] = m
                    break
        else:
            m.declared_at_seq = current_seq
            state.milestones.append(m)

    # If no milestone is currently active, activate the first pending
    if not has_active and state.milestones:
        for m in state.milestones:
            if m.status == "pending":
                m.status = "active"
                break

    return new_ids


def complete_active_milestone(state: ReviewState, *, seq: int) -> bool:
    """Mark the active milestone as completed. Returns True if one was completed."""
    for m in state.milestones:
        if m.status == "active":
            m.status = "completed"
            m.completed_at_seq = seq
            state.is_review_pending = True
            return True
    return False


def activate_next_milestone(state: ReviewState) -> Milestone | None:
    """Activate the first pending milestone. Returns it, or None."""
    for m in state.milestones:
        if m.status == "pending":
            m.status = "active"
            state.is_review_pending = True
            return m
    return None


def get_active_milestone(state: ReviewState) -> Milestone | None:
    """Return the currently active milestone, or None."""
    for m in state.milestones:
        if m.status == "active":
            return m
    return None


__all__ = [
    "GoalManager",
    "activate_next_milestone",
    "complete_active_milestone",
    "declare_milestones",
    "get_active_milestone",
]
