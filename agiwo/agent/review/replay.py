"""Replay helpers for goal-directed review state."""

from collections.abc import Iterable

from agiwo.agent.models.log import (
    ReviewCheckpointRecorded,
    ReviewMilestonesUpdated,
    ReviewOutcomeRecorded,
    ReviewTriggerDecided,
    RunLogEntry,
    ToolStepCommitted,
)
from agiwo.agent.models.review import (
    Milestone,
    PendingReviewNotice,
    ReviewCheckpoint,
    ReviewState,
)


def build_review_state_from_entries(entries: Iterable[RunLogEntry]) -> ReviewState:
    """Rebuild live review state from committed run-log facts.

    ``consecutive_errors`` is intentionally transient runtime state. It is not
    persisted as a review fact and therefore remains at the ReviewState default
    during replay.
    """

    state = ReviewState()
    checkpoint_entry_seq = 0
    for entry in sorted(entries, key=lambda item: item.sequence):
        if isinstance(entry, ReviewMilestonesUpdated):
            if _milestone_transition_requires_review(
                previous=state.milestones,
                current=entry.milestones,
                reason=entry.reason,
                active_milestone_id=entry.active_milestone_id,
            ):
                state.pending_review_reason = "milestone_switch"
            state.milestones = list(entry.milestones)
            continue
        if isinstance(entry, ReviewCheckpointRecorded):
            state.latest_checkpoint = ReviewCheckpoint(
                seq=entry.checkpoint_seq,
                milestone_id=entry.milestone_id or "",
                confirmed_at=entry.created_at,
            )
            state.review_count_since_checkpoint = 0
            checkpoint_entry_seq = entry.sequence
            continue
        if isinstance(entry, ReviewTriggerDecided):
            state.pending_review_reason = None
            state.pending_review_notice = PendingReviewNotice(
                trigger_reason=entry.trigger_reason,
                active_milestone_id=entry.active_milestone_id,
                review_count_since_checkpoint=entry.review_count_since_checkpoint,
                trigger_tool_call_id=entry.trigger_tool_call_id,
                trigger_tool_step_id=entry.trigger_tool_step_id,
                notice_step_id=entry.notice_step_id,
            )
            continue
        if isinstance(entry, ReviewOutcomeRecorded):
            state.pending_review_notice = None
            state.pending_review_reason = None
            state.review_count_since_checkpoint = 0
            checkpoint_entry_seq = entry.sequence
            continue
        if isinstance(entry, ToolStepCommitted):
            if entry.name == "review_trajectory":
                continue
            if entry.sequence > checkpoint_entry_seq:
                state.review_count_since_checkpoint += 1
            continue
    return state


def _active_milestone_id(milestones: list[Milestone]) -> str | None:
    for milestone in milestones:
        if milestone.status == "active":
            return milestone.id
    return None


def _milestone_transition_requires_review(
    *,
    previous: list[Milestone],
    current: list[Milestone],
    reason: str,
    active_milestone_id: str | None,
) -> bool:
    previous_active_id = _active_milestone_id(previous)
    current_active_id = active_milestone_id or _active_milestone_id(current)
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


__all__ = ["build_review_state_from_entries"]
