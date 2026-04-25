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
    PendingReviewNotice,
    ReviewCheckpoint,
    ReviewState,
)


def build_review_state_from_entries(entries: Iterable[RunLogEntry]) -> ReviewState:
    """Rebuild live review state from committed run-log facts."""

    state = ReviewState()
    checkpoint_entry_seq = 0
    for entry in sorted(entries, key=lambda item: item.sequence):
        if isinstance(entry, ReviewMilestonesUpdated):
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


__all__ = ["build_review_state_from_entries"]
