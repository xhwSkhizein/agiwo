"""Replay helpers for goal and trajectory introspection state."""

from collections.abc import Iterable
from dataclasses import dataclass

from agiwo.agent.introspect.goal import (
    active_milestone_id_from_milestones,
    milestone_transition_requires_introspection,
)
from agiwo.agent.introspect.models import (
    GoalState,
    IntrospectionCheckpoint,
    IntrospectionState,
    PendingIntrospectionNotice,
)
from agiwo.agent.models.log import (
    GoalMilestonesUpdated,
    IntrospectionCheckpointRecorded,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    RunLogEntry,
    ToolStepCommitted,
)


@dataclass(frozen=True)
class IntrospectReplayState:
    goal: GoalState
    introspection: IntrospectionState


def build_introspect_state_from_entries(
    entries: Iterable[RunLogEntry],
) -> IntrospectReplayState:
    """Rebuild live introspection state from committed run-log facts.

    ``consecutive_errors`` is intentionally transient runtime state. It is not
    persisted as an introspection fact and therefore remains at the default.
    """

    goal = GoalState()
    introspection = IntrospectionState()
    for entry in sorted(entries, key=lambda item: item.sequence):
        if isinstance(entry, GoalMilestonesUpdated):
            if milestone_transition_requires_introspection(
                previous=goal.milestones,
                current=entry.milestones,
                reason=entry.reason,
                active_milestone_id=entry.active_milestone_id,
            ):
                introspection.pending_milestone_switch = True
            goal.milestones = list(entry.milestones)
            goal.active_milestone_id = (
                entry.active_milestone_id
                or active_milestone_id_from_milestones(goal.milestones)
            )
            continue
        if isinstance(entry, IntrospectionCheckpointRecorded):
            introspection.latest_aligned_checkpoint = IntrospectionCheckpoint(
                seq=entry.checkpoint_seq,
                milestone_id=entry.milestone_id or "",
                confirmed_at=entry.created_at,
            )
            introspection.review_count_since_boundary = 0
            introspection.last_boundary_seq = entry.checkpoint_seq
            continue
        if isinstance(entry, IntrospectionTriggered):
            introspection.pending_milestone_switch = False
            introspection.notice_requested = True
            introspection.pending_trigger = PendingIntrospectionNotice(
                trigger_reason=entry.trigger_reason,
                active_milestone_id=entry.active_milestone_id,
                review_count_since_boundary=entry.review_count_since_boundary,
                trigger_tool_call_id=entry.trigger_tool_call_id,
                trigger_tool_step_id=entry.trigger_tool_step_id,
                notice_step_id=entry.notice_step_id,
            )
            continue
        if isinstance(entry, IntrospectionOutcomeRecorded):
            introspection.pending_trigger = None
            introspection.notice_requested = False
            introspection.pending_milestone_switch = False
            introspection.review_count_since_boundary = 0
            introspection.last_boundary_seq = entry.boundary_seq
            continue
        if isinstance(entry, ToolStepCommitted):
            if entry.name == "review_trajectory":
                continue
            if entry.sequence > introspection.last_boundary_seq:
                introspection.review_count_since_boundary += 1
            continue
    return IntrospectReplayState(goal=goal, introspection=introspection)


__all__ = ["IntrospectReplayState", "build_introspect_state_from_entries"]
