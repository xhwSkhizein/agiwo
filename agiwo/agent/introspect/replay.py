"""Replay helpers for RunPlan and trajectory introspection state."""

from collections.abc import Iterable
from dataclasses import dataclass

from agiwo.agent.introspect.apply import parse_tool_usefulness_output
from agiwo.agent.introspect.models import (
    IntrospectionCheckpoint,
    IntrospectionState,
    Milestone,
    PendingIntrospectionNotice,
)
from agiwo.agent.models.log import (
    IntrospectionCheckpointRecorded,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    RunLogEntry,
    RunPlanUpdated,
    ToolStepCommitted,
)
from agiwo.agent.models.plan import RunPlan


@dataclass(frozen=True)
class IntrospectReplayState:
    plan: RunPlan
    introspection: IntrospectionState


def build_introspect_state_from_entries(
    entries: Iterable[RunLogEntry],
) -> IntrospectReplayState:
    """Rebuild live introspection state from committed run-log facts.

    ``consecutive_errors`` is intentionally transient runtime state. It is not
    persisted as an introspection fact and therefore remains at the default.
    """

    plan = RunPlan()
    introspection = IntrospectionState()
    for entry in sorted(entries, key=lambda item: item.sequence):
        if isinstance(entry, RunPlanUpdated):
            if _milestone_transition_requires_introspection(
                previous=plan.milestones,
                current=entry.milestones,
                reason=entry.reason,
                active_milestone_id=entry.active_milestone_id,
            ):
                introspection.pending_milestone_switch = True
            plan.milestones = list(entry.milestones)
            plan.revision = entry.revision
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
            introspection.latest_tool_usefulness = parse_tool_usefulness_output(
                entry.tool_usefulness
            )
            continue
        if isinstance(entry, ToolStepCommitted):
            if entry.name == "review_trajectory":
                continue
            if entry.sequence > introspection.last_boundary_seq:
                introspection.review_count_since_boundary += 1
            continue
    return IntrospectReplayState(plan=plan, introspection=introspection)


def _active_milestone_id(milestones: list[Milestone]) -> str | None:
    for milestone in milestones:
        if milestone.status == "active":
            return milestone.id
    return None


def _milestone_transition_requires_introspection(
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


__all__ = ["IntrospectReplayState", "build_introspect_state_from_entries"]
