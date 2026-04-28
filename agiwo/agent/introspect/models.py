"""Data models for agent goal, trajectory introspection, and context repair."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

MilestoneStatus = Literal["pending", "active", "completed", "abandoned"]
GoalUpdateReason = Literal["declared", "updated", "completed", "activated"]
IntrospectionTriggerReason = Literal[
    "step_interval", "consecutive_errors", "milestone_switch"
]
IntrospectionMode = Literal["metadata_only", "step_back"]
ContextRepairMode = Literal["metadata_only", "step_back"]


@dataclass
class Milestone:
    """A verifiable sub-goal declared by the agent."""

    id: str
    description: str
    status: MilestoneStatus = "pending"
    declared_at_seq: int = 0
    completed_at_seq: int | None = None


@dataclass
class GoalState:
    """Goal contract state for the current agent session."""

    milestones: list[Milestone] = field(default_factory=list)
    active_milestone_id: str | None = None

    @property
    def active_milestone(self) -> Milestone | None:
        if self.active_milestone_id is None:
            return None
        for milestone in self.milestones:
            if milestone.id == self.active_milestone_id:
                return milestone
        return None


@dataclass
class IntrospectionCheckpoint:
    """A confirmed aligned checkpoint."""

    seq: int
    milestone_id: str
    confirmed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PendingIntrospectionNotice:
    """Outstanding one-shot introspection notice in prompt-visible context."""

    trigger_reason: IntrospectionTriggerReason
    active_milestone_id: str | None
    review_count_since_boundary: int
    trigger_tool_call_id: str | None
    trigger_tool_step_id: str | None
    notice_step_id: str | None


@dataclass
class IntrospectionState:
    """Runtime state for goal trajectory introspection."""

    review_count_since_boundary: int = 0
    consecutive_errors: int = 0
    pending_trigger: PendingIntrospectionNotice | None = None
    last_boundary_seq: int = 0
    latest_aligned_checkpoint: IntrospectionCheckpoint | None = None
    pending_milestone_switch: bool = False
    notice_requested: bool = False


@dataclass(frozen=True)
class GoalUpdate:
    milestones: list[Milestone]
    active_milestone_id: str | None
    source_tool_call_id: str | None
    reason: GoalUpdateReason
    milestone_switch: bool = False


@dataclass(frozen=True)
class IntrospectionNotice:
    content: str
    active_milestone: Milestone | None
    step_count: int
    trigger_reason: IntrospectionTriggerReason


@dataclass(frozen=True)
class ContentUpdate:
    step_id: str
    tool_call_id: str
    content: str


@dataclass
class ContextRepairPlan:
    mode: ContextRepairMode
    start_seq: int
    end_seq: int
    experience: str
    content_updates: list[ContentUpdate] = field(default_factory=list)
    notice_cleaned_step_ids: list[str] = field(default_factory=list)

    @property
    def affected_count(self) -> int:
        return len(self.content_updates)

    @property
    def condensed_step_ids(self) -> list[str]:
        return [update.step_id for update in self.content_updates]


@dataclass
class IntrospectionOutcome:
    aligned: bool | None
    mode: IntrospectionMode
    boundary_seq: int
    experience: str | None = None
    active_milestone_id: str | None = None
    review_tool_call_id: str | None = None
    review_step_id: str | None = None
    hidden_step_ids: list[str] = field(default_factory=list)
    repair_plan: ContextRepairPlan | None = None


__all__ = [
    "ContentUpdate",
    "ContextRepairMode",
    "ContextRepairPlan",
    "GoalState",
    "GoalUpdate",
    "GoalUpdateReason",
    "IntrospectionCheckpoint",
    "IntrospectionMode",
    "IntrospectionNotice",
    "IntrospectionOutcome",
    "IntrospectionState",
    "IntrospectionTriggerReason",
    "Milestone",
    "MilestoneStatus",
    "PendingIntrospectionNotice",
]
