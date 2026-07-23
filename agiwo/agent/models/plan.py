"""Run-scoped plan models (RunPlan / Milestone)."""

from dataclasses import dataclass, field
from typing import Literal

MilestoneStatus = Literal["pending", "active", "completed", "abandoned"]
RunPlanUpdateReason = Literal["declared", "updated", "completed", "activated"]


@dataclass
class Milestone:
    """A verifiable sub-goal in the current RunPlan."""

    id: str
    description: str
    status: MilestoneStatus = "pending"
    declared_at_seq: int = 0
    completed_at_seq: int | None = None


@dataclass
class RunPlan:
    """Work plan for the current Run. Active focus is derived from status."""

    milestones: list[Milestone] = field(default_factory=list)
    revision: int = 0

    @property
    def active_milestone_id(self) -> str | None:
        for milestone in self.milestones:
            if milestone.status == "active":
                return milestone.id
        return None

    @property
    def active_milestone(self) -> Milestone | None:
        active_id = self.active_milestone_id
        if active_id is None:
            return None
        for milestone in self.milestones:
            if milestone.id == active_id:
                return milestone
        return None

    def status_counts(self) -> dict[str, int]:
        counts = {
            "pending": 0,
            "active": 0,
            "completed": 0,
            "abandoned": 0,
        }
        for milestone in self.milestones:
            counts[milestone.status] = counts.get(milestone.status, 0) + 1
        return counts


@dataclass(frozen=True)
class RunPlanUpdate:
    milestones: list[Milestone]
    active_milestone_id: str | None
    source_tool_call_id: str | None
    reason: RunPlanUpdateReason
    revision: int
    milestone_switch: bool = False


__all__ = [
    "Milestone",
    "MilestoneStatus",
    "RunPlan",
    "RunPlanUpdate",
    "RunPlanUpdateReason",
]
