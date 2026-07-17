"""Data models for trajectory introspection."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from agiwo.agent.models.plan import Milestone

IntrospectionTriggerReason = Literal[
    "step_interval", "consecutive_errors", "milestone_switch"
]


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
    latest_tool_usefulness: list["ToolUsefulnessEntry"] = field(default_factory=list)


@dataclass(frozen=True)
class IntrospectionNotice:
    content: str
    active_milestone: Milestone | None
    step_count: int
    trigger_reason: IntrospectionTriggerReason


@dataclass(frozen=True)
class ToolUsefulnessEntry:
    tool_call_id: str
    tool_name: str | None = None
    score: int | None = None


@dataclass
class IntrospectionOutcome:
    aligned: bool | None
    boundary_seq: int
    experience: str | None = None
    tool_usefulness: list[ToolUsefulnessEntry] = field(default_factory=list)
    active_milestone_id: str | None = None
    review_tool_call_id: str | None = None
    review_step_id: str | None = None


__all__ = [
    "IntrospectionCheckpoint",
    "IntrospectionNotice",
    "IntrospectionOutcome",
    "IntrospectionState",
    "IntrospectionTriggerReason",
    "Milestone",
    "PendingIntrospectionNotice",
    "ToolUsefulnessEntry",
]
