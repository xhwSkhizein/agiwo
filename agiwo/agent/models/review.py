"""Review and milestone data models for goal-directed review."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class Milestone:
    """A verifiable sub-goal declared by the agent."""

    id: str
    description: str
    status: Literal["pending", "active", "completed", "abandoned"] = "pending"
    declared_at_seq: int = 0
    completed_at_seq: int | None = None


@dataclass
class ReviewCheckpoint:
    """A confirmed-aligned checkpoint recorded after a successful review."""

    seq: int
    milestone_id: str
    confirmed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ReviewState:
    """Per-run review tracking state, stored on RunLedger."""

    milestones: list[Milestone] = field(default_factory=list)
    last_review_seq: int = 0
    latest_checkpoint: ReviewCheckpoint | None = None
    consecutive_errors: int = 0
    pending_review_reason: Literal["milestone_switch"] | None = None


__all__ = [
    "Milestone",
    "ReviewCheckpoint",
    "ReviewState",
]
