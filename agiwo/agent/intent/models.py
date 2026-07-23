"""SessionIntent data models (I-S1 compass, not Objective)."""

from dataclasses import dataclass, field
from typing import Literal

from agiwo.agent.models.plan import RunPlan

IntentEntryKind = Literal["user_input", "run_report"]


@dataclass
class IntentEntry:
    """One SessionIntent timeline item."""

    kind: IntentEntryKind
    text: str
    at: int
    run_id: str | None = None


@dataclass
class SessionIntent:
    """Cross-run alignment view for one Session."""

    entries: list[IntentEntry] = field(default_factory=list)
    last_run_plan: RunPlan | None = None
    updated_at: int = 0


__all__ = ["IntentEntry", "IntentEntryKind", "SessionIntent"]
