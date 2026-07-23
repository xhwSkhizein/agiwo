"""SessionIntent data models (I-S1 compass, not Objective)."""

from dataclasses import dataclass, field
from typing import Literal

from agiwo.agent.models.plan import RunPlan

IntentEntryKind = Literal["user_input", "run_report"]

# Compass retention: keep the newest N entries on read; lazy physical
# cleanup kicks in after ~2N appends (see SessionIntentStore).
MAX_INTENT_ENTRIES = 200


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


__all__ = ["IntentEntry", "IntentEntryKind", "MAX_INTENT_ENTRIES", "SessionIntent"]
