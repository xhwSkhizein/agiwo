"""Goal-directed review — replaces token/round-based retrospect.

Public API consumed by ``run_tool_batch.py``:

* ``ReviewBatch`` — per-batch lifecycle object
* ``StepBackOutcome`` — result of a step-back pass
"""

from agiwo.agent.review.goal_manager import (
    activate_next_milestone,
    complete_active_milestone,
    declare_milestones,
    get_active_milestone,
)

__all__ = [
    "ReviewBatch",
    "StepBackOutcome",
    "activate_next_milestone",
    "complete_active_milestone",
    "declare_milestones",
    "get_active_milestone",
]
