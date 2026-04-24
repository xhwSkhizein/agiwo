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
from agiwo.agent.review.review_enforcer import (
    ReviewTrigger,
    check_review_trigger,
    inject_system_review,
)
from agiwo.agent.review.step_back_executor import (
    StepBackOutcome,
    execute_step_back,
)

__all__ = [
    "ReviewBatch",
    "StepBackOutcome",
    "ReviewTrigger",
    "activate_next_milestone",
    "check_review_trigger",
    "complete_active_milestone",
    "declare_milestones",
    "execute_step_back",
    "get_active_milestone",
    "inject_system_review",
]
