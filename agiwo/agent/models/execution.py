"""Internal execution request for Objective-managed and Scheduler-owned runs.

Not a normal user API. Public ``Agent.start/run/run_stream`` keep generating
their own run ids; Scheduler/Objective pass a validated request instead.
"""

from dataclasses import dataclass
from enum import Enum


class RunTreeRole(str, Enum):
    """Role of this Run relative to an Objective-managed root/child tree."""

    NONE = "none"
    ROOT = "root"
    CHILD = "child"


@dataclass(frozen=True, slots=True)
class RunExecutionRequest:
    """Narrow internal request carrying preallocated Run identity.

    Does not carry FinalizationSpec — root finalization is triggered by
    ``run_tree_role=root`` inside the Agent package (ADR 0006 / P2-05).
    """

    run_id: str
    objective_id: str | None = None
    run_tree_role: RunTreeRole = RunTreeRole.NONE
    template_hash: str | None = None
    config_revision: str | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("RunExecutionRequest.run_id is required")
        if self.run_tree_role is RunTreeRole.ROOT and not self.objective_id:
            raise ValueError("run_tree_role=root requires objective_id")


__all__ = [
    "RunTreeRole",
    "RunExecutionRequest",
]
