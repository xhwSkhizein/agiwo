"""Internal execution request for Scheduler-owned and Session root runs.

Not a normal user API. Public ``Agent.start/run/run_stream`` keep generating
their own run ids; Scheduler passes a validated request instead.
"""

from dataclasses import dataclass
from enum import Enum


class RunTreeRole(str, Enum):
    """Role of this Run relative to a Session root / child execution tree."""

    NONE = "none"
    ROOT = "root"
    CHILD = "child"


@dataclass(frozen=True, slots=True)
class RunExecutionRequest:
    """Narrow internal request carrying preallocated Run identity.

    Root next-action is derived mechanically inside the Agent package —
    no finalization LLM.
    """

    run_id: str
    run_tree_role: RunTreeRole = RunTreeRole.NONE
    template_hash: str | None = None
    config_revision: str | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("RunExecutionRequest.run_id is required")


__all__ = [
    "RunTreeRole",
    "RunExecutionRequest",
]
