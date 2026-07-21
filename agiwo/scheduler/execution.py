"""Scheduler mechanical execution DTOs for Session root dispatch."""

from dataclasses import dataclass

from agiwo.agent import RunStatus
from agiwo.agent.models.execution import RunTreeRole, RunExecutionRequest
from agiwo.agent.models.input import UserInput


@dataclass(frozen=True, slots=True)
class SchedulerExecutionRequest:
    """Root dispatch request with preallocated Run identity.

    Scheduler validates IDs/state and forwards into Agent. The live ``Agent``
    instance is passed separately to ``Scheduler.dispatch_execution``.
    """

    state_id: str
    session_id: str
    user_input: UserInput | None
    execution: RunExecutionRequest
    persistent: bool = True
    agent_config_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionDispatchResult:
    """Result of ``Scheduler.dispatch_execution``."""

    state_id: str
    run_id: str
    attached: bool
    """True when an already-running handle was reused."""
    status: RunStatus | None = None


@dataclass(frozen=True, slots=True)
class ExecutionTreeNode:
    run_id: str
    agent_id: str
    status: RunStatus
    parent_run_id: str | None = None
    run_tree_role: RunTreeRole = RunTreeRole.NONE
    depth: int = 0


class SchedulerCapabilityUnavailable(Exception):
    """Facade method defined for later phases but not implemented yet."""

    def __init__(self, method: str, reason: str = "not_implemented") -> None:
        self.method = method
        self.reason = reason
        super().__init__(f"Scheduler.{method} is unavailable: {reason}")


__all__ = [
    "ExecutionDispatchResult",
    "ExecutionTreeNode",
    "SchedulerCapabilityUnavailable",
    "SchedulerExecutionRequest",
]
