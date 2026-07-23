"""Scheduler mechanical execution DTOs."""

from dataclasses import dataclass

from agiwo.agent import RunStatus
from agiwo.agent.models.execution import RunTreeRole


@dataclass(frozen=True, slots=True)
class ExecutionTreeNode:
    run_id: str
    agent_id: str
    status: RunStatus
    parent_run_id: str | None = None
    run_tree_role: RunTreeRole = RunTreeRole.NONE
    depth: int = 0


__all__ = [
    "ExecutionTreeNode",
]
