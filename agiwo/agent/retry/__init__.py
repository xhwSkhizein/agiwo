"""Structured retry contract for agent model/tool attempts."""

from agiwo.agent.retry.coordinator import RetryCoordinator, RetryPolicy
from agiwo.agent.retry.faults import (
    ExecutionFault,
    FaultDisposition,
    IdempotencyKind,
    RunBlockingFaultError,
    may_auto_retry,
)
from agiwo.agent.retry.provider import map_provider_exception
from agiwo.agent.retry.reports import (
    build_fault_report,
    finalization_for_blocking_fault,
)

__all__ = [
    "ExecutionFault",
    "FaultDisposition",
    "IdempotencyKind",
    "RetryCoordinator",
    "RetryPolicy",
    "RunBlockingFaultError",
    "build_fault_report",
    "finalization_for_blocking_fault",
    "map_provider_exception",
    "may_auto_retry",
]
