"""Objective-facing helpers for structured fault boundaries (re-exports agent reports)."""

from agiwo.agent.retry.reports import (
    build_fault_report,
    finalization_for_blocking_fault,
)

__all__ = [
    "build_fault_report",
    "finalization_for_blocking_fault",
]
