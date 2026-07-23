"""System reports and completion snapshots for run-blocking faults."""

from typing import Any

from agiwo.agent.models.finalization import (
    RunFinalizationResult,
    fault_result,
)
from agiwo.agent.retry.faults import (
    FaultDisposition,
    RunBlockingFaultError,
)


def build_fault_report(
    error: RunBlockingFaultError,
    *,
    carry_forward: list[dict[str, Any]] | None = None,
    plan_items: list[dict[str, Any]] | None = None,
) -> str:
    """Build a system provenance report from fault evidence (no LLM)."""
    fault = error.fault
    lines = [
        f"[system:{_provenance_tag(error)}]",
        f"operation={fault.operation}",
        f"disposition={fault.disposition.value}",
        f"run_blocking={fault.run_blocking}",
        f"response_observed={fault.response_observed}",
        f"external_effect_may_have_started={fault.external_effect_may_have_started}",
    ]
    if fault.provider_code:
        lines.append(f"provider_code={fault.provider_code}")
    if fault.tool_code:
        lines.append(f"tool_code={fault.tool_code}")
    if fault.message:
        lines.append(f"message={fault.message}")
    lines.append(f"attempts={len(error.attempts)}")
    for attempt in error.attempts:
        lines.append(
            f"- attempt={attempt.attempt_no} disposition={attempt.disposition.value} "
            f"code={attempt.provider_code or attempt.tool_code or '-'}"
        )
    if plan_items:
        lines.append("run_plan:")
        for item in plan_items:
            lines.append(
                f"- {item.get('id')}: {item.get('description')} [{item.get('status')}]"
            )
    if carry_forward:
        lines.append("carry_forward:")
        for item in carry_forward:
            lines.append(
                f"- {item.get('id')}: {item.get('description')} [{item.get('status')}]"
            )
    if fault.disposition is FaultDisposition.OUTCOME_UNKNOWN:
        lines.append(
            "verification_advice=Confirm whether the external side effect completed "
            "before repeating the same operation."
        )
    return "\n".join(lines)


def finalization_for_blocking_fault(
    error: RunBlockingFaultError,
    *,
    carry_forward: list[dict[str, Any]] | None = None,
    plan_items: list[dict[str, Any]] | None = None,
) -> RunFinalizationResult:
    """Map a blocking fault to a report snapshot (no cross-run routing)."""
    report = build_fault_report(
        error, carry_forward=carry_forward, plan_items=plan_items
    )
    fault = error.fault
    if fault.disposition is FaultDisposition.RETRYABLE and (
        error.exhausted or len(error.attempts) > 1
    ):
        return fault_result(
            report,
            reason="system_retry_exhausted",
            carry_forward=carry_forward,
        )
    if fault.disposition is FaultDisposition.OUTCOME_UNKNOWN:
        return fault_result(
            report,
            reason="system_outcome_unknown",
            carry_forward=carry_forward,
        )
    return fault_result(
        report,
        reason="system_non_retryable",
        carry_forward=carry_forward,
    )


def _provenance_tag(error: RunBlockingFaultError) -> str:
    if error.fault.disposition is FaultDisposition.OUTCOME_UNKNOWN:
        return "system_outcome_unknown"
    if error.exhausted or (
        error.fault.disposition is FaultDisposition.RETRYABLE
        and len(error.attempts) > 1
    ):
        return "system_retry_exhausted"
    return "system_non_retryable"


__all__ = ["build_fault_report", "finalization_for_blocking_fault"]
