"""Run completion metadata (report-only; no cross-run routing).

ADR 0048: Session owns conversation. ``decision`` may carry a short reason
tag for fault/report metadata but is not a handoff protocol.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunFinalizationResult:
    """Work-loop completion snapshot attached to RunFinished when present."""

    report: str
    decision: dict[str, Any] = field(default_factory=dict)
    new_contributions: list[dict[str, Any]] = field(default_factory=list)
    contribution_annotations: list[dict[str, Any]] = field(default_factory=list)
    objective_update: dict[str, Any] | None = None
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    carry_forward: list[dict[str, Any]] = field(default_factory=list)
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "decision": dict(self.decision),
            "new_contributions": list(self.new_contributions),
            "contribution_annotations": list(self.contribution_annotations),
            "objective_update": self.objective_update,
            "artifact_refs": list(self.artifact_refs),
            "carry_forward": list(self.carry_forward),
            "parse_error": self.parse_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunFinalizationResult":
        return cls(
            report=str(data.get("report") or ""),
            decision=dict(data.get("decision") or {}),
            new_contributions=list(data.get("new_contributions") or []),
            contribution_annotations=list(data.get("contribution_annotations") or []),
            objective_update=data.get("objective_update"),
            artifact_refs=list(data.get("artifact_refs") or []),
            carry_forward=list(data.get("carry_forward") or []),
            parse_error=data.get("parse_error"),
        )


def completion_result(
    report: str,
    *,
    reason: str = "run_completed",
) -> RunFinalizationResult:
    """Mark a normal run completion with the last assistant report text."""
    return RunFinalizationResult(
        report=report,
        decision={"reason": reason},
        parse_error=reason,
    )


def fault_result(
    report: str,
    *,
    reason: str,
    carry_forward: list[dict[str, Any]] | None = None,
) -> RunFinalizationResult:
    """Attach a system fault/report snapshot (no cross-run routing)."""
    return RunFinalizationResult(
        report=report,
        decision={"reason": reason},
        carry_forward=list(carry_forward or []),
        parse_error=reason,
    )


# Names still imported by retry / run_loop.
mechanical_user_delivery_result = completion_result
mechanical_agent_handoff_result = fault_result
mechanical_user_boundary_result = fault_result


__all__ = [
    "RunFinalizationResult",
    "completion_result",
    "fault_result",
    "mechanical_agent_handoff_result",
    "mechanical_user_boundary_result",
    "mechanical_user_delivery_result",
]
