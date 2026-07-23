"""Run completion metadata (report-only; no cross-run routing).

ADR 0048: Session owns conversation. Finalization is an optional snapshot of
the last report text plus a short reason tag — not a handoff protocol.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunFinalizationResult:
    """Work-loop completion snapshot attached to RunFinished when present."""

    report: str
    decision: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    carry_forward: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": self.report,
            "decision": dict(self.decision),
            "artifact_refs": list(self.artifact_refs),
            "carry_forward": list(self.carry_forward),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunFinalizationResult":
        # Ignore retired Objective-era keys if present in old RunLog rows.
        return cls(
            report=str(data.get("report") or ""),
            decision=dict(data.get("decision") or {}),
            artifact_refs=list(data.get("artifact_refs") or []),
            carry_forward=list(data.get("carry_forward") or []),
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
    )


__all__ = [
    "RunFinalizationResult",
    "completion_result",
    "fault_result",
]
