"""Root-run mechanical finalization (ADR 0047).

Outcome ``report`` is always the last ordinary assistant text from the work
loop. Routing is derived mechanically — no finalization LLM call.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunFinalizationResult:
    """Mechanically derived root-run finalization; ``report`` is work-loop text."""

    report: str
    decision: dict[str, Any]
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


def mechanical_agent_handoff_result(
    report: str,
    *,
    reason: str,
    carry_forward: list[dict[str, Any]] | None,
) -> RunFinalizationResult:
    """Produce the system-owned fallback without inferring a model decision."""
    return RunFinalizationResult(
        report=report,
        decision={"target": "agent", "reason": reason},
        carry_forward=list(carry_forward or []),
        parse_error=reason,
    )


def mechanical_user_delivery_result(
    report: str,
    *,
    reason: str = "simple_path_mechanical_delivery",
) -> RunFinalizationResult:
    """Deliver to the user without a finalization LLM call (ADR 0047)."""
    return RunFinalizationResult(
        report=report,
        decision={
            "target": "user",
            "expects_reply": False,
            "reason": reason,
        },
        parse_error=reason,
    )


def mechanical_verifier_handoff_result(
    report: str,
    *,
    reason: str = "verification_required_mechanical",
) -> RunFinalizationResult:
    """Hand off to a verification root Run (ADR 0047)."""
    return RunFinalizationResult(
        report=report,
        decision={"target": "verifier", "reason": reason},
        parse_error=reason,
    )


def derive_mechanical_finalization(
    *,
    report: str,
    verification_required: bool,
    objective_run_role: str | None,
) -> RunFinalizationResult:
    """Derive mechanical HandoffDecision without a finalization LLM (ADR 0047).

    Rules:
    - verification role → deliver
    - work with verification_required (or plan latch) → verify
    - otherwise → deliver
    """
    if objective_run_role == "verification":
        return mechanical_user_delivery_result(
            report, reason="verification_run_mechanical_delivery"
        )
    if verification_required:
        return mechanical_verifier_handoff_result(report)
    return mechanical_user_delivery_result(report)


def mechanical_user_boundary_result(
    report: str,
    *,
    reason: str,
    carry_forward: list[dict[str, Any]] | None = None,
) -> RunFinalizationResult:
    """System-owned handoff to the user (expects reply); does not consume handoffs."""
    return RunFinalizationResult(
        report=report,
        decision={
            "target": "user",
            "expects_reply": True,
            "reason": reason,
        },
        carry_forward=list(carry_forward or []),
        parse_error=reason,
    )


__all__ = [
    "RunFinalizationResult",
    "derive_mechanical_finalization",
    "mechanical_agent_handoff_result",
    "mechanical_user_boundary_result",
    "mechanical_user_delivery_result",
    "mechanical_verifier_handoff_result",
]
