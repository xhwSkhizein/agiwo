"""Root-run finalization wire format and parsing.

Outcome ``report`` is always the last ordinary assistant text from the work
loop. The finalization model call only emits routing/structure fields — it
must not restate or invent the report.
"""

import json
from dataclasses import dataclass, field
from typing import Any


FINALIZATION_USER_INSTRUCTION = """\
Produce the root-run finalization now. Return exactly one JSON object and do
not call any tools. Do not include a report field — the ordinary assistant
text already spoken in this run is the Outcome report. The object must contain:
{
  "decision": {
    "target": "agent|verifier|user",
    "expects_reply": false
  },
  "new_contributions": [{"content": "text", "summary": "optional"}],
  "contribution_annotations": [
    {"contribution_id": "existing id", "annotation": "text", "deactivate": false}
  ],
  "objective_update": null,
  "artifact_refs": [{"path": "optional path", "summary": "optional"}],
  "carry_forward": [
    {"id": "milestone id", "description": "text", "status": "pending|active"}
  ]
}
Decision routing:
- work that fully answers a simple request: target=user, expects_reply=false
- work that needs another work run: target=agent (omit expects_reply)
- work that finished a proposal needing verification: target=verifier
  (omit expects_reply)
- work that must ask the user a blocking question: target=user,
  expects_reply=true
- verification that accepts delivery: target=user, expects_reply=false
- verification that rejects: target=agent (omit expects_reply)
When `decision.target` is `user`, `expects_reply` MUST be a boolean. Do not
include `expects_reply` for any other target."""

FINALIZATION_CORRECTION_INSTRUCTION = """\
Your prior finalization response was invalid. Return exactly one valid JSON
object matching the finalization schema. Do not call tools, do not include a
report field, and do not include Markdown fences or explanatory text."""


@dataclass
class RunFinalizationResult:
    """Validated root-run finalization; ``report`` is system-injected text."""

    report: str
    decision: dict[str, Any]
    new_contributions: list[dict[str, Any]] = field(default_factory=list)
    contribution_annotations: list[dict[str, Any]] = field(default_factory=list)
    objective_update: dict[str, Any] | None = None
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    carry_forward: list[dict[str, Any]] = field(default_factory=list)
    mechanical_handoff: bool = False
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
            "mechanical_handoff": self.mechanical_handoff,
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
            mechanical_handoff=bool(data.get("mechanical_handoff", False)),
            parse_error=data.get("parse_error"),
        )


def parse_finalization_json(
    text: str,
    *,
    report: str,
) -> RunFinalizationResult:
    """Parse finalization JSON and inject the work-loop ``report``."""
    try:
        payload = json.loads(_extract_json_object(text))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid finalization JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("finalization JSON must be an object")
    if "report" in payload:
        raise ValueError(
            "finalization must not include report; the work-loop text is authoritative"
        )

    decision = payload.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("finalization.decision must be an object")
    target = decision.get("target")
    if target not in {"agent", "verifier", "user"}:
        raise ValueError("finalization.decision.target is invalid")
    if target == "user":
        if "expects_reply" not in decision:
            # Prefer waiting over a mechanical agent handoff when the model
            # forgets the field; delivery still requires an explicit false.
            decision = {**decision, "expects_reply": True}
        elif not isinstance(decision.get("expects_reply"), bool):
            raise ValueError("user decision requires boolean expects_reply")
    elif "expects_reply" in decision:
        raise ValueError("only user decisions may include expects_reply")

    new_contributions = _list_of_objects(payload, "new_contributions")
    contribution_annotations = _list_of_objects(payload, "contribution_annotations")
    objective_update = _optional_object(payload, "objective_update")
    artifact_refs = _list_of_objects(payload, "artifact_refs")
    carry_forward = _list_of_objects(payload, "carry_forward")
    _validate_new_contributions(new_contributions)
    _validate_contribution_annotations(contribution_annotations)
    _validate_objective_update(objective_update)
    _validate_carry_forward(carry_forward)

    return RunFinalizationResult(
        report=report,
        decision=decision,
        new_contributions=new_contributions,
        contribution_annotations=contribution_annotations,
        objective_update=objective_update,
        artifact_refs=artifact_refs,
        carry_forward=carry_forward,
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
        mechanical_handoff=True,
        parse_error=reason,
    )


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
        mechanical_handoff=True,
        parse_error=reason,
    )


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    return text[start : end + 1]


def _list_of_objects(payload: dict[str, Any], field_name: str) -> list[dict[str, Any]]:
    value = payload.get(field_name, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"finalization.{field_name} must be a list of objects")
    return value


def _optional_object(
    payload: dict[str, Any],
    field_name: str,
) -> dict[str, Any] | None:
    value = payload.get(field_name)
    if value is not None and not isinstance(value, dict):
        raise ValueError(f"finalization.{field_name} must be an object or null")
    return value


def _validate_new_contributions(items: list[dict[str, Any]]) -> None:
    for item in items:
        if not isinstance(item.get("content"), str):
            raise ValueError("every new contribution requires string content")
        if "summary" in item and not isinstance(item["summary"], str):
            raise ValueError("new contribution summary must be a string")


def _validate_contribution_annotations(items: list[dict[str, Any]]) -> None:
    for item in items:
        if not isinstance(item.get("contribution_id"), str) or not isinstance(
            item.get("annotation"), str
        ):
            raise ValueError(
                "every contribution annotation requires contribution_id and annotation"
            )
        if "deactivate" in item and not isinstance(item["deactivate"], bool):
            raise ValueError("annotation deactivate must be a boolean")


def _validate_objective_update(update: dict[str, Any] | None) -> None:
    if update is None:
        return
    if not isinstance(update.get("expected_revision"), int):
        raise ValueError("objective_update requires integer expected_revision")


def _validate_carry_forward(items: list[dict[str, Any]]) -> None:
    for item in items:
        if (
            not isinstance(item.get("id"), str)
            or not isinstance(item.get("description"), str)
            or item.get("status") not in {"pending", "active"}
        ):
            raise ValueError(
                "carry_forward items require id, description, and pending or active status"
            )


__all__ = [
    "FINALIZATION_CORRECTION_INSTRUCTION",
    "FINALIZATION_USER_INSTRUCTION",
    "RunFinalizationResult",
    "mechanical_agent_handoff_result",
    "mechanical_user_boundary_result",
    "parse_finalization_json",
]
