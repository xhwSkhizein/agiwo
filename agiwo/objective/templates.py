"""Run input templates and their safe renderer."""

import hashlib
import string
from dataclasses import dataclass
from typing import Any

from agiwo.objective.errors import ValidationError


ALLOWED_RUN_TEMPLATE_PLACEHOLDERS = frozenset(
    {
        "current_goal",
        "objective_contributions",
        "objective_budget",
        "run_outcomes",
        "run_role",
    }
)
_TEMPLATE_ROLES = ("work", "verification")


def _template_placeholders(template: str) -> set[str]:
    """Return placeholders after enforcing the deliberately small format grammar."""
    if not isinstance(template, str):
        raise ValidationError("Run template must be a string")

    try:
        parsed = string.Formatter().parse(template)
    except ValueError as exc:
        raise ValidationError(f"Invalid run template: {exc}") from exc

    placeholders: set[str] = set()
    for _, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if (
            field_name not in ALLOWED_RUN_TEMPLATE_PLACEHOLDERS
            or format_spec
            or conversion
        ):
            raise ValidationError(
                f"Unsupported run template placeholder: {field_name!r}"
            )
        placeholders.add(field_name)
    return placeholders


def _validate_template(template: str, *, role: str) -> None:
    placeholders = _template_placeholders(template)
    missing = ALLOWED_RUN_TEMPLATE_PLACEHOLDERS - placeholders
    if missing:
        raise ValidationError(
            f"{role} run template is missing required placeholders: "
            f"{', '.join(sorted(missing))}"
        )


@dataclass(frozen=True)
class RunTemplateSet:
    """The fixed input templates used for each Run role."""

    work: str
    verification: str

    def __post_init__(self) -> None:
        _validate_template(self.work, role="work")
        _validate_template(self.verification, role="verification")

    def to_dict(self) -> dict[str, str]:
        return {
            "work": self.work,
            "verification": self.verification,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunTemplateSet":
        if not isinstance(data, dict):
            raise ValidationError("Run templates must be an object")

        keys = set(data)
        expected = set(_TEMPLATE_ROLES)
        unknown = keys - expected
        missing = expected - keys
        if unknown:
            raise ValidationError(
                f"Unknown run template roles: {', '.join(sorted(unknown))}"
            )
        if missing:
            raise ValidationError(
                f"Missing run template roles: {', '.join(sorted(missing))}"
            )
        return cls(
            work=data["work"],
            verification=data["verification"],
        )


# Back-compat alias for Console imports during transition.
AssignmentTemplateSet = RunTemplateSet


def default_run_templates() -> RunTemplateSet:
    """Build the standard responsibility-focused Run templates."""
    return RunTemplateSet(
        work=(
            "Run role: {run_role}\n\n"
            "Advance the Objective within this root Run's responsibility. Use the "
            "current goal, accumulated contributions, budget, and prior outcomes to "
            "choose and complete the next action.\n\n"
            "Routing guidance:\n"
            "- When verification is required, finalize with decision.target=verifier.\n"
            "- When verification is not required, you may deliver with "
            "decision.target=user and expects_reply=false.\n"
            "- If blocked on a user answer, finalize with target=user and "
            "expects_reply=true.\n\n"
            "Current goal:\n{current_goal}\n\n"
            "Objective contributions:\n{objective_contributions}\n\n"
            "Objective budget:\n{objective_budget}\n\n"
            "Prior run outcomes:\n{run_outcomes}"
        ),
        verification=(
            "Run role: {run_role}\n\n"
            "Verify whether the Objective has been satisfied. Inspect the current "
            "goal, available contributions, remaining budget, and prior outcomes; "
            "report evidence and the appropriate next responsibility.\n\n"
            "Routing guidance:\n"
            "- Accept and deliver with decision.target=user and "
            "expects_reply=false.\n"
            "- Reject and request more work with decision.target=agent.\n\n"
            "Current goal:\n{current_goal}\n\n"
            "Objective contributions:\n{objective_contributions}\n\n"
            "Objective budget:\n{objective_budget}\n\n"
            "Prior run outcomes:\n{run_outcomes}"
        ),
    )


def default_assignment_templates() -> RunTemplateSet:
    return default_run_templates()


def render_run_template(template: str, context: dict[str, Any]) -> str:
    """Safely substitute the fixed placeholder set into a Run template."""
    placeholders = _template_placeholders(template)
    missing = placeholders - set(context)
    if missing:
        raise ValidationError(
            f"Run template context is missing values: {', '.join(sorted(missing))}"
        )
    return template.format(**{name: context[name] for name in placeholders})


def render_assignment_template(template: str, context: dict[str, Any]) -> str:
    return render_run_template(template, context)


def template_content_hash(template: str) -> str:
    """Return the stable SHA-256 identifier for template content."""
    if not isinstance(template, str):
        raise ValidationError("Run template must be a string")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


__all__ = [
    "ALLOWED_RUN_TEMPLATE_PLACEHOLDERS",
    "AssignmentTemplateSet",
    "RunTemplateSet",
    "default_assignment_templates",
    "default_run_templates",
    "render_assignment_template",
    "render_run_template",
    "template_content_hash",
]
