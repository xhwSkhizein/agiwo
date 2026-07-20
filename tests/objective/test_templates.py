"""Run template tests."""

import pytest

from agiwo.objective import (
    AssignmentTemplateSet,
    ValidationError,
    default_assignment_templates,
    render_assignment_template,
)


def test_unknown_placeholder_fails() -> None:
    with pytest.raises(ValidationError, match="Unsupported"):
        AssignmentTemplateSet(
            work="{current_goal} {objective_contributions} {objective_budget} "
            "{run_outcomes} {run_role} {unknown}",
            verification="{current_goal} {objective_contributions} {objective_budget} "
            "{run_outcomes} {run_role}",
        )


def test_missing_required_placeholder_fails() -> None:
    with pytest.raises(ValidationError, match="missing required"):
        AssignmentTemplateSet(
            work="{current_goal} {objective_contributions} {objective_budget} "
            "{run_role}",
            verification="{current_goal} {objective_contributions} {objective_budget} "
            "{run_outcomes} {run_role}",
        )


def test_default_templates_render_with_sample_context() -> None:
    context = {
        "current_goal": "Ship the feature",
        "objective_contributions": "A design is ready.",
        "objective_budget": "Two remaining handoffs.",
        "run_outcomes": "No prior runs.",
        "run_role": "work",
    }

    for template in default_assignment_templates().to_dict().values():
        rendered = render_assignment_template(template, context)
        assert "Ship the feature" in rendered
        assert "{current_goal}" not in rendered


def test_template_set_round_trips_through_dict() -> None:
    templates = default_assignment_templates()

    assert AssignmentTemplateSet.from_dict(templates.to_dict()) == templates
