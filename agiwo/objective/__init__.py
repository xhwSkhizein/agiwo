"""Objective deep module: cross-Assignment lifecycle control.

Public surface is intentionally narrow. Import from ``agiwo.objective`` only.
"""

from agiwo.objective.errors import (
    BudgetBoundaryHit,
    CommandUnavailable,
    IdempotencyConflict,
    InvariantViolation,
    ObjectiveError,
    ValidationError,
)
from agiwo.objective.models import (
    AdjustBudgetRequest,
    BudgetLimits,
    CommandResult,
    CreateObjectiveRequest,
    ExternalizeUserInputRequest,
    ObjectiveStatus,
    PauseObjectiveRequest,
    ResumeObjectiveRequest,
    SubmitUserInputRequest,
)
from agiwo.objective.metrics import ObjectiveMetrics, project_objective_metrics
from agiwo.objective.projection import ObjectiveView
from agiwo.objective.service import ObjectiveService
from agiwo.objective.store import ObjectiveStore, create_objective_store
from agiwo.objective.templates import (
    AssignmentTemplateSet,
    default_assignment_templates,
    render_assignment_template,
    template_content_hash,
)

__all__ = [
    "AdjustBudgetRequest",
    "AssignmentTemplateSet",
    "BudgetBoundaryHit",
    "BudgetLimits",
    "CommandResult",
    "CommandUnavailable",
    "CreateObjectiveRequest",
    "ExternalizeUserInputRequest",
    "IdempotencyConflict",
    "InvariantViolation",
    "ObjectiveError",
    "ObjectiveMetrics",
    "ObjectiveService",
    "ObjectiveStatus",
    "ObjectiveStore",
    "ObjectiveView",
    "PauseObjectiveRequest",
    "ResumeObjectiveRequest",
    "SubmitUserInputRequest",
    "ValidationError",
    "create_objective_store",
    "default_assignment_templates",
    "project_objective_metrics",
    "render_assignment_template",
    "template_content_hash",
]
