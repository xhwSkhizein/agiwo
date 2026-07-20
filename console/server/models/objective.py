"""Console API DTOs for the Objective gateway (P5-01..P5-06)."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agiwo.objective import BudgetLimits


class BudgetLimitsBody(BaseModel):
    handoffs: float = Field(gt=0)
    verification_attempts: float = Field(gt=0)
    llm_cost_usd: float = Field(gt=0)
    active_seconds: float = Field(gt=0)

    def to_budget_limits(self) -> BudgetLimits:
        return BudgetLimits(
            handoffs=self.handoffs,
            verification_attempts=self.verification_attempts,
            llm_cost_usd=self.llm_cost_usd,
            active_seconds=self.active_seconds,
        )


class CreateObjectiveBody(BaseModel):
    session_id: str
    message: str = ""
    parts: list[dict[str, Any]] | None = None
    budget: BudgetLimitsBody | None = None
    related_outcome_id: str | None = None

    @model_validator(mode="after")
    def _require_content(self) -> "CreateObjectiveBody":
        if not self.message and not self.parts:
            raise ValueError("message or parts is required")
        return self


class SubmitInputBody(BaseModel):
    message: str = ""
    parts: list[dict[str, Any]] | None = None
    in_reply_to_message_id: str | None = None
    related_outcome_id: str | None = None

    @model_validator(mode="after")
    def _require_content(self) -> "SubmitInputBody":
        if not self.message and not self.parts:
            raise ValueError("message or parts is required")
        return self


class ExternalizeBody(BaseModel):
    summary: str
    content_hash: str | None = None


class PauseBody(BaseModel):
    reason: str = "user_pause"


class ResumeBody(BaseModel):
    reason: str = "user_resume"


class AdjustBudgetBody(BaseModel):
    handoffs: float | None = Field(default=None, gt=0)
    verification_attempts: float | None = Field(default=None, gt=0)
    llm_cost_usd: float | None = Field(default=None, gt=0)
    active_seconds: float | None = Field(default=None, gt=0)


class CommandResultResponse(BaseModel):
    objective_id: str
    status: str
    replayed: bool = False
    payload: dict[str, Any] = Field(default_factory=dict)


class TimelineNodeResponse(BaseModel):
    sequence: int
    fact_id: str
    kind: str
    occurred_at: str
    summary: str
    refs: dict[str, Any] = Field(default_factory=dict)


class ObjectiveEventResponse(BaseModel):
    sequence: int
    fact_id: str
    kind: str
    occurred_at: str
    summary: str
    refs: dict[str, Any] = Field(default_factory=dict)


class BudgetDimensionResponse(BaseModel):
    limit: float
    used: float
    remaining: float


class ObjectiveBudgetResponse(BaseModel):
    handoffs: BudgetDimensionResponse
    verification_attempts: BudgetDimensionResponse
    llm_cost_usd: BudgetDimensionResponse
    active_seconds: BudgetDimensionResponse


class RootRunViewResponse(BaseModel):
    run_id: str
    role: str
    status: str
    run_ids: list[str] = Field(default_factory=list)
    outcome_report: str | None = None
    decision_target: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ArtifactSummaryResponse(BaseModel):
    artifact_id: str
    path: str
    summary: str


class ObjectiveViewResponse(BaseModel):
    objective_id: str
    session_id: str
    status: str
    is_terminal: bool
    budget: ObjectiveBudgetResponse
    timeline: list[TimelineNodeResponse] = Field(default_factory=list)
    root_runs: list[RootRunViewResponse] = Field(default_factory=list)
    delivery_report: str | None = None
    delivery_outcome_id: str | None = None
    artifacts: list[ArtifactSummaryResponse] = Field(default_factory=list)
    context_capacity: dict[str, Any] | None = None
    last_sequence: int = 0
    created_at: str | None = None
    updated_at: str | None = None


TemplateKind = Literal["work", "verification"]


class TemplateSetBody(BaseModel):
    work: str
    verification: str


class TemplateSetResponse(BaseModel):
    work: str
    verification: str


class TemplatePreviewBody(BaseModel):
    kind: TemplateKind
    template: str


class TemplatePreviewResponse(BaseModel):
    rendered: str


__all__ = [
    "AdjustBudgetBody",
    "ArtifactSummaryResponse",
    "RootRunViewResponse",
    "BudgetDimensionResponse",
    "BudgetLimitsBody",
    "CommandResultResponse",
    "CreateObjectiveBody",
    "ExternalizeBody",
    "ObjectiveBudgetResponse",
    "ObjectiveEventResponse",
    "ObjectiveViewResponse",
    "PauseBody",
    "ResumeBody",
    "SubmitInputBody",
    "TemplateKind",
    "TemplatePreviewBody",
    "TemplatePreviewResponse",
    "TemplateSetBody",
    "TemplateSetResponse",
    "TimelineNodeResponse",
]
