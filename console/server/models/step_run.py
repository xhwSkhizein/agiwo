"""Step and Run response models."""

from typing import TYPE_CHECKING

from pydantic import BaseModel

from agiwo.agent import UserInput
from server.models.metrics import RunMetricsResponse, StepMetricsResponse

if TYPE_CHECKING:
    from agiwo.agent import StepRecord


class StepResponse(BaseModel):
    id: str
    session_id: str
    run_id: str
    sequence: int
    role: str
    agent_id: str | None = None
    content: object | None = None
    content_for_user: str | None = None
    reasoning_content: str | None = None
    user_input: UserInput | None = None
    tool_calls: list[dict[str, object]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metrics: StepMetricsResponse | None = None
    created_at: str | None = None
    parent_run_id: str | None = None
    depth: int = 0

    @classmethod
    def from_sdk(cls, step: "StepRecord") -> "StepResponse":
        return cls(
            id=step.id,
            session_id=step.session_id,
            run_id=step.run_id,
            sequence=step.sequence,
            role=step.role.value,
            agent_id=step.agent_id,
            content=step.content,
            content_for_user=step.content_for_user,
            reasoning_content=step.reasoning_content,
            user_input=step.user_input,
            tool_calls=[tc.model_dump() for tc in step.tool_calls]
            if step.tool_calls
            else None,
            tool_call_id=step.tool_call_id,
            name=step.name,
            metrics=StepMetricsResponse.from_sdk(step.metrics)
            if step.metrics
            else None,
            created_at=step.created_at.isoformat() if step.created_at else None,
            parent_run_id=step.parent_run_id,
            depth=step.depth,
        )


class RunResponse(BaseModel):
    id: str
    agent_id: str
    session_id: str
    user_id: str | None = None
    user_input: UserInput
    status: str
    response_content: str | None = None
    metrics: RunMetricsResponse | None = None
    created_at: str | None = None
    updated_at: str | None = None
    parent_run_id: str | None = None
