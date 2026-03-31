"""Scheduler control request models."""

from typing import Any

from pydantic import BaseModel, Field


class SteerRequest(BaseModel):
    message: str
    urgent: bool = False


class CancelRequest(BaseModel):
    reason: str = "Cancelled by operator"


class ResumeRequest(BaseModel):
    message: str


class CreateAgentRequest(BaseModel):
    agent_config_id: str | None = None
    initial_task: str | None = None
    session_id: str | None = None


class SchedulerChatCancelRequest(BaseModel):
    state_id: str


class PendingEventResponse(BaseModel):
    id: str
    target_agent_id: str
    source_agent_id: str | None = None
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
