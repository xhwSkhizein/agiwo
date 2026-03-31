"""Scheduler state response models."""

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from agiwo.agent import UserInput
from server.models.metrics import RunMetricsSummary

if TYPE_CHECKING:
    from agiwo.scheduler.models import AgentState


class WakeConditionResponse(BaseModel):
    type: str
    wait_for: list[str] = Field(default_factory=list)
    wait_mode: str = "all"
    completed_ids: list[str] = Field(default_factory=list)
    time_value: float | None = None
    time_unit: str | None = None
    wakeup_at: str | None = None
    timeout_at: str | None = None


class AgentStateBase(BaseModel):
    id: str
    status: str
    task: UserInput
    parent_id: str | None = None
    wake_condition: WakeConditionResponse | None = None
    result_summary: str | None = None
    agent_config_id: str | None = None
    is_persistent: bool = False
    depth: int = 0
    wake_count: int = 0
    metrics: RunMetricsSummary = Field(default_factory=RunMetricsSummary)
    created_at: str | None = None
    updated_at: str | None = None

    @field_validator("task", mode="before")
    @classmethod
    def _extract_content_parts(cls, v: object) -> object:
        if isinstance(v, str):
            try:
                data = json.loads(v)
                if isinstance(data, dict) and data.get("__type") == "content_parts":
                    return data.get("parts", [])
                return data
            except (json.JSONDecodeError, TypeError):
                return v
        return v


class AgentStateResponse(AgentStateBase):
    session_id: str
    pending_input: UserInput | None = None
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    signal_propagated: bool = False

    @field_validator("pending_input", mode="before")
    @classmethod
    def _extract_pending_input(cls, v: object) -> object:
        if isinstance(v, str):
            try:
                data = json.loads(v)
                if isinstance(data, dict) and data.get("__type") == "content_parts":
                    return data.get("parts", [])
                return data
            except (json.JSONDecodeError, TypeError):
                return v
        return v

    @classmethod
    def from_sdk(cls, state: "AgentState") -> "AgentStateResponse":
        wake_condition = None
        if hasattr(state, "wake_condition") and state.wake_condition is not None:
            wc = state.wake_condition
            wakeup_at = getattr(wc, "wakeup_at", None)
            timeout_at = getattr(wc, "timeout_at", None)
            if wakeup_at is not None and hasattr(wakeup_at, "isoformat"):
                wakeup_at = wakeup_at.isoformat()
            if timeout_at is not None and hasattr(timeout_at, "isoformat"):
                timeout_at = timeout_at.isoformat()
            wake_condition = WakeConditionResponse(
                type=wc.type.value if hasattr(wc.type, "value") else str(wc.type),
                wait_for=list(getattr(wc, "wait_for", []) or []),
                wait_mode=wc.wait_mode.value
                if hasattr(wc.wait_mode, "value")
                else str(getattr(wc, "wait_mode", "all")),
                completed_ids=list(getattr(wc, "completed_ids", []) or []),
                time_value=getattr(wc, "time_value", None),
                time_unit=getattr(wc, "time_unit", None),
                wakeup_at=wakeup_at,
                timeout_at=timeout_at,
            )

        base_dict = {
            "id": state.id,
            "status": state.status.value
            if hasattr(state.status, "value")
            else str(state.status),
            "task": state.task,
            "parent_id": state.parent_id,
            "result_summary": state.result_summary,
            "agent_config_id": state.agent_config_id,
            "is_persistent": state.is_persistent,
            "depth": state.depth,
            "wake_count": state.wake_count,
            "metrics": getattr(state, "metrics", None) or RunMetricsSummary(),
            "created_at": state.created_at.isoformat() if state.created_at else None,
            "updated_at": state.updated_at.isoformat() if state.updated_at else None,
            "wake_condition": wake_condition,
        }
        return cls(
            **base_dict,
            session_id=state.session_id,
            pending_input=state.pending_input,
            config_overrides=state.config_overrides,
            signal_propagated=state.signal_propagated,
        )


class AgentStateListItem(AgentStateBase):
    @classmethod
    def from_sdk(cls, state: "AgentState") -> "AgentStateListItem":
        return cls(
            id=state.id,
            status=state.status.value
            if hasattr(state.status, "value")
            else str(state.status),
            task=state.task,
            parent_id=state.parent_id,
            result_summary=state.result_summary,
            agent_config_id=state.agent_config_id,
            is_persistent=state.is_persistent,
            depth=state.depth,
            wake_count=state.wake_count,
            metrics=getattr(state, "metrics", None) or RunMetricsSummary(),
            created_at=state.created_at.isoformat() if state.created_at else None,
            updated_at=state.updated_at.isoformat() if state.updated_at else None,
        )


class SchedulerStatsResponse(BaseModel):
    total: int
    pending: int
    running: int
    waiting: int
    idle: int
    queued: int
    completed: int
    failed: int
