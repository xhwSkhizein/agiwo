"""Streaming protocol: AgentStreamItem event types."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, TypeAlias

from agiwo.agent.runtime.core import TerminationReason
from agiwo.agent.runtime.run import RunMetrics
from agiwo.agent.runtime.step import StepDelta, StepRecord
from agiwo.utils.tojson import to_json


@dataclass(kw_only=True)
class AgentStreamItemBase:
    """Base payload shared by all public agent stream items."""

    session_id: str
    run_id: str
    agent_id: str
    parent_run_id: str | None
    depth: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_sse(self) -> str:
        return f"data: {to_json(self)}\n\n"


@dataclass(kw_only=True)
class RunStartedEvent(AgentStreamItemBase):
    type: Literal["run_started"] = "run_started"


@dataclass(kw_only=True)
class StepDeltaEvent(AgentStreamItemBase):
    step_id: str
    delta: StepDelta
    type: Literal["step_delta"] = "step_delta"


@dataclass(kw_only=True)
class StepCompletedEvent(AgentStreamItemBase):
    step: StepRecord
    type: Literal["step_completed"] = "step_completed"


@dataclass(kw_only=True)
class RunCompletedEvent(AgentStreamItemBase):
    response: str | None = None
    metrics: RunMetrics | None = None
    termination_reason: TerminationReason | None = None
    type: Literal["run_completed"] = "run_completed"


@dataclass(kw_only=True)
class RunFailedEvent(AgentStreamItemBase):
    error: str
    type: Literal["run_failed"] = "run_failed"


AgentStreamItem: TypeAlias = (
    RunStartedEvent
    | StepDeltaEvent
    | StepCompletedEvent
    | RunCompletedEvent
    | RunFailedEvent
)


__all__ = [
    "AgentStreamItem",
    "AgentStreamItemBase",
    "RunCompletedEvent",
    "RunFailedEvent",
    "RunStartedEvent",
    "StepCompletedEvent",
    "StepDeltaEvent",
]
