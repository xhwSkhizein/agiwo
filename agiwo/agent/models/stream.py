"""Public agent stream event payloads."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from agiwo.agent.models.run import RunMetrics, TerminationReason
from agiwo.agent.models.step import StepDelta, StepRecord
from agiwo.utils.serialization import serialize_optional_datetime

if TYPE_CHECKING:
    from agiwo.agent.runtime.context import RunContext


@dataclass(kw_only=True)
class AgentStreamItemBase:
    """Base payload shared by all public agent stream items."""

    session_id: str
    run_id: str
    agent_id: str
    parent_run_id: str | None
    depth: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_context(cls, ctx: "RunContext", **kwargs: Any) -> "AgentStreamItemBase":
        return cls(
            session_id=ctx.session_id,
            run_id=ctx.run_id,
            agent_id=ctx.agent_id,
            parent_run_id=ctx.parent_run_id,
            depth=ctx.depth,
            **kwargs,
        )

    def _base_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,  # type: ignore[attr-defined]
            "session_id": self.session_id,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "parent_run_id": self.parent_run_id,
            "depth": self.depth,
            "timestamp": serialize_optional_datetime(self.timestamp),
        }

    def to_dict(self) -> dict[str, Any]:
        return self._base_dict()


@dataclass(kw_only=True)
class RunStartedEvent(AgentStreamItemBase):
    type: Literal["run_started"] = "run_started"


@dataclass(kw_only=True)
class StepDeltaEvent(AgentStreamItemBase):
    step_id: str
    delta: StepDelta
    type: Literal["step_delta"] = "step_delta"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["step_id"] = self.step_id
        payload["delta"] = self.delta.to_dict()
        return payload


@dataclass(kw_only=True)
class StepCompletedEvent(AgentStreamItemBase):
    step: StepRecord
    type: Literal["step_completed"] = "step_completed"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["step"] = self.step.to_dict()
        return payload


@dataclass(kw_only=True)
class RunCompletedEvent(AgentStreamItemBase):
    response: str | None = None
    metrics: RunMetrics | None = None
    termination_reason: TerminationReason | None = None
    type: Literal["run_completed"] = "run_completed"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["response"] = self.response
        payload["metrics"] = self.metrics.to_dict() if self.metrics else None
        payload["termination_reason"] = (
            self.termination_reason.value if self.termination_reason else None
        )
        return payload


@dataclass(kw_only=True)
class RunFailedEvent(AgentStreamItemBase):
    error: str
    type: Literal["run_failed"] = "run_failed"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["error"] = self.error
        return payload


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
