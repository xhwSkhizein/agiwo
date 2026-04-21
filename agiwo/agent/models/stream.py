"""Public agent stream event payloads."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CommittedStep,
    CompactionApplied,
    MessagesRebuilt,
    RetrospectApplied,
    RunFailed,
    RunFinished,
    RunLogEntry,
    RunStarted,
    TerminationDecided,
    ToolStepCommitted,
    UserStepCommitted,
)
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
class MessagesRebuiltEvent(AgentStreamItemBase):
    reason: str
    message_count: int
    type: Literal["messages_rebuilt"] = "messages_rebuilt"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["reason"] = self.reason
        payload["message_count"] = self.message_count
        return payload


@dataclass(kw_only=True)
class CompactionAppliedEvent(AgentStreamItemBase):
    start_sequence: int
    end_sequence: int
    transcript_path: str
    summary: str | None = None
    type: Literal["compaction_applied"] = "compaction_applied"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["start_sequence"] = self.start_sequence
        payload["end_sequence"] = self.end_sequence
        payload["transcript_path"] = self.transcript_path
        payload["summary"] = self.summary
        return payload


@dataclass(kw_only=True)
class RetrospectAppliedEvent(AgentStreamItemBase):
    affected_sequences: list[int]
    affected_step_ids: list[str]
    feedback: str | None = None
    replacement: str | None = None
    trigger: str | None = None
    type: Literal["retrospect_applied"] = "retrospect_applied"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["affected_sequences"] = list(self.affected_sequences)
        payload["affected_step_ids"] = list(self.affected_step_ids)
        payload["feedback"] = self.feedback
        payload["replacement"] = self.replacement
        payload["trigger"] = self.trigger
        return payload


@dataclass(kw_only=True)
class TerminationDecidedEvent(AgentStreamItemBase):
    termination_reason: TerminationReason
    phase: str
    source: str
    type: Literal["termination_decided"] = "termination_decided"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["termination_reason"] = self.termination_reason.value
        payload["phase"] = self.phase
        payload["source"] = self.source
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


def _base_kwargs_from_entry(entry: RunLogEntry) -> dict[str, Any]:
    return {
        "session_id": entry.session_id,
        "run_id": entry.run_id,
        "agent_id": entry.agent_id,
        "parent_run_id": None,
        "depth": 0,
        "timestamp": entry.created_at,
    }


@dataclass(frozen=True)
class _ReplayContext:
    session_id: str
    run_id: str
    agent_id: str
    parent_run_id: str | None = None
    depth: int = 0


def _step_from_entry(entry: CommittedStep) -> StepRecord:
    ctx = _ReplayContext(
        session_id=entry.session_id,
        run_id=entry.run_id,
        agent_id=entry.agent_id,
        parent_run_id=entry.parent_run_id,
        depth=entry.depth,
    )
    if isinstance(entry, UserStepCommitted):
        step = StepRecord.user(
            ctx,
            sequence=entry.sequence,
            user_input=entry.user_input,
            content=entry.content,
            name=entry.name,
        )
    elif isinstance(entry, AssistantStepCommitted):
        step = StepRecord.assistant(
            ctx,
            sequence=entry.sequence,
            content=entry.content if isinstance(entry.content, str) else None,
            tool_calls=entry.tool_calls,
            reasoning_content=entry.reasoning_content,
            metrics=entry.metrics,
            name=entry.name,
        )
    else:
        step = StepRecord.tool(
            ctx,
            sequence=entry.sequence,
            tool_call_id=entry.tool_call_id or "",
            name=entry.name or "tool",
            content=entry.content if isinstance(entry.content, str) else "",
            content_for_user=entry.content_for_user,
            is_error=entry.is_error,
            metrics=entry.metrics,
        )
    step.id = entry.step_id
    step.created_at = entry.created_at
    step.condensed_content = entry.condensed_content
    if isinstance(entry, UserStepCommitted) and entry.user_input is None:
        step.content = entry.content
    if isinstance(entry, AssistantStepCommitted) and not isinstance(entry.content, str):
        step.content = entry.content
    if isinstance(entry, ToolStepCommitted) and not isinstance(entry.content, str):
        step.content = entry.content
    return step


def _run_metrics_from_dict(data: dict[str, Any] | None) -> RunMetrics | None:
    if data is None:
        return None
    return RunMetrics(**data)


def stream_items_from_entries(entries: list[RunLogEntry]) -> list["AgentStreamItem"]:
    items: list[AgentStreamItem] = []
    run_contexts: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, RunStarted):
            run_contexts[entry.run_id] = {
                "parent_run_id": entry.parent_run_id,
                "depth": entry.depth,
            }
            base_kwargs = _base_kwargs_from_entry(entry)
            base_kwargs["parent_run_id"] = entry.parent_run_id
            base_kwargs["depth"] = entry.depth
            items.append(RunStartedEvent(**base_kwargs))
        elif isinstance(
            entry,
            (UserStepCommitted, AssistantStepCommitted, ToolStepCommitted),
        ):
            base_kwargs = _base_kwargs_from_entry(entry)
            base_kwargs["parent_run_id"] = entry.parent_run_id
            base_kwargs["depth"] = entry.depth
            items.append(
                StepCompletedEvent(
                    **base_kwargs,
                    step=_step_from_entry(entry),
                )
            )
        elif isinstance(entry, MessagesRebuilt):
            base_kwargs = _base_kwargs_from_entry(entry)
            context = run_contexts.get(entry.run_id)
            if context is not None:
                base_kwargs.update(context)
            items.append(
                MessagesRebuiltEvent(
                    **base_kwargs,
                    reason=entry.reason,
                    message_count=len(entry.messages),
                )
            )
        elif isinstance(entry, CompactionApplied):
            base_kwargs = _base_kwargs_from_entry(entry)
            context = run_contexts.get(entry.run_id)
            if context is not None:
                base_kwargs.update(context)
            items.append(
                CompactionAppliedEvent(
                    **base_kwargs,
                    start_sequence=entry.start_sequence,
                    end_sequence=entry.end_sequence,
                    transcript_path=entry.transcript_path,
                    summary=entry.summary,
                )
            )
        elif isinstance(entry, RetrospectApplied):
            base_kwargs = _base_kwargs_from_entry(entry)
            context = run_contexts.get(entry.run_id)
            if context is not None:
                base_kwargs.update(context)
            items.append(
                RetrospectAppliedEvent(
                    **base_kwargs,
                    affected_sequences=list(entry.affected_sequences),
                    affected_step_ids=list(entry.affected_step_ids),
                    feedback=entry.feedback,
                    replacement=entry.replacement,
                    trigger=entry.trigger,
                )
            )
        elif isinstance(entry, TerminationDecided):
            base_kwargs = _base_kwargs_from_entry(entry)
            context = run_contexts.get(entry.run_id)
            if context is not None:
                base_kwargs.update(context)
            items.append(
                TerminationDecidedEvent(
                    **base_kwargs,
                    termination_reason=entry.termination_reason,
                    phase=entry.phase,
                    source=entry.source,
                )
            )
        elif isinstance(entry, RunFinished):
            base_kwargs = _base_kwargs_from_entry(entry)
            context = run_contexts.get(entry.run_id)
            if context is not None:
                base_kwargs.update(context)
            items.append(
                RunCompletedEvent(
                    **base_kwargs,
                    response=entry.response,
                    metrics=_run_metrics_from_dict(entry.metrics),
                    termination_reason=entry.termination_reason,
                )
            )
        elif isinstance(entry, RunFailed):
            base_kwargs = _base_kwargs_from_entry(entry)
            context = run_contexts.get(entry.run_id)
            if context is not None:
                base_kwargs.update(context)
            items.append(
                RunFailedEvent(
                    **base_kwargs,
                    error=entry.error,
                )
            )
    return items


AgentStreamItem: TypeAlias = (
    RunStartedEvent
    | StepDeltaEvent
    | StepCompletedEvent
    | MessagesRebuiltEvent
    | CompactionAppliedEvent
    | RetrospectAppliedEvent
    | TerminationDecidedEvent
    | RunCompletedEvent
    | RunFailedEvent
)


__all__ = [
    "AgentStreamItem",
    "AgentStreamItemBase",
    "CompactionAppliedEvent",
    "MessagesRebuiltEvent",
    "RetrospectAppliedEvent",
    "RunCompletedEvent",
    "RunFailedEvent",
    "RunStartedEvent",
    "StepCompletedEvent",
    "StepDeltaEvent",
    "TerminationDecidedEvent",
    "stream_items_from_entries",
]
