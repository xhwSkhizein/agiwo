"""Run-log models for replayable agent execution facts."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agiwo.agent.models.input import MessageContent, UserInput
from agiwo.agent.models.step import MessageRole, StepMetrics
from agiwo.config.termination import TerminationReason


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RunLogEntryKind(str, Enum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"
    CONTEXT_ASSEMBLED = "context_assembled"
    MESSAGES_REBUILT = "messages_rebuilt"
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"
    USER_STEP_COMMITTED = "user_step_committed"
    ASSISTANT_STEP_COMMITTED = "assistant_step_committed"
    TOOL_STEP_COMMITTED = "tool_step_committed"
    COMPACTION_APPLIED = "compaction_applied"
    RETROSPECT_APPLIED = "retrospect_applied"
    TERMINATION_DECIDED = "termination_decided"
    HOOK_FAILED = "hook_failed"


@dataclass(frozen=True, kw_only=True)
class RunLogEntry:
    sequence: int
    session_id: str
    run_id: str
    agent_id: str
    created_at: datetime = field(default_factory=_now_utc)
    kind: RunLogEntryKind = field(init=False)


@dataclass(frozen=True, kw_only=True)
class RunStarted(RunLogEntry):
    user_input: UserInput | None = None
    user_id: str | None = None
    parent_run_id: str | None = None
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_STARTED)


@dataclass(frozen=True, kw_only=True)
class RunFinished(RunLogEntry):
    response: str | None = None
    termination_reason: TerminationReason | None = None
    metrics: dict[str, Any] | None = None
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_FINISHED)


@dataclass(frozen=True, kw_only=True)
class RunFailed(RunLogEntry):
    error: str
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_FAILED)


@dataclass(frozen=True, kw_only=True)
class ContextAssembled(RunLogEntry):
    messages: list[dict[str, Any]] = field(default_factory=list)
    memory_count: int = 0
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.CONTEXT_ASSEMBLED)


@dataclass(frozen=True, kw_only=True)
class MessagesRebuilt(RunLogEntry):
    reason: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.MESSAGES_REBUILT)


@dataclass(frozen=True, kw_only=True)
class LLMCallStarted(RunLogEntry):
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.LLM_CALL_STARTED)


@dataclass(frozen=True, kw_only=True)
class LLMCallCompleted(RunLogEntry):
    content: MessageContent | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    metrics: StepMetrics | None = None
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.LLM_CALL_COMPLETED
    )


@dataclass(frozen=True, kw_only=True)
class CommittedStep(RunLogEntry):
    role: MessageRole
    content: MessageContent | None = None
    content_for_user: str | None = None
    reasoning_content: str | None = None
    user_input: UserInput | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metrics: StepMetrics | None = None
    condensed_content: str | None = None


@dataclass(frozen=True, kw_only=True)
class UserStepCommitted(CommittedStep):
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.USER_STEP_COMMITTED
    )


@dataclass(frozen=True, kw_only=True)
class AssistantStepCommitted(CommittedStep):
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.ASSISTANT_STEP_COMMITTED
    )


@dataclass(frozen=True, kw_only=True)
class ToolStepCommitted(CommittedStep):
    is_error: bool = False
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.TOOL_STEP_COMMITTED
    )


@dataclass(frozen=True, kw_only=True)
class CompactionApplied(RunLogEntry):
    start_sequence: int
    end_sequence: int
    transcript_path: str
    summary: str | None = None
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.COMPACTION_APPLIED
    )


@dataclass(frozen=True, kw_only=True)
class RetrospectApplied(RunLogEntry):
    affected_sequences: list[int] = field(default_factory=list)
    affected_step_ids: list[str] = field(default_factory=list)
    feedback: str | None = None
    replacement: str | None = None
    trigger: str | None = None
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.RETROSPECT_APPLIED
    )


@dataclass(frozen=True, kw_only=True)
class TerminationDecided(RunLogEntry):
    termination_reason: TerminationReason
    phase: str
    source: str
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.TERMINATION_DECIDED
    )


@dataclass(frozen=True, kw_only=True)
class HookFailed(RunLogEntry):
    phase: str
    hook_name: str
    error: str
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.HOOK_FAILED)


__all__ = [
    "AssistantStepCommitted",
    "CommittedStep",
    "CompactionApplied",
    "ContextAssembled",
    "HookFailed",
    "LLMCallCompleted",
    "LLMCallStarted",
    "MessagesRebuilt",
    "RetrospectApplied",
    "RunFailed",
    "RunFinished",
    "RunLogEntry",
    "RunLogEntryKind",
    "RunStarted",
    "TerminationDecided",
    "ToolStepCommitted",
    "UserStepCommitted",
]
