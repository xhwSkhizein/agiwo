"""Run-log models for replayable agent execution facts."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agiwo.agent.models.input import MessageContent, UserInput
from agiwo.agent.models.run import CompactMetadata
from agiwo.agent.models.step import MessageRole, StepMetrics, StepView
from agiwo.config.termination import TerminationReason


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class RunLogEntryKind(str, Enum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"
    RUN_ROLLED_BACK = "run_rolled_back"
    CONTEXT_ASSEMBLED = "context_assembled"
    MESSAGES_REBUILT = "messages_rebuilt"
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"
    USER_STEP_COMMITTED = "user_step_committed"
    ASSISTANT_STEP_COMMITTED = "assistant_step_committed"
    TOOL_STEP_COMMITTED = "tool_step_committed"
    COMPACTION_APPLIED = "compaction_applied"
    COMPACTION_FAILED = "compaction_failed"
    STEP_BACK_APPLIED = "step_back_applied"
    STEP_CONDENSED_CONTENT_UPDATED = "step_condensed_content_updated"
    CONTEXT_STEPS_HIDDEN = "context_steps_hidden"
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
    depth: int = 0
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
class RunRolledBack(RunLogEntry):
    start_sequence: int
    end_sequence: int
    reason: str
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_ROLLED_BACK)


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
    step_id: str
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
    parent_run_id: str | None = None
    depth: int = 0


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
    before_token_estimate: int
    after_token_estimate: int
    message_count: int
    transcript_path: str
    analysis: dict[str, Any] = field(default_factory=dict)
    summary: str | None = None
    compact_model: str = ""
    compact_tokens: int = 0
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.COMPACTION_APPLIED
    )


@dataclass(frozen=True, kw_only=True)
class CompactionFailed(RunLogEntry):
    error: str
    attempt: int
    max_attempts: int
    terminal: bool = False
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.COMPACTION_FAILED)


@dataclass(frozen=True, kw_only=True)
class StepBackApplied(RunLogEntry):
    """Log entry recorded when step-back condenses off-track tool results."""

    affected_count: int
    checkpoint_seq: int
    experience: str
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.STEP_BACK_APPLIED)


@dataclass(frozen=True, kw_only=True)
class StepCondensedContentUpdated(RunLogEntry):
    step_id: str
    condensed_content: str
    kind: RunLogEntryKind = field(
        init=False,
        default=RunLogEntryKind.STEP_CONDENSED_CONTENT_UPDATED,
    )


@dataclass(frozen=True, kw_only=True)
class ContextStepsHidden(RunLogEntry):
    step_ids: list[str]
    reason: str
    kind: RunLogEntryKind = field(
        init=False,
        default=RunLogEntryKind.CONTEXT_STEPS_HIDDEN,
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
    handler_name: str
    critical: bool = False
    error: str
    traceback: str | None = None
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.HOOK_FAILED)


def build_committed_step_entry(step: StepView) -> CommittedStep:
    if step.agent_id is None:
        raise ValueError(
            "Committed step requires a non-null agent_id: "
            f"step_id={step.id or '<pending>'} run_id={step.run_id}"
        )
    common = {
        "sequence": step.sequence,
        "session_id": step.session_id,
        "run_id": step.run_id,
        "agent_id": step.agent_id,
        "step_id": step.id,
        "role": step.role,
        "content": step.content,
        "content_for_user": step.content_for_user,
        "reasoning_content": step.reasoning_content,
        "user_input": step.user_input,
        "tool_calls": step.tool_calls,
        "tool_call_id": step.tool_call_id,
        "name": step.name,
        "metrics": step.metrics,
        "condensed_content": step.condensed_content,
        "parent_run_id": step.parent_run_id,
        "depth": step.depth,
        "created_at": step.created_at,
    }
    if step.role == MessageRole.USER:
        return UserStepCommitted(**common)
    if step.role == MessageRole.ASSISTANT:
        return AssistantStepCommitted(**common)
    return ToolStepCommitted(**common, is_error=step.is_error)


def build_compact_metadata_from_entry(entry: CompactionApplied) -> CompactMetadata:
    return CompactMetadata(
        session_id=entry.session_id,
        agent_id=entry.agent_id,
        start_seq=entry.start_sequence,
        end_seq=entry.end_sequence,
        before_token_estimate=entry.before_token_estimate,
        after_token_estimate=entry.after_token_estimate,
        message_count=entry.message_count,
        transcript_path=entry.transcript_path,
        analysis=dict(entry.analysis),
        created_at=entry.created_at,
        compact_model=entry.compact_model,
        compact_tokens=entry.compact_tokens,
    )


__all__ = [
    "AssistantStepCommitted",
    "build_compact_metadata_from_entry",
    "build_committed_step_entry",
    "CommittedStep",
    "CompactionApplied",
    "CompactionFailed",
    "ContextStepsHidden",
    "ContextAssembled",
    "HookFailed",
    "LLMCallCompleted",
    "LLMCallStarted",
    "MessagesRebuilt",
    "RunRolledBack",
    "StepBackApplied",
    "StepCondensedContentUpdated",
    "RunFailed",
    "RunFinished",
    "RunLogEntry",
    "RunLogEntryKind",
    "RunStarted",
    "TerminationDecided",
    "ToolStepCommitted",
    "UserStepCommitted",
]
