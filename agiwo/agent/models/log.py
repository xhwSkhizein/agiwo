"""Run-log models for replayable agent execution facts."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from agiwo.agent.models.input import MessageContent, UserInput
from agiwo.agent.models.model_call import ModelCallPhase
from agiwo.agent.models.plan import Milestone
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
    LLM_CALL_FAILED = "llm_call_failed"
    USER_STEP_COMMITTED = "user_step_committed"
    ASSISTANT_STEP_COMMITTED = "assistant_step_committed"
    TOOL_STEP_COMMITTED = "tool_step_committed"
    COMPACTION_APPLIED = "compaction_applied"
    COMPACTION_FAILED = "compaction_failed"
    TERMINATION_DECIDED = "termination_decided"
    HOOK_FAILED = "hook_failed"
    RUN_PLAN_UPDATED = "run_plan_updated"
    INTROSPECTION_TRIGGERED = "introspection_triggered"
    INTROSPECTION_CHECKPOINT_RECORDED = "introspection_checkpoint_recorded"
    INTROSPECTION_OUTCOME_RECORDED = "introspection_outcome_recorded"
    RUN_CHECKPOINT = "run_checkpoint"
    RUN_PAUSED = "run_paused"
    RUN_RESUME_PREPARED = "run_resume_prepared"
    RUN_RESUMED = "run_resumed"
    EXTERNAL_EFFECT_MAY_HAVE_STARTED = "external_effect_may_have_started"
    RETRY_BACKOFF = "retry_backoff"


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
    objective_id: str | None = None
    run_tree_role: str | None = None
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_STARTED)


@dataclass(frozen=True, kw_only=True)
class RunFinished(RunLogEntry):
    response: str | None = None
    termination_reason: TerminationReason | None = None
    metrics: dict[str, Any] | None = None
    finalization: dict[str, Any] | None = None
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_FINISHED)


@dataclass(frozen=True, kw_only=True)
class RunFailed(RunLogEntry):
    error: str
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_FAILED)


@dataclass(frozen=True, kw_only=True)
class RunCheckpoint(RunLogEntry):
    """Minimal resume cursor; messages are rebuilt from RunLog replay."""

    checkpoint_id: str
    last_committed_sequence: int
    agent_config_hash: str | None = None
    template_hash: str | None = None
    reason: str | None = None
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_CHECKPOINT)


@dataclass(frozen=True, kw_only=True)
class RunPaused(RunLogEntry):
    """Recoverable interrupt; must not pair with RunFinished/Failed/TerminationDecided."""

    checkpoint_id: str
    reason: str
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_PAUSED)


@dataclass(frozen=True, kw_only=True)
class RunResumePrepared(RunLogEntry):
    """Runtime rebuilt and validated; still PAUSED until barrier release."""

    checkpoint_id: str
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.RUN_RESUME_PREPARED
    )


@dataclass(frozen=True, kw_only=True)
class RunResumed(RunLogEntry):
    checkpoint_id: str
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_RESUMED)


@dataclass(frozen=True, kw_only=True)
class ExternalEffectMayHaveStarted(RunLogEntry):
    """Marker committed before a tool adapter that may cause external side effects."""

    tool_name: str
    tool_call_id: str
    idempotency: str
    idempotency_key: str | None = None
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.EXTERNAL_EFFECT_MAY_HAVE_STARTED
    )


@dataclass(frozen=True, kw_only=True)
class RetryBackoff(RunLogEntry):
    """Recorded wait before a safe automatic retry attempt."""

    operation: str
    attempt_no: int
    wait_seconds: float
    reason: str
    logical_call_id: str | None = None
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RETRY_BACKOFF)


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
    logical_call_id: str
    phase: ModelCallPhase
    attempt_no: int
    call_ordinal: int
    retry_reason: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    request_tokens: int | None = None
    call_cost_ceiling: float | None = None
    price_snapshot: dict[str, float] | None = None
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.LLM_CALL_STARTED)


@dataclass(frozen=True, kw_only=True)
class LLMCallCompleted(RunLogEntry):
    logical_call_id: str
    phase: ModelCallPhase
    attempt_no: int
    call_ordinal: int
    retry_reason: str | None = None
    content: MessageContent | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    metrics: StepMetrics | None = None
    response_observed: bool = True
    request_tokens: int | None = None
    call_cost_ceiling: float | None = None
    price_snapshot: dict[str, float] | None = None
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.LLM_CALL_COMPLETED
    )


@dataclass(frozen=True, kw_only=True)
class LLMCallFailed(RunLogEntry):
    logical_call_id: str
    phase: ModelCallPhase
    attempt_no: int
    call_ordinal: int
    retry_reason: str | None = None
    error: str
    content: MessageContent | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    metrics: StepMetrics | None = None
    response_observed: bool = False
    request_tokens: int | None = None
    call_cost_ceiling: float | None = None
    price_snapshot: dict[str, float] | None = None
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.LLM_CALL_FAILED)


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


@dataclass(frozen=True, kw_only=True)
class RunPlanUpdated(RunLogEntry):
    milestones: list[Milestone] = field(default_factory=list)
    revision: int = 0
    source_tool_call_id: str | None = None
    source_step_id: str | None = None
    reason: Literal["declared", "updated", "completed", "activated"] = "updated"
    kind: RunLogEntryKind = field(init=False, default=RunLogEntryKind.RUN_PLAN_UPDATED)

    @property
    def active_milestone_id(self) -> str | None:
        for milestone in self.milestones:
            if milestone.status == "active":
                return milestone.id
        return None


@dataclass(frozen=True, kw_only=True)
class IntrospectionTriggered(RunLogEntry):
    trigger_reason: Literal["step_interval", "consecutive_errors", "milestone_switch"]
    active_milestone_id: str | None = None
    review_count_since_boundary: int = 0
    trigger_tool_call_id: str | None = None
    trigger_tool_step_id: str | None = None
    notice_step_id: str | None = None
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.INTROSPECTION_TRIGGERED
    )


@dataclass(frozen=True, kw_only=True)
class IntrospectionCheckpointRecorded(RunLogEntry):
    checkpoint_seq: int
    milestone_id: str | None = None
    review_tool_call_id: str | None = None
    review_step_id: str | None = None
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.INTROSPECTION_CHECKPOINT_RECORDED
    )


@dataclass(frozen=True, kw_only=True)
class IntrospectionOutcomeRecorded(RunLogEntry):
    boundary_seq: int
    aligned: bool | None = None
    experience: str | None = None
    active_milestone_id: str | None = None
    review_tool_call_id: str | None = None
    review_step_id: str | None = None
    tool_usefulness: list[dict[str, Any]] = field(default_factory=list)
    kind: RunLogEntryKind = field(
        init=False, default=RunLogEntryKind.INTROSPECTION_OUTCOME_RECORDED
    )


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
    "ContextAssembled",
    "RunPlanUpdated",
    "HookFailed",
    "IntrospectionCheckpointRecorded",
    "IntrospectionOutcomeRecorded",
    "IntrospectionTriggered",
    "LLMCallCompleted",
    "LLMCallFailed",
    "LLMCallStarted",
    "MessagesRebuilt",
    "RunCheckpoint",
    "RunPaused",
    "RunResumePrepared",
    "RunResumed",
    "ExternalEffectMayHaveStarted",
    "RetryBackoff",
    "RunRolledBack",
    "RunFailed",
    "RunFinished",
    "RunLogEntry",
    "RunLogEntryKind",
    "RunStarted",
    "TerminationDecided",
    "ToolStepCommitted",
    "UserStepCommitted",
]
