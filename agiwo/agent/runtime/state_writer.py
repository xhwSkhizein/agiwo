"""Helpers for writing run-log entries from runtime state."""

from typing import Any

from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CompactionApplied,
    ContextAssembled,
    HookFailed,
    LLMCallCompleted,
    LLMCallStarted,
    MessagesRebuilt,
    RetrospectApplied,
    RunFailed,
    RunFinished,
    RunStarted,
    TerminationDecided,
    ToolStepCommitted,
    UserStepCommitted,
)
from agiwo.agent.models.input import UserInput
from agiwo.agent.models.run import CompactMetadata, RunOutput, TerminationReason
from agiwo.agent.models.step import LLMCallContext, StepRecord
from agiwo.agent.runtime.context import RunContext


def build_run_started_entry(
    state: RunContext,
    *,
    sequence: int,
    user_input: UserInput,
) -> RunStarted:
    return RunStarted(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        user_input=user_input,
    )


def build_run_finished_entry(
    state: RunContext,
    *,
    sequence: int,
    result: RunOutput,
) -> RunFinished:
    return RunFinished(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        response=result.response,
        termination_reason=result.termination_reason,
        metrics=result.metrics.to_dict() if result.metrics else None,
    )


def build_run_failed_entry(
    state: RunContext,
    *,
    sequence: int,
    error: Exception,
) -> RunFailed:
    return RunFailed(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        error=str(error),
    )


def build_context_assembled_entry(
    state: RunContext,
    *,
    sequence: int,
    messages: list[dict[str, Any]],
    memory_count: int,
) -> ContextAssembled:
    return ContextAssembled(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        messages=messages,
        memory_count=memory_count,
    )


def build_llm_call_started_entry(
    state: RunContext,
    *,
    sequence: int,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> LLMCallStarted:
    return LLMCallStarted(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        messages=messages,
        tools=tools,
    )


def build_llm_call_completed_entry(
    state: RunContext,
    *,
    sequence: int,
    step: StepRecord,
    llm: LLMCallContext,
) -> LLMCallCompleted:
    return LLMCallCompleted(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        content=step.content,
        reasoning_content=step.reasoning_content,
        tool_calls=step.tool_calls,
        finish_reason=llm.finish_reason,
        metrics=step.metrics,
    )


def build_step_log_entry(
    step: StepRecord,
) -> UserStepCommitted | AssistantStepCommitted | ToolStepCommitted:
    common = {
        "sequence": step.sequence,
        "session_id": step.session_id,
        "run_id": step.run_id,
        "agent_id": step.agent_id or "",
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
    }
    if step.is_user_step():
        return UserStepCommitted(**common)
    if step.is_assistant_step():
        return AssistantStepCommitted(**common)
    return ToolStepCommitted(**common, is_error=step.is_error)


def build_hook_failed_entry(
    state: RunContext,
    *,
    sequence: int,
    phase: str,
    hook_name: str,
    error: Exception,
) -> HookFailed:
    return HookFailed(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        phase=phase,
        hook_name=hook_name,
        error=str(error),
    )


def build_messages_rebuilt_entry(
    state: RunContext,
    *,
    sequence: int,
    reason: str,
    messages: list[dict[str, Any]],
) -> MessagesRebuilt:
    return MessagesRebuilt(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        reason=reason,
        messages=messages,
    )


def build_compaction_applied_entry(
    state: RunContext,
    *,
    sequence: int,
    metadata: CompactMetadata,
) -> CompactionApplied:
    return CompactionApplied(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        start_sequence=metadata.start_seq,
        end_sequence=metadata.end_seq,
        transcript_path=metadata.transcript_path,
        summary=metadata.get_summary() or None,
    )


def build_retrospect_applied_entry(
    state: RunContext,
    *,
    sequence: int,
    affected_sequences: list[int],
    affected_step_ids: list[str],
    feedback: str | None,
    replacement: str | None,
    trigger: str | None = None,
) -> RetrospectApplied:
    return RetrospectApplied(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        affected_sequences=affected_sequences,
        affected_step_ids=affected_step_ids,
        feedback=feedback,
        replacement=replacement,
        trigger=trigger,
    )


def build_termination_decided_entry(
    state: RunContext,
    *,
    sequence: int,
    termination_reason: TerminationReason,
    phase: str,
    source: str,
) -> TerminationDecided:
    return TerminationDecided(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        termination_reason=termination_reason,
        phase=phase,
        source=source,
    )
