"""Helpers and write coordinator for runtime-truth run-log writes."""

from typing import Any

from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CompactionApplied,
    CompactionFailed,
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
    build_committed_step_entry,
)
from agiwo.agent.models.input import UserInput
from agiwo.agent.models.run import CompactMetadata, RunOutput, TerminationReason
from agiwo.agent.models.step import LLMCallContext, StepView
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.state_ops import (
    record_compaction_metadata,
    replace_messages,
    set_termination_reason,
    set_tool_schemas,
    track_step_state,
)


class RunStateWriter:
    """Own committed state updates plus canonical run-log writes."""

    def __init__(self, state: RunContext) -> None:
        self._state = state

    async def append_entries(self, entries: list[object]) -> list[object]:
        typed_entries = list(entries)
        await self._state.session_runtime.append_run_log_entries(typed_entries)
        return typed_entries

    async def start_run(self, user_input: UserInput) -> list[object]:
        return await self.append_entries(
            [
                build_run_started_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    user_input=user_input,
                )
            ]
        )

    async def finish_run(self, result: RunOutput) -> list[object]:
        return await self.append_entries(
            [
                build_run_finished_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    result=result,
                )
            ]
        )

    async def fail_run(self, error: Exception) -> list[object]:
        return await self.append_entries(
            [
                build_run_failed_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    error=error,
                )
            ]
        )

    async def record_context_assembled(
        self,
        *,
        messages: list[dict[str, Any]],
        memory_count: int,
        run_start_seq: int,
        tool_schemas: list[dict[str, Any]] | None,
        latest_compaction: CompactMetadata | None,
    ) -> list[object]:
        replace_messages(self._state, messages)
        self._state.ledger.run_start_seq = run_start_seq
        set_tool_schemas(self._state, tool_schemas)
        record_compaction_metadata(self._state, latest_compaction)
        return await self.append_entries(
            [
                build_context_assembled_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    messages=self._state.snapshot_messages(),
                    memory_count=memory_count,
                )
            ]
        )

    async def rebuild_messages(
        self,
        *,
        reason: str,
        messages: list[dict[str, Any]],
    ) -> list[object]:
        replace_messages(self._state, messages)
        return await self.append_entries(
            [
                build_messages_rebuilt_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    reason=reason,
                    messages=self._state.snapshot_messages(),
                )
            ]
        )

    async def record_llm_call_started(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_llm_call_started_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    messages=messages,
                    tools=tools,
                )
            ]
        )

    async def record_llm_call_completed(
        self,
        *,
        step: StepView,
        llm: LLMCallContext,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_llm_call_completed_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    step=step,
                    llm=llm,
                )
            ]
        )

    async def commit_step(
        self,
        step: StepView,
        *,
        append_message: bool = True,
        track_state: bool = True,
    ) -> list[object]:
        if track_state:
            track_step_state(self._state, step, append_message=append_message)
        return await self.append_entries([build_step_log_entry(step)])

    async def record_termination_decided(
        self,
        *,
        termination_reason: TerminationReason,
        phase: str,
        source: str,
    ) -> list[object]:
        set_termination_reason(self._state, termination_reason)
        return await self.append_entries(
            [
                build_termination_decided_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    termination_reason=termination_reason,
                    phase=phase,
                    source=source,
                )
            ]
        )

    async def record_compaction_applied(
        self,
        metadata: CompactMetadata,
    ) -> list[object]:
        record_compaction_metadata(self._state, metadata)
        return await self.append_entries(
            [
                build_compaction_applied_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    metadata=metadata,
                )
            ]
        )

    async def record_compaction_failed(
        self,
        *,
        error: str,
        attempt: int,
        max_attempts: int,
        terminal: bool,
    ) -> list[object]:
        self._state.ledger.compaction.failure_count = attempt
        return await self.append_entries(
            [
                build_compaction_failed_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    error=error,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    terminal=terminal,
                )
            ]
        )

    def next_compaction_failure_attempt(self) -> int:
        return self._state.ledger.compaction.failure_count + 1

    async def record_retrospect_applied(
        self,
        *,
        affected_sequences: list[int],
        affected_step_ids: list[str],
        feedback: str | None,
        replacement: str | None,
        trigger: str | None = None,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_retrospect_applied_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    affected_sequences=affected_sequences,
                    affected_step_ids=affected_step_ids,
                    feedback=feedback,
                    replacement=replacement,
                    trigger=trigger,
                )
            ]
        )

    async def record_hook_failed(
        self,
        *,
        phase: str,
        handler_name: str,
        critical: bool,
        error: str,
        traceback: str | None = None,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_hook_failed_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    phase=phase,
                    handler_name=handler_name,
                    critical=critical,
                    error=error,
                    traceback=traceback,
                )
            ]
        )


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
        user_id=state.user_id,
        parent_run_id=state.parent_run_id,
        depth=state.depth,
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
    step: StepView,
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
    step: StepView,
) -> UserStepCommitted | AssistantStepCommitted | ToolStepCommitted:
    return build_committed_step_entry(step)


def build_hook_failed_entry(
    state: RunContext,
    *,
    sequence: int,
    phase: str,
    handler_name: str,
    critical: bool,
    error: str,
    traceback: str | None = None,
) -> HookFailed:
    return HookFailed(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        phase=phase,
        handler_name=handler_name,
        critical=critical,
        error=error,
        traceback=traceback,
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
        before_token_estimate=metadata.before_token_estimate,
        after_token_estimate=metadata.after_token_estimate,
        message_count=metadata.message_count,
        transcript_path=metadata.transcript_path,
        analysis=dict(metadata.analysis),
        summary=metadata.get_summary() or None,
        compact_model=metadata.compact_model,
        compact_tokens=metadata.compact_tokens,
        created_at=metadata.created_at,
    )


def build_compaction_failed_entry(
    state: RunContext,
    *,
    sequence: int,
    error: str,
    attempt: int,
    max_attempts: int,
    terminal: bool,
) -> CompactionFailed:
    return CompactionFailed(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        error=error,
        attempt=attempt,
        max_attempts=max_attempts,
        terminal=terminal,
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


__all__ = [
    "RunStateWriter",
    "build_compaction_applied_entry",
    "build_compaction_failed_entry",
    "build_context_assembled_entry",
    "build_hook_failed_entry",
    "build_llm_call_completed_entry",
    "build_llm_call_started_entry",
    "build_messages_rebuilt_entry",
    "build_retrospect_applied_entry",
    "build_run_failed_entry",
    "build_run_finished_entry",
    "build_run_started_entry",
    "build_step_log_entry",
    "build_termination_decided_entry",
]
