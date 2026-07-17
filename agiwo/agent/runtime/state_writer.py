"""Helpers and write coordinator for runtime-truth run-log writes."""

from typing import Any, Literal

from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CompactionApplied,
    CompactionFailed,
    ContextAssembled,
    HookFailed,
    IntrospectionCheckpointRecorded,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    LLMCallCompleted,
    LLMCallFailed,
    LLMCallStarted,
    MessagesRebuilt,
    RunFailed,
    RunFinished,
    RunPlanUpdated,
    RunStarted,
    TerminationDecided,
    ToolStepCommitted,
    UserStepCommitted,
    build_committed_step_entry,
)
from agiwo.agent.models.input import UserInput
from agiwo.agent.models.model_call import ModelCallPhase
from agiwo.agent.models.plan import Milestone
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
        logical_call_id: str,
        phase: ModelCallPhase,
        attempt_no: int,
        call_ordinal: int,
        retry_reason: str | None = None,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_llm_call_started_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    messages=messages,
                    tools=tools,
                    logical_call_id=logical_call_id,
                    phase=phase,
                    attempt_no=attempt_no,
                    call_ordinal=call_ordinal,
                    retry_reason=retry_reason,
                )
            ]
        )

    async def record_llm_call_completed(
        self,
        *,
        step: StepView,
        llm: LLMCallContext,
        logical_call_id: str,
        phase: ModelCallPhase,
        attempt_no: int,
        call_ordinal: int,
        retry_reason: str | None = None,
        response_observed: bool = True,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_llm_call_completed_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    step=step,
                    llm=llm,
                    logical_call_id=logical_call_id,
                    phase=phase,
                    attempt_no=attempt_no,
                    call_ordinal=call_ordinal,
                    retry_reason=retry_reason,
                    response_observed=response_observed,
                )
            ]
        )

    async def record_llm_call_failed(
        self,
        *,
        logical_call_id: str,
        phase: ModelCallPhase,
        attempt_no: int,
        call_ordinal: int,
        error: str,
        retry_reason: str | None = None,
        step: StepView | None = None,
        response_observed: bool = False,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_llm_call_failed_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    logical_call_id=logical_call_id,
                    phase=phase,
                    attempt_no=attempt_no,
                    call_ordinal=call_ordinal,
                    retry_reason=retry_reason,
                    error=error,
                    step=step,
                    response_observed=response_observed,
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

    async def record_run_plan_updated(
        self,
        *,
        milestones: list[Milestone],
        revision: int,
        source_tool_call_id: str | None,
        source_step_id: str | None,
        reason: Literal["declared", "updated", "completed", "activated"],
    ) -> list[object]:
        return await self.append_entries(
            [
                build_run_plan_updated_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    milestones=list(milestones),
                    revision=revision,
                    source_tool_call_id=source_tool_call_id,
                    source_step_id=source_step_id,
                    reason=reason,
                )
            ]
        )

    async def record_introspection_triggered(
        self,
        *,
        trigger_reason: Literal[
            "step_interval", "consecutive_errors", "milestone_switch"
        ],
        active_milestone_id: str | None,
        review_count_since_boundary: int,
        trigger_tool_call_id: str | None,
        trigger_tool_step_id: str | None,
        notice_step_id: str | None,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_introspection_triggered_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    trigger_reason=trigger_reason,
                    active_milestone_id=active_milestone_id,
                    review_count_since_boundary=review_count_since_boundary,
                    trigger_tool_call_id=trigger_tool_call_id,
                    trigger_tool_step_id=trigger_tool_step_id,
                    notice_step_id=notice_step_id,
                )
            ]
        )

    async def record_introspection_checkpoint_recorded(
        self,
        *,
        checkpoint_seq: int,
        milestone_id: str | None,
        review_tool_call_id: str | None,
        review_step_id: str | None,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_introspection_checkpoint_recorded_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    checkpoint_seq=checkpoint_seq,
                    milestone_id=milestone_id,
                    review_tool_call_id=review_tool_call_id,
                    review_step_id=review_step_id,
                )
            ]
        )

    async def record_introspection_outcome_recorded(
        self,
        *,
        aligned: bool | None,
        experience: str | None,
        tool_usefulness: list[dict[str, object]],
        active_milestone_id: str | None,
        review_tool_call_id: str | None,
        review_step_id: str | None,
        boundary_seq: int,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_introspection_outcome_recorded_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    aligned=aligned,
                    experience=experience,
                    tool_usefulness=list(tool_usefulness),
                    active_milestone_id=active_milestone_id,
                    review_tool_call_id=review_tool_call_id,
                    review_step_id=review_step_id,
                    boundary_seq=boundary_seq,
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
    logical_call_id: str,
    phase: ModelCallPhase,
    attempt_no: int,
    call_ordinal: int,
    retry_reason: str | None = None,
) -> LLMCallStarted:
    return LLMCallStarted(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        logical_call_id=logical_call_id,
        phase=phase,
        attempt_no=attempt_no,
        call_ordinal=call_ordinal,
        retry_reason=retry_reason,
        messages=messages,
        tools=tools,
    )


def build_llm_call_completed_entry(
    state: RunContext,
    *,
    sequence: int,
    step: StepView,
    llm: LLMCallContext,
    logical_call_id: str,
    phase: ModelCallPhase,
    attempt_no: int,
    call_ordinal: int,
    retry_reason: str | None = None,
    response_observed: bool = True,
) -> LLMCallCompleted:
    return LLMCallCompleted(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        logical_call_id=logical_call_id,
        phase=phase,
        attempt_no=attempt_no,
        call_ordinal=call_ordinal,
        retry_reason=retry_reason,
        content=step.content,
        reasoning_content=step.reasoning_content,
        tool_calls=step.tool_calls,
        finish_reason=llm.finish_reason,
        metrics=step.metrics,
        response_observed=response_observed,
    )


def build_llm_call_failed_entry(
    state: RunContext,
    *,
    sequence: int,
    logical_call_id: str,
    phase: ModelCallPhase,
    attempt_no: int,
    call_ordinal: int,
    error: str,
    retry_reason: str | None = None,
    step: StepView | None = None,
    response_observed: bool = False,
) -> LLMCallFailed:
    return LLMCallFailed(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        logical_call_id=logical_call_id,
        phase=phase,
        attempt_no=attempt_no,
        call_ordinal=call_ordinal,
        retry_reason=retry_reason,
        error=error,
        content=step.content if step is not None else None,
        reasoning_content=step.reasoning_content if step is not None else None,
        tool_calls=step.tool_calls if step is not None else None,
        metrics=step.metrics if step is not None else None,
        response_observed=response_observed,
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


def build_run_plan_updated_entry(
    state: RunContext,
    *,
    sequence: int,
    milestones: list[Milestone],
    revision: int,
    source_tool_call_id: str | None,
    source_step_id: str | None,
    reason: Literal["declared", "updated", "completed", "activated"],
) -> RunPlanUpdated:
    return RunPlanUpdated(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        milestones=list(milestones),
        revision=revision,
        source_tool_call_id=source_tool_call_id,
        source_step_id=source_step_id,
        reason=reason,
    )


def build_introspection_triggered_entry(
    state: RunContext,
    *,
    sequence: int,
    trigger_reason: Literal["step_interval", "consecutive_errors", "milestone_switch"],
    active_milestone_id: str | None,
    review_count_since_boundary: int,
    trigger_tool_call_id: str | None,
    trigger_tool_step_id: str | None,
    notice_step_id: str | None,
) -> IntrospectionTriggered:
    return IntrospectionTriggered(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        trigger_reason=trigger_reason,
        active_milestone_id=active_milestone_id,
        review_count_since_boundary=review_count_since_boundary,
        trigger_tool_call_id=trigger_tool_call_id,
        trigger_tool_step_id=trigger_tool_step_id,
        notice_step_id=notice_step_id,
    )


def build_introspection_checkpoint_recorded_entry(
    state: RunContext,
    *,
    sequence: int,
    checkpoint_seq: int,
    milestone_id: str | None,
    review_tool_call_id: str | None,
    review_step_id: str | None,
) -> IntrospectionCheckpointRecorded:
    return IntrospectionCheckpointRecorded(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        checkpoint_seq=checkpoint_seq,
        milestone_id=milestone_id,
        review_tool_call_id=review_tool_call_id,
        review_step_id=review_step_id,
    )


def build_introspection_outcome_recorded_entry(
    state: RunContext,
    *,
    sequence: int,
    aligned: bool | None,
    experience: str | None,
    tool_usefulness: list[dict[str, object]],
    active_milestone_id: str | None,
    review_tool_call_id: str | None,
    review_step_id: str | None,
    boundary_seq: int,
) -> IntrospectionOutcomeRecorded:
    return IntrospectionOutcomeRecorded(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        aligned=aligned,
        experience=experience,
        tool_usefulness=list(tool_usefulness),
        active_milestone_id=active_milestone_id,
        review_tool_call_id=review_tool_call_id,
        review_step_id=review_step_id,
        boundary_seq=boundary_seq,
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
    "build_introspection_checkpoint_recorded_entry",
    "build_introspection_outcome_recorded_entry",
    "build_introspection_triggered_entry",
    "build_llm_call_completed_entry",
    "build_llm_call_failed_entry",
    "build_llm_call_started_entry",
    "build_messages_rebuilt_entry",
    "build_run_failed_entry",
    "build_run_finished_entry",
    "build_run_started_entry",
    "build_step_log_entry",
    "build_termination_decided_entry",
]
