"""Helpers and write coordinator for runtime-truth run-log writes."""

from typing import Any, Literal

from agiwo.agent.models.log import (
    AssistantStepCommitted,
    CompactionApplied,
    CompactionFailed,
    ContextRepairApplied,
    ContextStepsHidden,
    ContextAssembled,
    GoalMilestonesUpdated,
    HookFailed,
    IntrospectionCheckpointRecorded,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    LLMCallCompleted,
    LLMCallStarted,
    MessagesRebuilt,
    RunFailed,
    RunFinished,
    RunStarted,
    StepBackApplied,
    StepCondensedContentUpdated,
    TerminationDecided,
    ToolStepCommitted,
    UserStepCommitted,
    build_committed_step_entry,
)
from agiwo.agent.models.input import UserInput
from agiwo.agent.introspect.models import Milestone
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

    async def project_entries(self, entries: list[object]) -> None:
        if not entries:
            return
        await self._state.session_runtime.project_run_log_entries(
            entries,
            run_id=self._state.run_id,
            agent_id=self._state.agent_id,
            parent_run_id=self._state.parent_run_id,
            depth=self._state.depth,
        )

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

    async def record_step_back_applied(
        self,
        *,
        affected_count: int,
        checkpoint_seq: int,
        experience: str,
    ) -> list[object]:
        """Record a step-back applied event to the run log."""
        return await self.append_entries(
            [
                build_step_back_applied_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    affected_count=affected_count,
                    checkpoint_seq=checkpoint_seq,
                    experience=experience,
                )
            ]
        )

    async def record_context_steps_hidden(
        self,
        *,
        step_ids: list[str],
        reason: str = "introspection_metadata",
    ) -> list[object]:
        return await self.append_entries(
            [
                build_context_steps_hidden_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    step_ids=step_ids,
                    reason=reason,
                )
            ]
        )

    async def record_goal_milestones_updated(
        self,
        *,
        milestones: list[Milestone],
        active_milestone_id: str | None,
        source_tool_call_id: str | None,
        source_step_id: str | None,
        reason: Literal["declared", "updated", "completed", "activated"],
    ) -> list[object]:
        return await self.append_entries(
            [
                build_goal_milestones_updated_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    milestones=list(milestones),
                    active_milestone_id=active_milestone_id,
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
        mode: Literal["metadata_only", "step_back"],
        experience: str | None,
        active_milestone_id: str | None,
        review_tool_call_id: str | None,
        review_step_id: str | None,
        hidden_step_ids: list[str],
        notice_cleaned_step_ids: list[str],
        condensed_step_ids: list[str],
        boundary_seq: int,
        repair_start_seq: int | None,
        repair_end_seq: int | None,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_introspection_outcome_recorded_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    aligned=aligned,
                    mode=mode,
                    experience=experience,
                    active_milestone_id=active_milestone_id,
                    review_tool_call_id=review_tool_call_id,
                    review_step_id=review_step_id,
                    hidden_step_ids=list(hidden_step_ids),
                    notice_cleaned_step_ids=list(notice_cleaned_step_ids),
                    condensed_step_ids=list(condensed_step_ids),
                    boundary_seq=boundary_seq,
                    repair_start_seq=repair_start_seq,
                    repair_end_seq=repair_end_seq,
                )
            ]
        )

    async def record_context_repair_applied(
        self,
        *,
        mode: Literal["step_back"],
        affected_count: int,
        start_seq: int,
        end_seq: int,
        experience: str,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_context_repair_applied_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    mode=mode,
                    affected_count=affected_count,
                    start_seq=start_seq,
                    end_seq=end_seq,
                    experience=experience,
                )
            ]
        )

    async def record_step_condensed_content_updated(
        self,
        *,
        step_id: str,
        condensed_content: str,
    ) -> list[object]:
        return await self.append_entries(
            [
                build_step_condensed_content_updated_entry(
                    self._state,
                    sequence=await self._state.session_runtime.allocate_sequence(),
                    step_id=step_id,
                    condensed_content=condensed_content,
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


def build_step_back_applied_entry(
    state: RunContext,
    *,
    sequence: int,
    affected_count: int,
    checkpoint_seq: int,
    experience: str,
) -> StepBackApplied:
    """Build a StepBackApplied log entry."""
    return StepBackApplied(
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        sequence=sequence,
        affected_count=affected_count,
        checkpoint_seq=checkpoint_seq,
        experience=experience,
    )


def build_context_steps_hidden_entry(
    state: RunContext,
    *,
    sequence: int,
    step_ids: list[str],
    reason: str,
) -> ContextStepsHidden:
    return ContextStepsHidden(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        step_ids=list(step_ids),
        reason=reason,
    )


def build_step_condensed_content_updated_entry(
    state: RunContext,
    *,
    sequence: int,
    step_id: str,
    condensed_content: str,
) -> StepCondensedContentUpdated:
    return StepCondensedContentUpdated(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        step_id=step_id,
        condensed_content=condensed_content,
    )


def build_goal_milestones_updated_entry(
    state: RunContext,
    *,
    sequence: int,
    milestones: list[Milestone],
    active_milestone_id: str | None,
    source_tool_call_id: str | None,
    source_step_id: str | None,
    reason: Literal["declared", "updated", "completed", "activated"],
) -> GoalMilestonesUpdated:
    return GoalMilestonesUpdated(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        milestones=list(milestones),
        active_milestone_id=active_milestone_id,
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
    mode: Literal["metadata_only", "step_back"],
    experience: str | None,
    active_milestone_id: str | None,
    review_tool_call_id: str | None,
    review_step_id: str | None,
    hidden_step_ids: list[str],
    notice_cleaned_step_ids: list[str],
    condensed_step_ids: list[str],
    boundary_seq: int,
    repair_start_seq: int | None,
    repair_end_seq: int | None,
) -> IntrospectionOutcomeRecorded:
    return IntrospectionOutcomeRecorded(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        aligned=aligned,
        mode=mode,
        experience=experience,
        active_milestone_id=active_milestone_id,
        review_tool_call_id=review_tool_call_id,
        review_step_id=review_step_id,
        hidden_step_ids=list(hidden_step_ids),
        notice_cleaned_step_ids=list(notice_cleaned_step_ids),
        condensed_step_ids=list(condensed_step_ids),
        boundary_seq=boundary_seq,
        repair_start_seq=repair_start_seq,
        repair_end_seq=repair_end_seq,
    )


def build_context_repair_applied_entry(
    state: RunContext,
    *,
    sequence: int,
    mode: Literal["step_back"],
    affected_count: int,
    start_seq: int,
    end_seq: int,
    experience: str,
) -> ContextRepairApplied:
    return ContextRepairApplied(
        sequence=sequence,
        session_id=state.session_id,
        run_id=state.run_id,
        agent_id=state.agent_id,
        mode=mode,
        affected_count=affected_count,
        start_seq=start_seq,
        end_seq=end_seq,
        experience=experience,
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
    "build_context_repair_applied_entry",
    "build_context_steps_hidden_entry",
    "build_goal_milestones_updated_entry",
    "build_hook_failed_entry",
    "build_introspection_checkpoint_recorded_entry",
    "build_introspection_outcome_recorded_entry",
    "build_introspection_triggered_entry",
    "build_llm_call_completed_entry",
    "build_llm_call_started_entry",
    "build_messages_rebuilt_entry",
    "build_step_back_applied_entry",
    "build_step_condensed_content_updated_entry",
    "build_run_failed_entry",
    "build_run_finished_entry",
    "build_run_started_entry",
    "build_step_log_entry",
    "build_termination_decided_entry",
]
