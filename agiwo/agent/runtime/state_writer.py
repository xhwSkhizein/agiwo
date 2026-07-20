"""Helpers and write coordinator for runtime-truth run-log writes."""

from typing import Any, Literal, TypeVar
from uuid import uuid4

from agiwo.agent.models.input import UserInput
from agiwo.agent.models.log import (
    CompactionApplied,
    CompactionFailed,
    ContextAssembled,
    ExternalEffectMayHaveStarted,
    HookFailed,
    IntrospectionCheckpointRecorded,
    IntrospectionOutcomeRecorded,
    IntrospectionTriggered,
    LLMCallCompleted,
    LLMCallFailed,
    LLMCallStarted,
    MessagesRebuilt,
    RetryBackoff,
    RunCheckpoint,
    RunFailed,
    RunFinished,
    RunLogEntry,
    RunPaused,
    RunPlanUpdated,
    RunResumePrepared,
    RunResumed,
    RunStarted,
    TerminationDecided,
    build_committed_step_entry,
)
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

EntryT = TypeVar("EntryT", bound=RunLogEntry)


class RunStateWriter:
    """Own committed state updates plus canonical run-log writes.

    ``append_entries`` always projects stream views after a successful write,
    so callers never need a paired ``project_entries`` call.
    """

    def __init__(self, state: RunContext) -> None:
        self._state = state

    async def emit(self, entry_cls: type[EntryT], **fields: Any) -> list[object]:
        """Allocate sequence, fill run identity, append, and project."""
        entry = entry_cls(
            sequence=await self._state.session_runtime.allocate_sequence(),
            session_id=self._state.session_id,
            run_id=self._state.run_id,
            agent_id=self._state.agent_id,
            **fields,
        )
        return await self.append_entries([entry])

    async def append_entries(self, entries: list[object]) -> list[object]:
        typed_entries = list(entries)
        await self._state.session_runtime.append_run_log_entries(typed_entries)
        await self._state.session_runtime.project_run_log_entries(
            typed_entries,
            run_id=self._state.run_id,
            agent_id=self._state.agent_id,
            parent_run_id=self._state.parent_run_id,
            depth=self._state.depth,
        )
        return typed_entries

    async def start_run(self, user_input: UserInput) -> list[object]:
        return await self.emit(
            RunStarted,
            user_input=user_input,
            user_id=self._state.user_id,
            parent_run_id=self._state.parent_run_id,
            depth=self._state.depth,
            objective_id=self._state.objective_id,
            run_tree_role=self._state.run_tree_role.value,
        )

    async def finish_run(self, result: RunOutput) -> list[object]:
        return await self.emit(
            RunFinished,
            response=result.response,
            termination_reason=result.termination_reason,
            metrics=result.metrics.to_dict() if result.metrics else None,
            finalization=(
                result.finalization.to_dict()
                if result.finalization is not None
                else None
            ),
        )

    async def fail_run(self, error: Exception) -> list[object]:
        return await self.emit(RunFailed, error=str(error))

    async def pause_at_checkpoint(
        self,
        *,
        reason: str,
        agent_config_hash: str | None = None,
        template_hash: str | None = None,
    ) -> tuple[list[object], str]:
        """Atomically write RunCheckpoint + RunPaused (or remain RUNNING on failure)."""
        checkpoint_id = f"chk_{uuid4().hex}"
        checkpoint_seq = await self._state.session_runtime.allocate_sequence()
        pause_seq = await self._state.session_runtime.allocate_sequence()
        last_committed = max(0, checkpoint_seq - 1)
        entries: list[object] = [
            RunCheckpoint(
                sequence=checkpoint_seq,
                session_id=self._state.session_id,
                run_id=self._state.run_id,
                agent_id=self._state.agent_id,
                checkpoint_id=checkpoint_id,
                last_committed_sequence=last_committed,
                agent_config_hash=agent_config_hash,
                template_hash=template_hash,
                reason=reason,
            ),
            RunPaused(
                sequence=pause_seq,
                session_id=self._state.session_id,
                run_id=self._state.run_id,
                agent_id=self._state.agent_id,
                checkpoint_id=checkpoint_id,
                reason=reason,
            ),
        ]
        await self.append_entries(entries)
        return entries, checkpoint_id

    async def record_resume_prepared(self, *, checkpoint_id: str) -> list[object]:
        return await self.emit(RunResumePrepared, checkpoint_id=checkpoint_id)

    async def record_resumed(self, *, checkpoint_id: str) -> list[object]:
        return await self.emit(RunResumed, checkpoint_id=checkpoint_id)

    async def record_external_effect_may_have_started(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        idempotency: str,
        idempotency_key: str | None = None,
    ) -> list[object]:
        return await self.emit(
            ExternalEffectMayHaveStarted,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            idempotency=idempotency,
            idempotency_key=idempotency_key,
        )

    async def record_retry_backoff(
        self,
        *,
        operation: str,
        attempt_no: int,
        wait_seconds: float,
        reason: str,
        logical_call_id: str | None = None,
    ) -> list[object]:
        return await self.emit(
            RetryBackoff,
            operation=operation,
            attempt_no=attempt_no,
            wait_seconds=wait_seconds,
            reason=reason,
            logical_call_id=logical_call_id,
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
        return await self.emit(
            ContextAssembled,
            messages=self._state.snapshot_messages(),
            memory_count=memory_count,
        )

    async def rebuild_messages(
        self,
        *,
        reason: str,
        messages: list[dict[str, Any]],
    ) -> list[object]:
        replace_messages(self._state, messages)
        return await self.emit(
            MessagesRebuilt,
            reason=reason,
            messages=self._state.snapshot_messages(),
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
        request_tokens: int | None = None,
        call_cost_ceiling: float | None = None,
        price_snapshot: dict[str, float] | None = None,
    ) -> list[object]:
        return await self.emit(
            LLMCallStarted,
            logical_call_id=logical_call_id,
            phase=phase,
            attempt_no=attempt_no,
            call_ordinal=call_ordinal,
            retry_reason=retry_reason,
            messages=messages,
            tools=tools,
            request_tokens=request_tokens,
            call_cost_ceiling=call_cost_ceiling,
            price_snapshot=price_snapshot,
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
        request_tokens: int | None = None,
        call_cost_ceiling: float | None = None,
        price_snapshot: dict[str, float] | None = None,
    ) -> list[object]:
        return await self.emit(
            LLMCallCompleted,
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
            request_tokens=request_tokens,
            call_cost_ceiling=call_cost_ceiling,
            price_snapshot=price_snapshot,
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
        request_tokens: int | None = None,
        call_cost_ceiling: float | None = None,
        price_snapshot: dict[str, float] | None = None,
    ) -> list[object]:
        return await self.emit(
            LLMCallFailed,
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
            request_tokens=request_tokens,
            call_cost_ceiling=call_cost_ceiling,
            price_snapshot=price_snapshot,
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
        return await self.append_entries([build_committed_step_entry(step)])

    async def record_termination_decided(
        self,
        *,
        termination_reason: TerminationReason,
        phase: str,
        source: str,
    ) -> list[object]:
        set_termination_reason(self._state, termination_reason)
        return await self.emit(
            TerminationDecided,
            termination_reason=termination_reason,
            phase=phase,
            source=source,
        )

    async def record_compaction_applied(
        self,
        metadata: CompactMetadata,
    ) -> list[object]:
        record_compaction_metadata(self._state, metadata)
        return await self.emit(
            CompactionApplied,
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

    async def record_compaction_failed(
        self,
        *,
        error: str,
        attempt: int,
        max_attempts: int,
        terminal: bool,
    ) -> list[object]:
        self._state.ledger.compaction.failure_count = attempt
        return await self.emit(
            CompactionFailed,
            error=error,
            attempt=attempt,
            max_attempts=max_attempts,
            terminal=terminal,
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
        return await self.emit(
            RunPlanUpdated,
            milestones=list(milestones),
            revision=revision,
            source_tool_call_id=source_tool_call_id,
            source_step_id=source_step_id,
            reason=reason,
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
        return await self.emit(
            IntrospectionTriggered,
            trigger_reason=trigger_reason,
            active_milestone_id=active_milestone_id,
            review_count_since_boundary=review_count_since_boundary,
            trigger_tool_call_id=trigger_tool_call_id,
            trigger_tool_step_id=trigger_tool_step_id,
            notice_step_id=notice_step_id,
        )

    async def record_introspection_checkpoint_recorded(
        self,
        *,
        checkpoint_seq: int,
        milestone_id: str | None,
        review_tool_call_id: str | None,
        review_step_id: str | None,
    ) -> list[object]:
        return await self.emit(
            IntrospectionCheckpointRecorded,
            checkpoint_seq=checkpoint_seq,
            milestone_id=milestone_id,
            review_tool_call_id=review_tool_call_id,
            review_step_id=review_step_id,
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
        return await self.emit(
            IntrospectionOutcomeRecorded,
            aligned=aligned,
            experience=experience,
            tool_usefulness=list(tool_usefulness),
            active_milestone_id=active_milestone_id,
            review_tool_call_id=review_tool_call_id,
            review_step_id=review_step_id,
            boundary_seq=boundary_seq,
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
        return await self.emit(
            HookFailed,
            phase=phase,
            handler_name=handler_name,
            critical=critical,
            error=error,
            traceback=traceback,
        )


__all__ = ["RunStateWriter"]
