"""Serialize SDK and runtime objects into API view models and SSE payloads."""

import json
from typing import Any

from agiwo.agent import (
    AgentStreamItem,
    ContextStepsHiddenEvent,
    RunView,
    StepCompletedEvent,
    StepMetrics,
    StepView,
)
from agiwo.observability.trace import Trace
from agiwo.scheduler.models import (
    AgentState,
    PendingEvent,
    SchedulerRunResult,
    WakeCondition,
)
from server.models.metrics import RunMetricsSummary
from server.models.session import (
    ConversationEventRecord,
    ChannelChatContext,
    MilestoneRecord,
    ReviewCheckpointRecord,
    ReviewCycleRecord,
    ReviewOutcomeRecord,
    RuntimeDecisionRecord,
    Session,
    SessionDetailRecord,
    SessionMilestoneBoardRecord,
    SessionObservabilityRecord,
    SessionSummaryRecord,
    TraceLlmCallRecord,
    TraceMainlineEventRecord,
)
from server.models.view import (
    AgentStateListItem,
    AgentStateResponse,
    ChatContextResponse,
    ConversationEventResponse,
    MilestoneResponse,
    PendingEventResponse,
    ReviewCheckpointResponse,
    ReviewCycleResponse,
    ReviewOutcomeResponse,
    RunMetricsResponse,
    RunResponse,
    RuntimeDecisionResponse,
    SchedulerTreeNodeResponse,
    SchedulerRunResultResponse,
    SchedulerTreeResponse,
    SchedulerTreeStatsResponse,
    SessionDetailResponse,
    SessionMilestoneBoardResponse,
    SessionObservabilityResponse,
    SessionRecordResponse,
    SessionSummaryResponse,
    SpanResponse,
    StepMetricsResponse,
    StepResponse,
    TraceLlmCallResponse,
    TraceListItem,
    TraceMainlineEventResponse,
    TraceTimelineEventResponse,
    TraceResponse,
    WakeConditionResponse,
)
from server.services.runtime.runtime_observability import (
    build_trace_llm_call_records,
    build_trace_mainline_events,
    build_trace_review_cycles,
    build_trace_runtime_decisions,
    build_trace_timeline_events,
)
from server.services.runtime.scheduler_tree_view_service import SchedulerTreeRecord


def step_metrics_response_from_sdk(metrics: StepMetrics) -> StepMetricsResponse:
    return StepMetricsResponse(
        duration_ms=metrics.duration_ms,
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        total_tokens=metrics.total_tokens,
        cache_read_tokens=metrics.cache_read_tokens,
        cache_creation_tokens=metrics.cache_creation_tokens,
        token_cost=metrics.token_cost,
        usage_source=metrics.usage_source,
        model_name=metrics.model_name,
        provider=metrics.provider,
        first_token_latency_ms=metrics.first_token_latency_ms,
    )


def step_response_from_sdk(step_view: StepView) -> StepResponse:
    return StepResponse(
        id=step_view.id or f"{step_view.run_id}:{step_view.sequence}",
        session_id=step_view.session_id,
        run_id=step_view.run_id,
        sequence=step_view.sequence,
        role=step_view.role.value,
        agent_id=step_view.agent_id,
        content=step_view.content,
        content_for_user=step_view.content_for_user,
        reasoning_content=step_view.reasoning_content,
        user_input=step_view.user_input,
        tool_calls=step_view.tool_calls if step_view.tool_calls else None,
        tool_call_id=step_view.tool_call_id,
        name=step_view.name,
        condensed_content=step_view.condensed_content,
        metrics=(
            step_metrics_response_from_sdk(step_view.metrics)
            if step_view.metrics
            else None
        ),
        created_at=step_view.created_at.isoformat() if step_view.created_at else None,
        parent_run_id=step_view.parent_run_id,
        depth=step_view.depth,
    )


def run_response_from_sdk(run: RunView) -> RunResponse:
    return RunResponse(
        id=run.run_id,
        agent_id=run.agent_id,
        session_id=run.session_id,
        user_id=run.user_id,
        user_input=run.last_user_input,
        status=run.status.value,
        response_content=run.response,
        metrics=RunMetricsResponse(
            duration_ms=metrics.duration_ms,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            total_tokens=metrics.total_tokens,
            cache_read_tokens=metrics.cache_read_tokens,
            cache_creation_tokens=metrics.cache_creation_tokens,
            token_cost=metrics.token_cost,
            steps_count=metrics.steps_count,
            tool_calls_count=metrics.tool_calls_count,
        )
        if (metrics := run.metrics) is not None
        else None,
        created_at=run.created_at.isoformat() if run.created_at else None,
        updated_at=run.updated_at.isoformat() if run.updated_at else None,
        parent_run_id=run.parent_run_id,
    )


def session_record_response_from_runtime(session: Session) -> SessionRecordResponse:
    return SessionRecordResponse(
        id=session.id,
        chat_context_scope_id=session.chat_context_scope_id,
        base_agent_id=session.base_agent_id,
        created_by=session.created_by,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        source_session_id=session.source_session_id,
        fork_context_summary=session.fork_context_summary,
    )


def chat_context_response_from_runtime(
    chat_context: ChannelChatContext,
) -> ChatContextResponse:
    return ChatContextResponse(
        scope_id=chat_context.scope_id,
        channel_instance_id=chat_context.channel_instance_id,
        chat_id=chat_context.chat_id,
        chat_type=chat_context.chat_type,
        user_open_id=chat_context.user_open_id,
        base_agent_id=chat_context.base_agent_id,
        current_session_id=chat_context.current_session_id,
        created_at=chat_context.created_at.isoformat(),
        updated_at=chat_context.updated_at.isoformat(),
    )


def session_summary_response_from_record(
    summary: SessionSummaryRecord,
) -> SessionSummaryResponse:
    return SessionSummaryResponse(
        session_id=summary.session_id,
        agent_id=summary.base_agent_id,
        last_user_input=summary.last_user_input,
        last_response=summary.last_response,
        run_count=summary.run_count,
        step_count=summary.step_count,
        metrics=summary.metrics,
        created_at=summary.created_at.isoformat() if summary.created_at else None,
        updated_at=summary.updated_at.isoformat() if summary.updated_at else None,
        chat_context_scope_id=summary.chat_context_scope_id,
        created_by=summary.created_by,
        base_agent_id=summary.base_agent_id,
        root_state_status=summary.root_state_status,
        source_session_id=summary.source_session_id,
        fork_context_summary=summary.fork_context_summary,
    )


def runtime_decision_response_from_record(
    decision: RuntimeDecisionRecord,
) -> RuntimeDecisionResponse:
    return RuntimeDecisionResponse(
        kind=decision.kind,
        sequence=decision.sequence,
        run_id=decision.run_id,
        agent_id=decision.agent_id,
        created_at=decision.created_at.isoformat(),
        summary=decision.summary,
        details=dict(decision.details),
    )


def milestone_response_from_record(record: MilestoneRecord) -> MilestoneResponse:
    return MilestoneResponse(
        id=record.id,
        description=record.description,
        status=record.status,
        declared_at_seq=record.declared_at_seq,
        completed_at_seq=record.completed_at_seq,
    )


def review_checkpoint_response_from_record(
    record: ReviewCheckpointRecord,
) -> ReviewCheckpointResponse:
    return ReviewCheckpointResponse(
        seq=record.seq,
        milestone_id=record.milestone_id,
        confirmed_at=record.confirmed_at.isoformat(),
    )


def review_outcome_response_from_record(
    record: ReviewOutcomeRecord,
) -> ReviewOutcomeResponse:
    return ReviewOutcomeResponse(
        aligned=record.aligned,
        experience=record.experience,
        step_back_applied=record.step_back_applied,
        affected_count=record.affected_count,
        trigger_reason=record.trigger_reason,
        active_milestone=record.active_milestone,
        resolved_at=record.resolved_at.isoformat() if record.resolved_at else None,
    )


def session_milestone_board_response_from_record(
    record: SessionMilestoneBoardRecord,
) -> SessionMilestoneBoardResponse:
    return SessionMilestoneBoardResponse(
        session_id=record.session_id,
        run_id=record.run_id,
        milestones=[milestone_response_from_record(item) for item in record.milestones],
        active_milestone_id=record.active_milestone_id,
        latest_checkpoint=(
            review_checkpoint_response_from_record(record.latest_checkpoint)
            if record.latest_checkpoint is not None
            else None
        ),
        latest_review_outcome=(
            review_outcome_response_from_record(record.latest_review_outcome)
            if record.latest_review_outcome is not None
            else None
        ),
        pending_review_reason=record.pending_review_reason,
    )


def review_cycle_response_from_record(
    record: ReviewCycleRecord,
) -> ReviewCycleResponse:
    return ReviewCycleResponse(
        cycle_id=record.cycle_id,
        run_id=record.run_id,
        agent_id=record.agent_id,
        trigger_reason=record.trigger_reason,
        steps_since_last_review=record.steps_since_last_review,
        active_milestone=record.active_milestone,
        active_milestone_id=record.active_milestone_id,
        hook_advice=record.hook_advice,
        aligned=record.aligned,
        experience=record.experience,
        step_back_applied=record.step_back_applied,
        rollback_range=list(record.rollback_range) if record.rollback_range else None,
        affected_count=record.affected_count,
        started_at=record.started_at.isoformat() if record.started_at else None,
        resolved_at=record.resolved_at.isoformat() if record.resolved_at else None,
        raw_notice=record.raw_notice,
    )


def conversation_event_response_from_record(
    record: ConversationEventRecord,
) -> ConversationEventResponse:
    return ConversationEventResponse(
        id=record.id,
        session_id=record.session_id,
        run_id=record.run_id,
        sequence=record.sequence,
        kind=record.kind,
        priority=record.priority,
        title=record.title,
        summary=record.summary,
        details=dict(record.details),
    )


def session_observability_response_from_record(
    observability: SessionObservabilityRecord,
) -> SessionObservabilityResponse:
    return SessionObservabilityResponse(
        recent_traces=[
            trace_list_item_from_sdk(trace) for trace in observability.recent_traces
        ],
        decision_events=[
            runtime_decision_response_from_record(decision)
            for decision in observability.decision_events
        ],
    )


def session_detail_response_from_record(
    detail: SessionDetailRecord,
) -> SessionDetailResponse:
    return SessionDetailResponse(
        summary=session_summary_response_from_record(detail.summary),
        session=(
            session_record_response_from_runtime(detail.session)
            if detail.session is not None
            else None
        ),
        chat_context=(
            chat_context_response_from_runtime(detail.chat_context)
            if detail.chat_context is not None
            else None
        ),
        scheduler_state=(
            agent_state_response_from_sdk(detail.scheduler_state)
            if detail.scheduler_state is not None
            else None
        ),
        observability=(
            session_observability_response_from_record(detail.observability)
            if detail.observability is not None
            else None
        ),
        milestone_board=(
            session_milestone_board_response_from_record(detail.milestone_board)
            if detail.milestone_board is not None
            else None
        ),
        review_cycles=[
            review_cycle_response_from_record(record) for record in detail.review_cycles
        ],
        conversation_events=[
            conversation_event_response_from_record(record)
            for record in detail.conversation_events
        ],
    )


def _wake_condition_response_from_sdk(
    wake_condition: WakeCondition,
) -> WakeConditionResponse:
    wakeup_at = getattr(wake_condition, "wakeup_at", None)
    timeout_at = getattr(wake_condition, "timeout_at", None)
    if wakeup_at is not None and hasattr(wakeup_at, "isoformat"):
        wakeup_at = wakeup_at.isoformat()
    if timeout_at is not None and hasattr(timeout_at, "isoformat"):
        timeout_at = timeout_at.isoformat()
    return WakeConditionResponse(
        type=wake_condition.type.value
        if hasattr(wake_condition.type, "value")
        else str(wake_condition.type),
        wait_for=list(getattr(wake_condition, "wait_for", []) or []),
        wait_mode=wake_condition.wait_mode.value
        if hasattr(wake_condition.wait_mode, "value")
        else str(getattr(wake_condition, "wait_mode", "all")),
        completed_ids=list(getattr(wake_condition, "completed_ids", []) or []),
        time_value=getattr(wake_condition, "time_value", None),
        time_unit=wake_condition.time_unit.value
        if hasattr(getattr(wake_condition, "time_unit", None), "value")
        else getattr(wake_condition, "time_unit", None),
        wakeup_at=wakeup_at,
        timeout_at=timeout_at,
    )


def scheduler_run_result_response_from_sdk(
    result: SchedulerRunResult | None,
) -> SchedulerRunResultResponse | None:
    if result is None:
        return None
    return SchedulerRunResultResponse(
        run_id=result.run_id,
        termination_reason=result.termination_reason.value,
        summary=result.summary,
        error=result.error,
        completed_at=(
            result.completed_at.isoformat()
            if getattr(result, "completed_at", None) is not None
            else None
        ),
    )


def agent_state_response_from_sdk(
    state: AgentState,
    *,
    root_state_id: str | None = None,
) -> AgentStateResponse:
    return AgentStateResponse(
        id=state.id,
        root_state_id=root_state_id or (state.id if state.parent_id is None else None),
        status=state.status.value
        if hasattr(state.status, "value")
        else str(state.status),
        task=state.task,
        parent_id=state.parent_id,
        wake_condition=(
            _wake_condition_response_from_sdk(state.wake_condition)
            if getattr(state, "wake_condition", None) is not None
            else None
        ),
        result_summary=state.result_summary,
        last_run_result=scheduler_run_result_response_from_sdk(state.last_run_result),
        agent_config_id=state.agent_config_id,
        is_persistent=state.is_persistent,
        depth=state.depth,
        wake_count=state.wake_count,
        metrics=getattr(state, "metrics", None) or RunMetricsSummary(),
        created_at=state.created_at.isoformat() if state.created_at else None,
        updated_at=state.updated_at.isoformat() if state.updated_at else None,
        session_id=state.session_id,
        pending_input=state.pending_input,
        config_overrides=state.config_overrides,
        signal_propagated=state.signal_propagated,
    )


def agent_state_list_item_from_sdk(
    state: AgentState,
    *,
    root_state_id: str | None = None,
) -> AgentStateListItem:
    return AgentStateListItem(
        id=state.id,
        root_state_id=root_state_id or (state.id if state.parent_id is None else None),
        status=state.status.value
        if hasattr(state.status, "value")
        else str(state.status),
        task=state.task,
        parent_id=state.parent_id,
        wake_condition=None,
        result_summary=state.result_summary,
        last_run_result=scheduler_run_result_response_from_sdk(state.last_run_result),
        agent_config_id=state.agent_config_id,
        is_persistent=state.is_persistent,
        depth=state.depth,
        wake_count=state.wake_count,
        metrics=getattr(state, "metrics", None) or RunMetricsSummary(),
        created_at=state.created_at.isoformat() if state.created_at else None,
        updated_at=state.updated_at.isoformat() if state.updated_at else None,
    )


def scheduler_tree_response_from_record(
    tree: SchedulerTreeRecord,
) -> SchedulerTreeResponse:
    return SchedulerTreeResponse(
        root_state_id=tree.root_state_id,
        root_session_id=tree.root_session_id,
        nodes=[
            SchedulerTreeNodeResponse(
                state_id=node.state_id,
                root_state_id=node.root_state_id,
                parent_state_id=node.parent_state_id,
                child_ids=node.child_ids,
                session_id=node.session_id,
                agent_id=node.agent_id,
                task_id=node.task_id,
                status=node.status,
                depth=node.depth,
                created_at=node.created_at.isoformat() if node.created_at else None,
                updated_at=node.updated_at.isoformat() if node.updated_at else None,
                completed_at=(
                    node.completed_at.isoformat() if node.completed_at else None
                ),
                wake_condition=(
                    _wake_condition_response_from_sdk(node.wake_condition)
                    if node.wake_condition is not None
                    else None
                ),
                pending_event_count=node.pending_event_count,
                last_error=node.last_error,
                result_summary=node.result_summary,
                last_run_result=scheduler_run_result_response_from_sdk(
                    node.last_run_result
                ),
            )
            for node in tree.nodes
        ],
        stats=SchedulerTreeStatsResponse(
            total=tree.stats.total,
            running=tree.stats.running,
            waiting=tree.stats.waiting,
            queued=tree.stats.queued,
            idle=tree.stats.idle,
            completed=tree.stats.completed,
            failed=tree.stats.failed,
            cancelled=tree.stats.cancelled,
        ),
        generated_at=tree.generated_at.isoformat(),
    )


def pending_event_response_from_sdk(event: PendingEvent) -> PendingEventResponse:
    created_at = getattr(event, "created_at", None)
    return PendingEventResponse(
        id=event.id,
        target_agent_id=event.target_agent_id,
        source_agent_id=event.source_agent_id,
        event_type=event.event_type,
        payload=dict(getattr(event, "payload", {}) or {}),
        created_at=created_at.isoformat()
        if hasattr(created_at, "isoformat")
        else created_at,
    )


def _trace_base_kwargs(trace: Trace) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "agent_id": getattr(trace, "agent_id", None),
        "session_id": getattr(trace, "session_id", None),
        "user_id": getattr(trace, "user_id", None),
        "start_time": trace.start_time.isoformat()
        if getattr(trace, "start_time", None)
        else None,
        "duration_ms": getattr(trace, "duration_ms", None),
        "status": getattr(trace, "status", "unknown"),
        "total_tokens": getattr(trace, "total_tokens", 0),
        "total_input_tokens": getattr(trace, "total_input_tokens", 0),
        "total_output_tokens": getattr(trace, "total_output_tokens", 0),
        "total_llm_calls": getattr(trace, "total_llm_calls", 0),
        "total_tool_calls": getattr(trace, "total_tool_calls", 0),
        "total_cache_read_tokens": getattr(trace, "total_cache_read_tokens", 0),
        "total_cache_creation_tokens": getattr(trace, "total_cache_creation_tokens", 0),
        "total_token_cost": getattr(trace, "total_token_cost", 0.0),
        "input_query": getattr(trace, "input_query", None),
        "final_output": getattr(trace, "final_output", None),
    }


def trace_list_item_from_sdk(trace: Trace) -> TraceListItem:
    return TraceListItem(**_trace_base_kwargs(trace))


def trace_response_from_sdk(trace: Trace) -> TraceResponse:
    spans = [
        SpanResponse(
            span_id=span.span_id,
            trace_id=span.trace_id,
            parent_span_id=getattr(span, "parent_span_id", None),
            kind=getattr(span, "kind", ""),
            name=getattr(span, "name", ""),
            start_time=span.start_time.isoformat()
            if getattr(span, "start_time", None)
            else None,
            end_time=span.end_time.isoformat()
            if getattr(span, "end_time", None)
            else None,
            duration_ms=getattr(span, "duration_ms", None),
            status=getattr(span, "status", "unknown"),
            error_message=getattr(span, "error_message", None),
            depth=getattr(span, "depth", 0),
            attributes=dict(getattr(span, "attributes", {}) or {}),
            input_preview=getattr(span, "input_preview", None),
            output_preview=getattr(span, "output_preview", None),
            metrics=dict(getattr(span, "metrics", {}) or {}),
            llm_details=getattr(span, "llm_details", None),
            tool_details=getattr(span, "tool_details", None),
            run_id=getattr(span, "run_id", None),
            step_id=getattr(span, "step_id", None),
        )
        for span in getattr(trace, "spans", []) or []
    ]
    runtime_decisions = [
        runtime_decision_response_from_record(decision)
        for decision in build_trace_runtime_decisions(trace)
    ]
    timeline_events = [
        TraceTimelineEventResponse(
            kind=event.kind,
            timestamp=event.timestamp.isoformat() if event.timestamp else None,
            sequence=event.sequence,
            run_id=event.run_id,
            agent_id=event.agent_id,
            span_id=event.span_id,
            step_id=event.step_id,
            title=event.title,
            summary=event.summary,
            status=event.status,
            details=dict(event.details),
        )
        for event in build_trace_timeline_events(trace)
    ]
    mainline_events = [
        trace_mainline_event_response_from_record(event)
        for event in build_trace_mainline_events(trace)
    ]
    review_cycles = [
        review_cycle_response_from_record(record)
        for record in build_trace_review_cycles(trace)
    ]
    llm_calls = [
        trace_llm_call_response_from_record(record)
        for record in build_trace_llm_call_records(trace)
    ]
    return TraceResponse(
        **_trace_base_kwargs(trace),
        end_time=trace.end_time.isoformat()
        if getattr(trace, "end_time", None)
        else None,
        root_span_id=getattr(trace, "root_span_id", None),
        max_depth=getattr(trace, "max_depth", 0),
        spans=spans,
        runtime_decisions=runtime_decisions,
        timeline_events=timeline_events,
        mainline_events=mainline_events,
        review_cycles=review_cycles,
        llm_calls=llm_calls,
    )


def trace_mainline_event_response_from_record(
    record: TraceMainlineEventRecord,
) -> TraceMainlineEventResponse:
    return TraceMainlineEventResponse(
        id=record.id,
        kind=record.kind,
        title=record.title,
        summary=record.summary,
        status=record.status,
        sequence=record.sequence,
        timestamp=record.timestamp.isoformat() if record.timestamp else None,
        run_id=record.run_id,
        agent_id=record.agent_id,
        details=dict(record.details),
    )


def trace_llm_call_response_from_record(
    record: TraceLlmCallRecord,
) -> TraceLlmCallResponse:
    return TraceLlmCallResponse(
        span_id=record.span_id,
        run_id=record.run_id,
        agent_id=record.agent_id,
        model=record.model,
        provider=record.provider,
        finish_reason=record.finish_reason,
        duration_ms=record.duration_ms,
        first_token_latency_ms=record.first_token_latency_ms,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        total_tokens=record.total_tokens,
        message_count=record.message_count,
        tool_schema_count=record.tool_schema_count,
        response_tool_call_count=record.response_tool_call_count,
        output_preview=record.output_preview,
    )


def stream_event_to_payload(event: AgentStreamItem) -> dict[str, Any]:
    if isinstance(event, StepCompletedEvent):
        return {
            "type": "step",
            "step": step_response_from_sdk(event.step).model_dump(),
        }
    if isinstance(event, ContextStepsHiddenEvent):
        return event.to_dict()
    return {"type": "unknown", "data": str(event)}


def stream_event_to_sse_message(event: AgentStreamItem) -> dict[str, str]:
    return {
        "event": event.type,
        "data": json.dumps(event.to_dict(), default=str),
    }
