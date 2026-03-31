"""Shared response serializers for console API and SSE boundaries.

This module is the single serialization layer between SDK/domain models and
API Pydantic schemas.
"""

from typing import Any

from agiwo.agent import Run, StepRecord
from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.observability.trace import Span, Trace
from agiwo.scheduler.models import AgentState, PendingEvent, WakeCondition, thaw_value
from agiwo.utils.serialization import (
    serialize_enum_value,
    serialize_optional_datetime,
)

from server.schemas import (
    AgentStateListItem,
    AgentStateResponse,
    RunMetricsSummary,
    PendingEventResponse,
    RunResponse,
    SpanResponse,
    StepResponse,
    TraceListItem,
    TraceResponse,
    WakeConditionResponse,
)


def serialize_user_input_payload(
    value: UserInput | dict[str, Any] | None,
) -> str | dict[str, Any] | list[Any] | None:
    return UserMessage.to_transport_payload(value)


# ── Response converters (SDK/domain → Pydantic schema) ───────────────────────


def step_to_response(step: StepRecord) -> StepResponse:
    return StepResponse(
        id=step.id,
        session_id=step.session_id,
        run_id=step.run_id,
        sequence=step.sequence,
        role=step.role.value,
        agent_id=step.agent_id,
        content=step.content,
        content_for_user=step.content_for_user,
        reasoning_content=step.reasoning_content,
        user_input=serialize_user_input_payload(step.user_input),
        tool_calls=step.tool_calls,
        tool_call_id=step.tool_call_id,
        name=step.name,
        metrics=step.metrics.to_dict() if step.metrics else None,
        created_at=serialize_optional_datetime(step.created_at),
        parent_run_id=step.parent_run_id,
        depth=step.depth,
    )


def run_to_response(run: Run) -> RunResponse:
    return RunResponse(
        id=run.id,
        agent_id=run.agent_id,
        session_id=run.session_id,
        user_id=run.user_id,
        user_input=UserMessage.to_storage_value(run.user_input),
        status=run.status.value,
        response_content=run.response_content,
        metrics=run.metrics.to_dict() if run.metrics else None,
        created_at=serialize_optional_datetime(run.created_at),
        updated_at=serialize_optional_datetime(run.updated_at),
        parent_run_id=run.parent_run_id,
    )


def span_to_response(span: Span) -> SpanResponse:
    return SpanResponse(
        span_id=span.span_id,
        trace_id=span.trace_id,
        parent_span_id=span.parent_span_id,
        kind=serialize_enum_value(span.kind),
        name=span.name,
        start_time=serialize_optional_datetime(span.start_time),
        end_time=serialize_optional_datetime(span.end_time),
        duration_ms=span.duration_ms,
        status=serialize_enum_value(span.status),
        error_message=span.error_message,
        depth=span.depth,
        attributes=span.attributes or {},
        input_preview=span.input_preview,
        output_preview=span.output_preview,
        metrics=span.metrics or {},
        llm_details=span.llm_details,
        tool_details=span.tool_details,
        run_id=span.run_id,
        step_id=span.step_id,
    )


def _trace_base_fields(trace: Trace) -> dict[str, Any]:
    return dict(
        trace_id=trace.trace_id,
        agent_id=trace.agent_id,
        session_id=trace.session_id,
        user_id=trace.user_id,
        start_time=serialize_optional_datetime(trace.start_time),
        duration_ms=trace.duration_ms,
        status=serialize_enum_value(trace.status),
        total_tokens=trace.total_tokens,
        total_input_tokens=trace.total_input_tokens,
        total_output_tokens=trace.total_output_tokens,
        total_llm_calls=trace.total_llm_calls,
        total_tool_calls=trace.total_tool_calls,
        total_cache_read_tokens=trace.total_cache_read_tokens,
        total_cache_creation_tokens=trace.total_cache_creation_tokens,
        total_token_cost=trace.total_token_cost,
    )


def trace_to_list_item(trace: Trace) -> TraceListItem:
    return TraceListItem(
        **_trace_base_fields(trace),
        input_query=trace.input_query[:200] if trace.input_query else None,
        final_output=trace.final_output[:200] if trace.final_output else None,
    )


def trace_to_response(trace: Trace) -> TraceResponse:
    return TraceResponse(
        **_trace_base_fields(trace),
        end_time=serialize_optional_datetime(trace.end_time),
        root_span_id=trace.root_span_id,
        max_depth=trace.max_depth,
        input_query=trace.input_query,
        final_output=trace.final_output,
        spans=[span_to_response(s) for s in (trace.spans or [])],
    )


def _wake_condition_to_response(
    wc: WakeCondition | None,
) -> WakeConditionResponse | None:
    if wc is None:
        return None
    return WakeConditionResponse(
        type=serialize_enum_value(wc.type),
        wait_for=list(wc.wait_for),
        wait_mode=serialize_enum_value(wc.wait_mode),
        completed_ids=list(wc.completed_ids),
        time_value=wc.time_value,
        time_unit=(
            serialize_enum_value(wc.time_unit) if wc.time_unit is not None else None
        ),
        wakeup_at=serialize_optional_datetime(wc.wakeup_at),
        timeout_at=serialize_optional_datetime(wc.timeout_at),
    )


def _state_base_fields(
    state: AgentState,
    metrics: RunMetricsSummary | None,
) -> dict[str, Any]:
    return dict(
        id=state.id,
        status=serialize_enum_value(state.status),
        task=serialize_user_input_payload(state.task),
        parent_id=state.parent_id,
        wake_condition=_wake_condition_to_response(state.wake_condition),
        agent_config_id=state.agent_config_id,
        is_persistent=state.is_persistent,
        depth=state.depth,
        wake_count=state.wake_count,
        metrics=metrics or RunMetricsSummary(),
        created_at=serialize_optional_datetime(state.created_at),
        updated_at=serialize_optional_datetime(state.updated_at),
    )


def state_to_list_item(
    state: AgentState,
    metrics: RunMetricsSummary | None = None,
) -> AgentStateListItem:
    return AgentStateListItem(
        **_state_base_fields(state, metrics),
        result_summary=state.result_summary[:200] if state.result_summary else None,
    )


def state_to_response(
    state: AgentState,
    metrics: RunMetricsSummary | None = None,
) -> AgentStateResponse:
    return AgentStateResponse(
        **_state_base_fields(state, metrics),
        session_id=state.session_id,
        pending_input=serialize_user_input_payload(state.pending_input),
        config_overrides=thaw_value(state.config_overrides),
        result_summary=state.result_summary,
        signal_propagated=state.signal_propagated,
    )


def pending_event_to_response(event: PendingEvent) -> PendingEventResponse:
    return PendingEventResponse(
        id=event.id,
        target_agent_id=event.target_agent_id,
        source_agent_id=event.source_agent_id,
        event_type=serialize_enum_value(event.event_type),
        payload=thaw_value(event.payload),
        created_at=serialize_optional_datetime(event.created_at),
    )
