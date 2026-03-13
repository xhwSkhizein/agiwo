"""Shared response serializers for console API and SSE boundaries."""

from typing import Any

from agiwo.agent import Run, StepRecord, StreamEvent
from agiwo.agent.serialization import (
    serialize_run_payload,
    serialize_step_record_payload,
    serialize_stream_event_payload,
)
from agiwo.observability.trace import Span, Trace
from agiwo.observability.serialization import (
    serialize_span_payload,
    serialize_trace_base_payload,
    serialize_trace_payload,
)
from agiwo.scheduler.models import AgentState, PendingEvent, WakeCondition
from agiwo.scheduler.serialization import (
    serialize_agent_state_payload,
    serialize_pending_event_payload,
    serialize_wake_condition_payload,
)

from server.domain.run_metrics import RunMetricsSummary
from server.schemas import (
    AgentStateListItem,
    AgentStateResponse,
    PendingEventResponse,
    RunResponse,
    SpanResponse,
    StepResponse,
    StreamEventPayload,
    TraceListItem,
    TraceResponse,
    WakeConditionResponse,
)


def step_to_response(step: StepRecord) -> StepResponse:
    return StepResponse(**serialize_step_record_payload(step))


def run_to_response(run: Run) -> RunResponse:
    return RunResponse(**serialize_run_payload(run))


def span_to_response(span: Span) -> SpanResponse:
    return SpanResponse(**serialize_span_payload(span))


def _trace_base_payload(trace: Trace) -> dict[str, Any]:
    payload = serialize_trace_base_payload(trace)
    return {
        "trace_id": payload["trace_id"],
        "agent_id": payload["agent_id"],
        "session_id": payload["session_id"],
        "user_id": payload["user_id"],
        "start_time": payload["start_time"],
        "duration_ms": payload["duration_ms"],
        "status": payload["status"],
        "total_tokens": payload["total_tokens"],
        "total_input_tokens": payload["total_input_tokens"],
        "total_output_tokens": payload["total_output_tokens"],
        "total_cache_read_tokens": payload["total_cache_read_tokens"],
        "total_cache_creation_tokens": payload["total_cache_creation_tokens"],
        "total_token_cost": payload["total_token_cost"],
        "total_llm_calls": payload["total_llm_calls"],
        "total_tool_calls": payload["total_tool_calls"],
    }


def trace_to_list_item(trace: Trace) -> TraceListItem:
    return TraceListItem(
        **_trace_base_payload(trace),
        input_query=trace.input_query[:200] if trace.input_query else None,
        final_output=trace.final_output[:200] if trace.final_output else None,
    )


def trace_to_response(trace: Trace) -> TraceResponse:
    return TraceResponse(**serialize_trace_payload(trace))


def wake_condition_to_response(wake_condition: WakeCondition | None) -> WakeConditionResponse | None:
    payload = serialize_wake_condition_payload(wake_condition)
    if payload is None:
        return None
    return WakeConditionResponse(**payload)


def state_to_list_item(
    state: AgentState,
    metrics: RunMetricsSummary | None = None,
) -> AgentStateListItem:
    payload = serialize_agent_state_payload(state)
    return AgentStateListItem(
        id=payload["id"],
        status=payload["status"],
        task=payload["task"],
        parent_id=payload["parent_id"],
        wake_condition=wake_condition_to_response(state.wake_condition),
        result_summary=state.result_summary[:200] if state.result_summary else None,
        agent_config_id=payload["agent_config_id"],
        is_persistent=payload["is_persistent"],
        depth=payload["depth"],
        wake_count=payload["wake_count"],
        metrics=metrics or RunMetricsSummary(),
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
    )


def state_to_response(
    state: AgentState,
    metrics: RunMetricsSummary | None = None,
) -> AgentStateResponse:
    payload = serialize_agent_state_payload(state)
    payload["wake_condition"] = wake_condition_to_response(state.wake_condition)
    return AgentStateResponse(
        **payload,
        metrics=metrics or RunMetricsSummary(),
    )


def pending_event_to_response(event: PendingEvent) -> PendingEventResponse:
    return PendingEventResponse(**serialize_pending_event_payload(event))


def stream_event_to_payload(event: StreamEvent) -> dict[str, Any]:
    return StreamEventPayload(**serialize_stream_event_payload(event)).model_dump()
