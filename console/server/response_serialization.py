"""Shared response serializers for console API and SSE boundaries.

This module is the single serialization layer between SDK/domain models and
API Pydantic schemas.  Dict-building helpers that previously lived under
``server.serialization.*`` are consolidated here.
"""

from typing import Any, Literal

from agiwo.agent import AgentStreamItem, Run, StepRecord
from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.models.run import RunMetrics
from agiwo.agent.models.step import StepMetrics
from agiwo.observability.trace import Span, Trace
from agiwo.scheduler.models import AgentState, PendingEvent, WakeCondition, thaw_value
from agiwo.utils.serialization import (
    serialize_enum_value,
    serialize_optional_datetime,
)

from server.domain.run_metrics import RunMetricsSummary
from server.schemas import (
    AgentStateListItem,
    AgentStateResponse,
    PendingEventResponse,
    RunResponse,
    SpanResponse,
    StepResponse,
    TraceListItem,
    TraceResponse,
    WakeConditionResponse,
)


# ── Agent serialization helpers ──────────────────────────────────────────────


def serialize_user_input_payload(
    value: UserInput | dict[str, Any] | None,
) -> str | dict[str, Any] | list[Any] | None:
    return UserMessage.to_transport_payload(value)


def serialize_run_user_input_payload(
    user_input: UserInput | dict[str, Any] | None,
    *,
    mode: Literal["stored", "structured"] = "stored",
) -> str | dict[str, Any] | list[Any] | None:
    if mode == "structured":
        return serialize_user_input_payload(user_input)
    return UserMessage.to_storage_value(user_input)


def serialize_step_metrics_payload(
    metrics: StepMetrics | None,
) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return metrics.to_dict()


def serialize_run_metrics_payload(
    metrics: RunMetrics | None,
) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return metrics.to_dict()


def serialize_step_record_payload(step: StepRecord) -> dict[str, Any]:
    return {
        "id": step.id,
        "session_id": step.session_id,
        "run_id": step.run_id,
        "sequence": step.sequence,
        "role": step.role.value,
        "agent_id": step.agent_id,
        "content": step.content,
        "content_for_user": step.content_for_user,
        "reasoning_content": step.reasoning_content,
        "user_input": serialize_user_input_payload(step.user_input),
        "tool_calls": step.tool_calls,
        "tool_call_id": step.tool_call_id,
        "name": step.name,
        "metrics": serialize_step_metrics_payload(step.metrics),
        "created_at": serialize_optional_datetime(step.created_at),
        "parent_run_id": step.parent_run_id,
        "depth": step.depth,
    }


def serialize_run_payload(
    run: Run,
    *,
    user_input_mode: Literal["stored", "structured"] = "stored",
) -> dict[str, Any]:
    return {
        "id": run.id,
        "agent_id": run.agent_id,
        "session_id": run.session_id,
        "user_id": run.user_id,
        "user_input": serialize_run_user_input_payload(
            run.user_input,
            mode=user_input_mode,
        ),
        "status": run.status.value,
        "response_content": run.response_content,
        "metrics": serialize_run_metrics_payload(run.metrics),
        "created_at": serialize_optional_datetime(run.created_at),
        "updated_at": serialize_optional_datetime(run.updated_at),
        "parent_run_id": run.parent_run_id,
    }


# ── Observability serialization helpers ──────────────────────────────────────


def serialize_span_payload(span: Span) -> dict[str, Any]:
    return {
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "parent_span_id": span.parent_span_id,
        "kind": serialize_enum_value(span.kind),
        "name": span.name,
        "start_time": serialize_optional_datetime(span.start_time),
        "end_time": serialize_optional_datetime(span.end_time),
        "duration_ms": span.duration_ms,
        "status": serialize_enum_value(span.status),
        "error_message": span.error_message,
        "depth": span.depth,
        "attributes": span.attributes or {},
        "input_preview": span.input_preview,
        "output_preview": span.output_preview,
        "metrics": span.metrics or {},
        "llm_details": span.llm_details,
        "tool_details": span.tool_details,
        "run_id": span.run_id,
        "step_id": span.step_id,
    }


def serialize_trace_base_payload(trace: Trace) -> dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "agent_id": trace.agent_id,
        "session_id": trace.session_id,
        "user_id": trace.user_id,
        "start_time": serialize_optional_datetime(trace.start_time),
        "end_time": serialize_optional_datetime(trace.end_time),
        "duration_ms": trace.duration_ms,
        "status": serialize_enum_value(trace.status),
        "root_span_id": trace.root_span_id,
        "total_tokens": trace.total_tokens,
        "total_input_tokens": trace.total_input_tokens,
        "total_output_tokens": trace.total_output_tokens,
        "total_llm_calls": trace.total_llm_calls,
        "total_tool_calls": trace.total_tool_calls,
        "total_cache_read_tokens": trace.total_cache_read_tokens,
        "total_cache_creation_tokens": trace.total_cache_creation_tokens,
        "total_token_cost": trace.total_token_cost,
        "max_depth": trace.max_depth,
        "input_query": trace.input_query,
        "final_output": trace.final_output,
    }


def serialize_trace_payload(
    trace: Trace,
    *,
    include_spans: bool = True,
) -> dict[str, Any]:
    payload = serialize_trace_base_payload(trace)
    if include_spans:
        payload["spans"] = [
            serialize_span_payload(span) for span in (trace.spans or [])
        ]
    return payload


# ── Scheduler serialization helpers ──────────────────────────────────────────


def serialize_wake_condition_payload(
    wake_condition: WakeCondition | None,
) -> dict[str, Any] | None:
    if wake_condition is None:
        return None
    return {
        "type": serialize_enum_value(wake_condition.type),
        "wait_for": list(wake_condition.wait_for),
        "wait_mode": serialize_enum_value(wake_condition.wait_mode),
        "completed_ids": list(wake_condition.completed_ids),
        "time_value": wake_condition.time_value,
        "time_unit": (
            serialize_enum_value(wake_condition.time_unit)
            if wake_condition.time_unit is not None
            else None
        ),
        "wakeup_at": serialize_optional_datetime(wake_condition.wakeup_at),
        "timeout_at": serialize_optional_datetime(wake_condition.timeout_at),
    }


def serialize_agent_state_payload(state: AgentState) -> dict[str, Any]:
    return {
        "id": state.id,
        "session_id": state.session_id,
        "status": serialize_enum_value(state.status),
        "task": serialize_user_input_payload(state.task),
        "parent_id": state.parent_id,
        "pending_input": serialize_user_input_payload(state.pending_input),
        "config_overrides": thaw_value(state.config_overrides),
        "wake_condition": serialize_wake_condition_payload(state.wake_condition),
        "result_summary": state.result_summary,
        "signal_propagated": state.signal_propagated,
        "agent_config_id": state.agent_config_id,
        "is_persistent": state.is_persistent,
        "depth": state.depth,
        "wake_count": state.wake_count,
        "created_at": serialize_optional_datetime(state.created_at),
        "updated_at": serialize_optional_datetime(state.updated_at),
    }


def serialize_pending_event_payload(event: PendingEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "target_agent_id": event.target_agent_id,
        "source_agent_id": event.source_agent_id,
        "event_type": serialize_enum_value(event.event_type),
        "payload": thaw_value(event.payload),
        "created_at": serialize_optional_datetime(event.created_at),
    }


# ── Response converters (dict → Pydantic schema) ────────────────────────────


def step_to_response(step: StepRecord) -> StepResponse:
    return StepResponse(**serialize_step_record_payload(step))


def run_to_response(run: Run) -> RunResponse:
    return RunResponse(**serialize_run_payload(run))


def span_to_response(span: Span) -> SpanResponse:
    return SpanResponse(**serialize_span_payload(span))


def trace_to_list_item(trace: Trace) -> TraceListItem:
    base = serialize_trace_base_payload(trace)
    base["input_query"] = trace.input_query[:200] if trace.input_query else None
    base["final_output"] = trace.final_output[:200] if trace.final_output else None
    return TraceListItem.model_validate(base)


def trace_to_response(trace: Trace) -> TraceResponse:
    return TraceResponse(**serialize_trace_payload(trace))


def wake_condition_to_response(
    wake_condition: WakeCondition | None,
) -> WakeConditionResponse | None:
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


def stream_event_to_payload(event: AgentStreamItem) -> dict[str, Any]:
    payload = event.to_dict()
    if event.type == "step_completed":
        payload["step"] = step_to_response(event.step).model_dump()
    return payload
