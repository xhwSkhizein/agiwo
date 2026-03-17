"""Transport serialization for scheduler-domain models."""

from typing import Any

from agiwo.agent.serialization import serialize_user_input_payload
from agiwo.scheduler.models import AgentState, PendingEvent, WakeCondition
from agiwo.utils.serialization import serialize_enum_value, serialize_optional_datetime


def serialize_wake_condition_payload(
    wake_condition: WakeCondition | None,
) -> dict[str, Any] | None:
    if wake_condition is None:
        return None
    return {
        "type": serialize_enum_value(wake_condition.type),
        "wait_for": wake_condition.wait_for,
        "wait_mode": serialize_enum_value(wake_condition.wait_mode),
        "completed_ids": wake_condition.completed_ids,
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
        "config_overrides": state.config_overrides,
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
        "payload": event.payload,
        "created_at": serialize_optional_datetime(event.created_at),
    }


__all__ = [
    "serialize_agent_state_payload",
    "serialize_pending_event_payload",
    "serialize_wake_condition_payload",
]
