"""Transport serialization for agent-domain models."""

import json
from typing import Any, Literal

from agiwo.agent.input import ContentPart, UserInput, UserMessage
from agiwo.agent.input_codec import deserialize_user_input, serialize_user_input
from agiwo.agent.runtime import (
    AgentStreamItem,
    AgentStreamItemBase,
    Run,
    RunCompletedEvent,
    RunFailedEvent,
    RunMetrics,
    RunStartedEvent,
    StepDelta,
    StepDeltaEvent,
    StepCompletedEvent,
    StepMetrics,
    StepRecord,
)
from agiwo.utils.serialization import serialize_enum_value, serialize_optional_datetime


def _deserialize_string_user_input(value: str) -> UserInput | str:
    decoded = deserialize_user_input(value)
    if decoded is value:
        return value
    return decoded


def _serialize_content_parts_payload(parts: list[Any]) -> list[Any]:
    payload: list[Any] = []
    for item in parts:
        payload.append(item.to_dict() if isinstance(item, ContentPart) else item)
    return payload


def _serialize_user_message_payload(message: UserMessage) -> dict[str, Any]:
    return {
        "content": _serialize_content_parts_payload(message.content),
        "context": message.context.to_dict() if message.context else None,
    }


def _normalize_tagged_user_input(value: dict[str, Any]) -> UserInput | dict[str, Any]:
    input_type = value.get("__type")
    if input_type in {"content_parts", "user_message"}:
        return deserialize_user_input(json.dumps(value))
    return value


def serialize_user_input_payload(
    value: UserInput | dict[str, Any] | None,
) -> str | dict[str, Any] | list[Any] | None:
    normalized = (
        _deserialize_string_user_input(value) if isinstance(value, str) else value
    )
    if isinstance(normalized, dict):
        normalized = _normalize_tagged_user_input(normalized)
    if isinstance(normalized, UserMessage):
        return _serialize_user_message_payload(normalized)
    if isinstance(normalized, list):
        return _serialize_content_parts_payload(normalized)
    return normalized


def serialize_run_user_input_payload(
    user_input: UserInput | dict[str, Any] | None,
    *,
    mode: Literal["stored", "structured"] = "stored",
) -> str | dict[str, Any] | list[Any] | None:
    if mode == "structured":
        return serialize_user_input_payload(user_input)
    if user_input is None or isinstance(user_input, str):
        return user_input
    return serialize_user_input(user_input)


def serialize_step_metrics_payload(
    metrics: StepMetrics | None,
) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return metrics.to_dict()


def serialize_run_metrics_payload(metrics: RunMetrics | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return metrics.to_dict()


def serialize_step_delta_payload(delta: StepDelta | None) -> dict[str, Any] | None:
    if delta is None:
        return None
    return delta.to_dict()


def serialize_step_record_payload(step: StepRecord) -> dict[str, Any]:
    return {
        "id": step.id,
        "session_id": step.session_id,
        "run_id": step.run_id,
        "sequence": step.sequence,
        "role": serialize_enum_value(step.role),
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
        "status": serialize_enum_value(run.status),
        "response_content": run.response_content,
        "metrics": serialize_run_metrics_payload(run.metrics),
        "created_at": serialize_optional_datetime(run.created_at),
        "updated_at": serialize_optional_datetime(run.updated_at),
        "parent_run_id": run.parent_run_id,
    }


def _serialize_stream_item_base(item: AgentStreamItemBase) -> dict[str, Any]:
    return {
        "type": item.type,
        "session_id": item.session_id,
        "run_id": item.run_id,
        "agent_id": item.agent_id,
        "parent_run_id": item.parent_run_id,
        "depth": item.depth,
        "timestamp": serialize_optional_datetime(item.timestamp),
    }


def serialize_stream_item_payload(item: AgentStreamItem) -> dict[str, Any]:
    payload = _serialize_stream_item_base(item)
    if isinstance(item, RunStartedEvent):
        pass
    elif isinstance(item, StepDeltaEvent):
        payload["step_id"] = item.step_id
        payload["delta"] = serialize_step_delta_payload(item.delta)
    elif isinstance(item, StepCompletedEvent):
        payload["step"] = serialize_step_record_payload(item.step)
    elif isinstance(item, RunCompletedEvent):
        payload["response"] = item.response
        payload["metrics"] = serialize_run_metrics_payload(item.metrics)
        payload["termination_reason"] = (
            serialize_enum_value(item.termination_reason)
            if item.termination_reason is not None
            else None
        )
    elif isinstance(item, RunFailedEvent):
        payload["error"] = item.error
    else:
        raise TypeError(f"Unsupported stream item type: {type(item)!r}")
    return payload


__all__ = [
    "serialize_run_metrics_payload",
    "serialize_run_payload",
    "serialize_run_user_input_payload",
    "serialize_step_delta_payload",
    "serialize_step_metrics_payload",
    "serialize_step_record_payload",
    "serialize_stream_item_payload",
    "serialize_user_input_payload",
]
