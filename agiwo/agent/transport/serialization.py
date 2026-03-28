"""Transport serialization for agent domain models."""

from typing import Any, Literal

from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.models.run import Run, RunMetrics
from agiwo.agent.models.step import StepDelta, StepMetrics, StepRecord
from agiwo.agent.models.stream import AgentStreamItem
from agiwo.utils.serialization import serialize_optional_datetime


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


def serialize_stream_item_payload(item: AgentStreamItem) -> dict[str, Any]:
    payload = item.to_dict()
    if item.type == "step_completed":
        payload["step"] = serialize_step_record_payload(item.step)
    elif item.type == "step_delta":
        payload["delta"] = serialize_step_delta_payload(item.delta)
    elif item.type == "run_completed":
        payload["metrics"] = serialize_run_metrics_payload(item.metrics)
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
