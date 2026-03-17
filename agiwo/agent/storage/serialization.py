"""Shared Run/Step storage serialization helpers."""

from dataclasses import asdict, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import TypeAdapter

from agiwo.agent.input_codec import (
    deserialize_user_input,
    normalize_to_message,
    serialize_user_input,
    to_message_content,
)
from agiwo.agent.runtime import (
    MessageRole,
    Run,
    StepRecord,
)

_RUN_FIELD_NAMES = {field.name for field in fields(Run)}
_STEP_FIELD_NAMES = {field.name for field in fields(StepRecord)}
_RUN_ADAPTER = TypeAdapter(Run)
_STEP_ADAPTER = TypeAdapter(StepRecord)


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_none(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value if item is not None]
    return value


def _normalize_storage_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {
            key: _normalize_storage_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [
            _normalize_storage_value(item) for item in value if item is not None
        ]
    if isinstance(value, tuple):
        return [_normalize_storage_value(item) for item in value if item is not None]
    return value


def _as_storage_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Expected storage payload dict")
    return value


def _filter_storage_fields(
    data: dict[str, Any],
    field_names: set[str],
) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key in field_names}


def serialize_run_for_storage(run: Run) -> dict[str, Any]:
    data = _as_storage_dict(_normalize_storage_value(asdict(run)))
    data["status"] = run.status.value
    data["user_input"] = serialize_user_input(run.user_input)
    if run.metrics is not None:
        data["metrics"] = _normalize_storage_value(run.metrics)
    return _as_storage_dict(_drop_none(data))


def serialize_step_for_storage(step: StepRecord) -> dict[str, Any]:
    data = _as_storage_dict(_normalize_storage_value(asdict(step)))
    data["role"] = step.role.value
    if step.user_input is not None:
        data["user_input"] = serialize_user_input(step.user_input)
    if step.metrics is not None:
        data["metrics"] = _normalize_storage_value(step.metrics)
    if step.is_user_step() and step.user_input is not None:
        data.pop("content", None)
    return _as_storage_dict(_drop_none(data))


def deserialize_run_from_storage(data: dict[str, Any]) -> Run:
    run_data = _filter_storage_fields(data, _RUN_FIELD_NAMES)
    if isinstance(run_data.get("user_input"), str):
        run_data["user_input"] = deserialize_user_input(run_data["user_input"])
    return _RUN_ADAPTER.validate_python(run_data)


def deserialize_step_from_storage(data: dict[str, Any]) -> StepRecord:
    step_data = _filter_storage_fields(data, _STEP_FIELD_NAMES)
    if isinstance(step_data.get("user_input"), str):
        step_data["user_input"] = deserialize_user_input(step_data["user_input"])

    step = _STEP_ADAPTER.validate_python(step_data)

    if (
        step.role == MessageRole.USER
        and step.user_input is not None
        and step.content is None
    ):
        step.content = to_message_content(
            normalize_to_message(step.user_input).content
        )

    return step


__all__ = [
    "deserialize_run_from_storage",
    "deserialize_step_from_storage",
    "serialize_run_for_storage",
    "serialize_step_for_storage",
]
