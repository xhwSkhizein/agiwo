"""Scheduler store codec helpers.

This module owns persistence-facing serialization for scheduler domain types.
"""

from datetime import datetime
from typing import Any

from agiwo.agent import UserInput, UserMessage
from agiwo.scheduler.models import (
    ChildAgentConfigOverrides,
    TimeUnit,
    WaitMode,
    WakeCondition,
    WakeType,
)


def serialize_user_input_for_store(value: UserInput) -> str:
    stored = UserMessage.to_storage_value(value)
    if stored is None:
        raise TypeError("scheduler store requires non-null user input")
    return stored


def deserialize_user_input_for_store(value: str) -> UserInput:
    restored = UserMessage.from_storage_value(value)
    if restored is None:
        raise TypeError("scheduler store requires non-null user input")
    return restored


def serialize_child_agent_config_overrides(
    overrides: ChildAgentConfigOverrides,
) -> dict[str, str]:
    data: dict[str, str] = {}
    if overrides.instruction:
        data["instruction"] = overrides.instruction
    if overrides.system_prompt:
        data["system_prompt"] = overrides.system_prompt
    return data


def deserialize_child_agent_config_overrides(
    data: dict[str, Any] | None,
) -> ChildAgentConfigOverrides:
    if not data:
        return ChildAgentConfigOverrides()
    return ChildAgentConfigOverrides(
        instruction=data.get("instruction"),
        system_prompt=data.get("system_prompt"),
    )


def serialize_wake_condition_for_store(
    wake_condition: WakeCondition | None,
) -> dict[str, Any] | None:
    if wake_condition is None:
        return None

    result: dict[str, Any] = {"type": wake_condition.type.value}
    if wake_condition.wait_for:
        result["wait_for"] = list(wake_condition.wait_for)
    result["wait_mode"] = wake_condition.wait_mode.value
    if wake_condition.completed_ids:
        result["completed_ids"] = list(wake_condition.completed_ids)
    if wake_condition.time_value is not None:
        result["time_value"] = wake_condition.time_value
    if wake_condition.time_unit is not None:
        result["time_unit"] = wake_condition.time_unit.value
    if wake_condition.wakeup_at is not None:
        result["wakeup_at"] = wake_condition.wakeup_at.isoformat()
    if wake_condition.timeout_at is not None:
        result["timeout_at"] = wake_condition.timeout_at.isoformat()
    return result


def deserialize_wake_condition_for_store(data: dict[str, Any]) -> WakeCondition:
    wakeup_at = None
    if data.get("wakeup_at"):
        wakeup_at = datetime.fromisoformat(data["wakeup_at"])
    time_unit = None
    if data.get("time_unit"):
        time_unit = TimeUnit(data["time_unit"])
    timeout_at = None
    if data.get("timeout_at"):
        timeout_at = datetime.fromisoformat(data["timeout_at"])
    wait_mode = WaitMode.ALL
    if data.get("wait_mode"):
        wait_mode = WaitMode(data["wait_mode"])
    return WakeCondition(
        type=WakeType(data["type"]),
        wait_for=data.get("wait_for", []),
        wait_mode=wait_mode,
        completed_ids=data.get("completed_ids", []),
        time_value=data.get("time_value"),
        time_unit=time_unit,
        wakeup_at=wakeup_at,
        timeout_at=timeout_at,
    )


__all__ = [
    "deserialize_child_agent_config_overrides",
    "deserialize_user_input_for_store",
    "deserialize_wake_condition_for_store",
    "serialize_child_agent_config_overrides",
    "serialize_user_input_for_store",
    "serialize_wake_condition_for_store",
]
