"""Shared serialization helpers for transport-facing payloads."""

from enum import Enum
from typing import Any


def serialize_optional_datetime(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def serialize_enum_value(value: Any) -> str:
    if isinstance(value, Enum):
        return value.value
    return str(value)


__all__ = ["serialize_enum_value", "serialize_optional_datetime"]
