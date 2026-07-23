"""Shared serialization helpers for transport-facing payloads."""

from datetime import datetime
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


def parse_datetime(value: Any) -> datetime:
    """Accept datetime or ISO string."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"expected datetime or ISO string, got {type(value).__name__}")


def parse_optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    return parse_datetime(value)


def parse_optional_datetime_or_default(value: Any, default: datetime) -> datetime:
    if value is None:
        return default
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    if isinstance(value, datetime):
        return value
    return default


__all__ = [
    "parse_datetime",
    "parse_optional_datetime",
    "parse_optional_datetime_or_default",
    "serialize_enum_value",
    "serialize_optional_datetime",
]
