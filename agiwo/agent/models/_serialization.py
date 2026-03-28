"""Shared serialization helpers for agent models."""

import dataclasses
from datetime import datetime
from enum import Enum
from typing import Any

from agiwo.utils.serialization import serialize_optional_datetime


def fields_to_dict(obj: object) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field_info in dataclasses.fields(obj):
        value = getattr(obj, field_info.name)
        if isinstance(value, datetime):
            result[field_info.name] = serialize_optional_datetime(value)
        elif isinstance(value, Enum):
            result[field_info.name] = value.value
        else:
            result[field_info.name] = value
    return result
