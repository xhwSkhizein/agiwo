import json
from typing import Any

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def _try_load_json(payload: str) -> dict[str, Any] | None:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _find_outer_json_bounds(response_content: str, json_start: int) -> tuple[int, int]:
    depth = 0
    in_string = False
    escape_next = False

    for index, char in enumerate(response_content[json_start:], start=json_start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return json_start, index + 1

    return json_start, -1


def parse_compact_response(response_content: str) -> dict[str, Any]:
    """Parse the model response and extract the outermost JSON object when present."""
    stripped = response_content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        parsed = _try_load_json(stripped)
        if parsed is not None:
            return parsed

    json_start = response_content.find("{")
    if json_start < 0:
        return {"summary": response_content}

    _, json_end = _find_outer_json_bounds(response_content, json_start)
    if json_end > json_start:
        parsed = _try_load_json(response_content[json_start:json_end])
        if parsed is not None:
            return parsed

    logger.warning(
        "compact_json_parse_failed",
        response_preview=response_content[:200],
    )
    return {"summary": response_content}


__all__ = ["parse_compact_response"]
