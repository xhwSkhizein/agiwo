"""Lightweight entry complexity assessment for first Objective user input."""

import asyncio
import json
import re
from typing import Any

from agiwo.llm.base import Model, StreamChunk
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

COMPLEXITY_PLANNING_THRESHOLD = 3
_DEFAULT_TIMEOUT_SECONDS = 8.0

_PLANNING_NOTICE = (
    '<system_notice origin="entry_complexity">\n'
    "This objective looks moderately complex. Consider using the update_plan "
    "tool to declare verifiable milestones before diving into implementation.\n"
    "</system_notice>"
)

_SCORE_PROMPT = (
    "Rate how complex the following user request is for an autonomous agent "
    "to complete, on a scale from 0 (trivial single-step) to 10 (large "
    "multi-phase project). Reply with JSON only: "
    '{"score": <integer 0-10>}.\n\n'
    "User request:\n"
)


async def assess_entry_complexity(
    text: str,
    *,
    model: Model | None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> int | None:
    """Return 0–10 complexity score, or None when skipped or on failure."""
    if model is None or not text.strip():
        return None
    messages = [
        {"role": "user", "content": _SCORE_PROMPT + text.strip()},
    ]
    try:
        raw = await asyncio.wait_for(
            _collect_model_text(model, messages),
            timeout=timeout_seconds,
        )
    except Exception:  # noqa: BLE001
        logger.info("objective_entry_complexity_failed", exc_info=True)
        return None
    score = _parse_score(raw)
    if score is None:
        logger.info("objective_entry_complexity_unparseable", raw=raw[:200])
    return score


def planning_notice_for_score(score: int | None, *, threshold: int) -> str | None:
    if score is None or score <= threshold:
        return None
    return _PLANNING_NOTICE


async def _collect_model_text(model: Model, messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    async for chunk in model.arun_stream(messages):
        if isinstance(chunk, StreamChunk) and chunk.content:
            parts.append(chunk.content)
    return "".join(parts).strip()


def _parse_score(raw: str) -> int | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "score" in data:
            return _clamp_score(data["score"])
    except json.JSONDecodeError:
        pass
    match = re.search(r"\b(\d{1,2})\b", raw)
    if match is not None:
        return _clamp_score(int(match.group(1)))
    return None


def _clamp_score(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 0 <= value <= 10:
        return value
    return None


__all__ = [
    "COMPLEXITY_PLANNING_THRESHOLD",
    "assess_entry_complexity",
    "planning_notice_for_score",
]
