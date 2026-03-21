"""Internal compaction runtime helpers."""

from agiwo.agent.engine.compaction.prompt import (
    DEFAULT_ASSISTANT_RESPONSE,
    DEFAULT_COMPACT_PROMPT,
)
from agiwo.agent.engine.compaction.runtime import CompactionRuntime

__all__ = [
    "CompactionRuntime",
    "DEFAULT_ASSISTANT_RESPONSE",
    "DEFAULT_COMPACT_PROMPT",
]
