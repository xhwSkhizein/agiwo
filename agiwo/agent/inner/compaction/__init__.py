"""Internal compaction runtime helpers."""

from agiwo.agent.inner.compaction.prompt import (
    DEFAULT_ASSISTANT_RESPONSE,
    DEFAULT_COMPACT_PROMPT,
)
from agiwo.agent.inner.compaction.runtime import CompactionRuntime

__all__ = [
    "CompactionRuntime",
    "DEFAULT_ASSISTANT_RESPONSE",
    "DEFAULT_COMPACT_PROMPT",
]
