"""Memory-domain record types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    content: str
    relevance_score: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["MemoryRecord"]
