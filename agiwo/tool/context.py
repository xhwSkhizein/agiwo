from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolContext:
    """Narrow runtime context visible to plain tools."""

    session_id: str
    agent_id: str | None = None
    agent_name: str | None = None
    user_id: str | None = None
    timeout_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["ToolContext"]
