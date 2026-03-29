from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RunContextLike(Protocol):
    """Minimal run-context contract required by ``BaseTool.build_context``."""

    session_id: str
    agent_id: str | None
    agent_name: str | None
    user_id: str | None
    timeout_at: float | None
    depth: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ToolContext:
    """Narrow runtime context visible to plain tools."""

    session_id: str
    agent_id: str | None = None
    agent_name: str | None = None
    user_id: str | None = None
    timeout_at: float | None = None
    depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    gate_checked: bool = False
    tool_call_id: str = ""


__all__ = ["RunContextLike", "ToolContext"]
