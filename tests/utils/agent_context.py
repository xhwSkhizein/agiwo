from typing import Any

from agiwo.tool.context import ToolContext


def build_tool_context(
    *,
    session_id: str = "test-session",
    run_id: str = "test-run",
    agent_id: str = "test-agent",
    agent_name: str = "test-agent",
    user_id: str | None = None,
    timeout_at: float | None = None,
    depth: int = 0,
    session_runtime: object | None = None,
    metadata: dict[str, Any] | None = None,
    gate_checked: bool = False,
) -> ToolContext:
    return ToolContext(
        session_id=session_id,
        agent_id=agent_id,
        agent_name=agent_name,
        user_id=user_id,
        timeout_at=timeout_at,
        depth=depth,
        metadata=dict(metadata or {}),
        gate_checked=gate_checked,
    )


__all__ = ["build_tool_context"]
