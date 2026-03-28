"""Runtime-only tool context for nested agent execution."""

from dataclasses import dataclass

from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.tool.context import ToolContext


@dataclass(frozen=True, kw_only=True)
class AgentToolContext(ToolContext):
    parent_run_id: str
    session_runtime: SessionRuntime

    @classmethod
    def from_run_context(
        cls,
        ctx: RunContext,
        *,
        timeout_at: float | None,
        gate_checked: bool = True,
        tool_call_id: str = "",
    ) -> "AgentToolContext":
        return cls(
            session_id=ctx.session_id,
            agent_id=ctx.agent_id,
            agent_name=ctx.agent_name,
            user_id=ctx.user_id,
            timeout_at=timeout_at,
            depth=ctx.depth,
            metadata=dict(ctx.metadata),
            gate_checked=gate_checked,
            tool_call_id=tool_call_id,
            parent_run_id=ctx.run_id,
            session_runtime=ctx.session_runtime,
        )


__all__ = ["AgentToolContext"]
