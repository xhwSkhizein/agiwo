from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agiwo.agent.runtime import TerminationReason
from agiwo.tool.base import ToolDefinition, ToolGateDecision, ToolResult
from agiwo.utils.abort_signal import AbortSignal


@dataclass(frozen=True)
class RuntimeToolOutcome:
    """Runtime-level outcome for tool execution inside ExecutionEngine."""

    result: ToolResult
    termination_reason: TerminationReason | None = None


@runtime_checkable
class AgentRuntimeContext(Protocol):
    session_id: str
    agent_id: str
    agent_name: str
    user_id: str | None
    depth: int
    timeout_at: float | None
    metadata: dict[str, Any]


@runtime_checkable
class AgentRuntimeTool(Protocol):
    """Tool contract understood by the agent runtime."""

    cacheable: bool
    timeout_seconds: int

    def get_name(self) -> str:
        """Return the tool name."""

    def get_definition(self) -> ToolDefinition:
        """Return the LLM-facing tool definition."""

    def get_short_description(self) -> str:
        """Return the prompt-friendly short description."""

    def is_concurrency_safe(self) -> bool:
        """Whether the tool can be executed concurrently."""

    async def gate_for_agent(
        self,
        parameters: dict[str, Any],
        context: AgentRuntimeContext,
    ) -> ToolGateDecision:
        """Run preflight permission/confirmation checks inside agent runtime."""

    async def execute_for_agent(
        self,
        parameters: dict[str, Any],
        context: AgentRuntimeContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome:
        """Execute tool inside an agent runtime."""
