from agiwo.agent.inner.context import AgentRunContext
from agiwo.agent.runtime_tools.contracts import AgentRuntimeTool, RuntimeToolOutcome
from agiwo.tool.base import BaseTool, ToolDefinition, ToolGateDecision
from agiwo.tool.context import ToolContext
from agiwo.utils.abort_signal import AbortSignal


def _build_tool_context(context: AgentRunContext) -> ToolContext:
    return ToolContext(
        session_id=context.session_id,
        agent_id=context.agent_id,
        agent_name=context.agent_name,
        user_id=context.user_id,
        timeout_at=context.timeout_at,
        metadata=dict(context.metadata),
    )


class BaseToolAdapter:
    """Adapt a plain BaseTool to the AgentRuntimeTool contract."""

    def __init__(self, tool: BaseTool) -> None:
        self._tool = tool
        self.cacheable = tool.cacheable
        self.timeout_seconds = tool.timeout_seconds

    def get_name(self) -> str:
        return self._tool.get_name()

    def get_definition(self) -> ToolDefinition:
        return self._tool.get_definition()

    def get_short_description(self) -> str:
        return self._tool.get_short_description()

    def is_concurrency_safe(self) -> bool:
        return self._tool.is_concurrency_safe()

    async def gate_for_agent(
        self,
        parameters: dict[str, object],
        context: AgentRunContext,
    ) -> ToolGateDecision:
        return await self._tool.gate(parameters, context=_build_tool_context(context))

    async def execute_for_agent(
        self,
        parameters: dict[str, object],
        context: AgentRunContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome:
        result = await self._tool.execute(
            parameters,
            context=_build_tool_context(context),
            abort_signal=abort_signal,
        )
        return RuntimeToolOutcome(result=result)


RuntimeToolLike = BaseTool | AgentRuntimeTool


def adapt_runtime_tool(tool: RuntimeToolLike) -> AgentRuntimeTool:
    if isinstance(tool, BaseTool):
        return BaseToolAdapter(tool)
    return tool


__all__ = [
    "BaseToolAdapter",
    "RuntimeToolLike",
    "adapt_runtime_tool",
]
