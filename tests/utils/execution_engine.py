from agiwo.agent import AgentHooks
from agiwo.agent.engine.context import AgentRunContext
from agiwo.agent.engine.engine import ExecutionEngine
from agiwo.agent.lifecycle.definition import ResolvedExecutionDefinition
from agiwo.agent.options import AgentOptions
from agiwo.llm.base import Model


def _build_executor(
    *,
    model: Model,
    tools: list[object],
    context: AgentRunContext,
    options: AgentOptions | None = None,
) -> ExecutionEngine:
    definition = ResolvedExecutionDefinition(
        agent_id=context.agent_id,
        agent_name=context.agent_name,
        description="test-agent",
        model=model,
        hooks=AgentHooks(),
        options=options or AgentOptions(),
        tools=tuple(tools),
        system_prompt="You are a test assistant.",
    )
    return ExecutionEngine(
        definition=definition,
        step_observers=(),
        root_path="/tmp",
    )


__all__ = ["_build_executor"]
