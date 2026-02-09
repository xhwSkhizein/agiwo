"""
Agiwo - AI Agent SDK

Usage:
    from agiwo import Agent, AgentOptions, AgentHooks
    from agiwo.llm import DeepseekModel
    from agiwo.tool import BaseTool, ToolResult

    agent = Agent(
        id="my-agent",
        description="A helpful assistant",
        model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
        tools=[MyTool()],
        system_prompt="You are helpful.",
    )

    result = await agent.run("Hello!")
"""

from agiwo.agent.agent import Agent
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.options import AgentOptions, RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.schema import RunOutput, StreamEvent, TerminationReason
from agiwo.agent.storage.base import RunStepStorage, InMemoryRunStepStorage
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.agent_tool import AgentTool, as_tool

__all__ = [
    "Agent",
    "AgentHooks",
    "AgentOptions",
    "AgentTool",
    "BaseTool",
    "RunOutput",
    "RunStepStorage",
    "RunStepStorageConfig",
    "InMemoryRunStepStorage",
    "StreamEvent",
    "TerminationReason",
    "ToolResult",
    "TraceStorageConfig",
    "as_tool",
]
