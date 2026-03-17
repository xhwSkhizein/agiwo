"""
Agiwo - AI Agent SDK

Usage:
    from agiwo import Agent, AgentConfig, AgentOptions, AgentHooks
    from agiwo.llm import DeepseekModel
    from agiwo.tool import BaseTool, ToolResult

    agent = Agent(
        AgentConfig(
            name="my-agent",
            description="A helpful assistant",
            system_prompt="You are helpful.",
        ),
        model=DeepseekModel(id="deepseek-chat", name="deepseek-chat"),
        tools=[MyTool()],
    )

    result = await agent.run("Hello!")
"""

from agiwo.agent.agent import Agent
from agiwo.agent import (
    AgentConfig,
    AgentStreamItem,
    AgentTool,
    RunOutput,
    TerminationReason,
    as_tool,
)
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.options import AgentOptions, RunStepStorageConfig, TraceStorageConfig
from agiwo.agent.storage.base import RunStepStorage, InMemoryRunStepStorage
from agiwo.scheduler.scheduler import Scheduler
from agiwo.tool.base import BaseTool, ToolResult

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentHooks",
    "AgentOptions",
    "AgentTool",
    "BaseTool",
    "InMemoryRunStepStorage",
    "RunOutput",
    "RunStepStorage",
    "RunStepStorageConfig",
    "Scheduler",
    "AgentStreamItem",
    "TerminationReason",
    "ToolResult",
    "TraceStorageConfig",
    "as_tool",
]
