"""Console tool catalog: canonical refs, metadata, and tool assembly."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from agiwo.agent import Agent
from agiwo.tool.base import BaseTool
from agiwo.tool.builtin.registry import BUILTIN_TOOLS
from agiwo.tool.storage.citation import CitationStoreConfig

from server.config import ConsoleConfig
from server.domain.tool_references import (
    AGENT_TOOL_PREFIX,
    AgentToolRef,
    BuiltinToolRef,
    ToolReference,
    parse_tool_references,
)
from server.services.agent_registry import AgentRegistry
from server.services.storage_wiring import build_citation_store_config


@dataclass(frozen=True)
class ConsoleToolDescriptor:
    ref: ToolReference
    name: str
    description: str
    type: Literal["builtin", "agent"]
    agent_name: str | None = None

    def to_payload(self) -> dict[str, str]:
        payload = {
            "name": self.name,
            "description": self.description,
            "type": self.type,
        }
        if self.agent_name is not None:
            payload["agent_name"] = self.agent_name
        return payload


@dataclass(frozen=True)
class BuiltinToolSpec:
    description: str
    needs_citation_store: bool = False


BUILTIN_TOOL_SPECS: dict[str, BuiltinToolSpec] = {
    "bash": BuiltinToolSpec(
        description="Execute shell commands in a terminal-style sandbox",
    ),
    "web_search": BuiltinToolSpec(
        description="Search the web for information",
        needs_citation_store=True,
    ),
    "web_reader": BuiltinToolSpec(
        description="Fetch and extract content from web URLs",
        needs_citation_store=True,
    ),
}


async def list_available_tools(
    registry: AgentRegistry,
    *,
    exclude_agent_id: str | None = None,
) -> list[dict[str, str]]:
    tools = [descriptor.to_payload() for descriptor in list_builtin_tools()]
    agents = await registry.list_agents()
    for agent in agents:
        if exclude_agent_id is not None and agent.id == exclude_agent_id:
            continue
        tools.append(
            ConsoleToolDescriptor(
                ref=AgentToolRef(agent_id=agent.id),
                name=f"{AGENT_TOOL_PREFIX}{agent.id}",
                description=agent.description or f"Delegate tasks to {agent.name}",
                type="agent",
                agent_name=agent.name,
            ).to_payload()
        )
    return tools


def list_builtin_tools() -> list[ConsoleToolDescriptor]:
    return [
        ConsoleToolDescriptor(
            ref=BuiltinToolRef(name=name),
            name=name,
            description=_get_builtin_description(name),
            type="builtin",
        )
        for name in BUILTIN_TOOLS
    ]


async def build_tools(
    tool_refs: list[str] | list[ToolReference],
    *,
    console_config: ConsoleConfig,
    build_agent_tool: Callable[[AgentToolRef], Awaitable[BaseTool | None]],
) -> list[BaseTool]:
    citation_store_config = build_citation_store_config(console_config)
    tools: list[BaseTool] = []
    for ref in parse_tool_references(list(tool_refs)):
        if isinstance(ref, BuiltinToolRef):
            tools.append(_build_builtin_tool(ref, citation_store_config))
            continue
        agent_tool = await build_agent_tool(ref)
        if agent_tool is not None:
            tools.append(agent_tool)
    return tools


def build_agent_tool(_ref: AgentToolRef, agent: Agent) -> BaseTool:
    return agent.as_tool()


def _build_builtin_tool(
    ref: BuiltinToolRef,
    citation_store_config: CitationStoreConfig,
) -> BaseTool:
    tool_cls = BUILTIN_TOOLS[ref.name]
    kwargs: dict[str, object] = {}
    spec = BUILTIN_TOOL_SPECS.get(ref.name)
    if spec is not None and spec.needs_citation_store:
        kwargs["citation_store_config"] = citation_store_config
    return tool_cls(**kwargs)


def _get_builtin_description(name: str) -> str:
    spec = BUILTIN_TOOL_SPECS.get(name)
    if spec is not None:
        return spec.description
    return ""


__all__ = [
    "AGENT_TOOL_PREFIX",
    "AgentToolRef",
    "BuiltinToolRef",
    "ConsoleToolDescriptor",
    "ToolReference",
    "build_agent_tool",
    "build_tools",
    "list_available_tools",
    "list_builtin_tools",
]
