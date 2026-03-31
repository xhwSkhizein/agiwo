"""Console tool catalog: canonical refs, metadata, and tool assembly."""

from collections.abc import Awaitable, Callable

from agiwo.tool.base import BaseTool
from agiwo.tool.builtin.registry import BUILTIN_TOOLS
from agiwo.tool.storage.citation import CitationStoreConfig

from server.config import ConsoleConfig
from server.schemas import AGENT_TOOL_PREFIX, parse_tool_references
from server.services.agent_registry import AgentRegistry
from server.services.storage_wiring import build_citation_store_config

BUILTIN_TOOL_DESCRIPTIONS: dict[str, str] = {
    "bash": "Execute shell commands in a terminal-style sandbox",
    "web_search": "Search the web for information",
    "web_reader": "Fetch and extract content from web URLs",
}

_CITATION_TOOLS: set[str] = {"web_search", "web_reader"}


async def list_available_tools(
    registry: AgentRegistry,
    *,
    exclude_agent_id: str | None = None,
) -> list[dict[str, str]]:
    tools: list[dict[str, str]] = [
        {
            "name": name,
            "description": BUILTIN_TOOL_DESCRIPTIONS.get(name, ""),
            "type": "builtin",
        }
        for name in BUILTIN_TOOLS
    ]
    for agent in await registry.list_agents():
        if exclude_agent_id is not None and agent.id == exclude_agent_id:
            continue
        tools.append(
            {
                "name": f"{AGENT_TOOL_PREFIX}{agent.id}",
                "description": agent.description or f"Delegate tasks to {agent.name}",
                "type": "agent",
                "agent_name": agent.name,
            }
        )
    return tools


async def build_tools(
    tool_refs: list[str],
    *,
    console_config: ConsoleConfig,
    build_agent_tool: Callable[[str], Awaitable[BaseTool | None]],
) -> list[BaseTool]:
    citation_store_config = build_citation_store_config(console_config)
    tools: list[BaseTool] = []
    for ref in parse_tool_references(list(tool_refs)):
        if ref.startswith(AGENT_TOOL_PREFIX):
            agent_tool = await build_agent_tool(ref)
            if agent_tool is not None:
                tools.append(agent_tool)
        else:
            tools.append(_build_builtin_tool(ref, citation_store_config))
    return tools


def _build_builtin_tool(
    name: str,
    citation_store_config: CitationStoreConfig,
) -> BaseTool:
    tool_cls = BUILTIN_TOOLS[name]
    kwargs: dict[str, object] = {}
    if name in _CITATION_TOOLS:
        kwargs["citation_store_config"] = citation_store_config
    return tool_cls(**kwargs)
