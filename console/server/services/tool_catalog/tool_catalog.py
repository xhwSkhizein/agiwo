"""Available tool listing for the Console API."""

from agiwo.tool.builtin.registry import BUILTIN_TOOLS

from server.services.agent_registry import AgentRegistry
from server.services.tool_catalog.tool_references import AGENT_TOOL_PREFIX

BUILTIN_TOOL_DESCRIPTIONS: dict[str, str] = {
    "bash": "Execute shell commands in a terminal-style sandbox",
    "web_search": "Search the web for information",
    "web_reader": "Fetch and extract content from web URLs",
}


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
