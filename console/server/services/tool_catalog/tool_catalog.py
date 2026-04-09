"""Available tool listing for the Console API."""

from agiwo.tool.builtin.registry import BUILTIN_TOOLS, ensure_builtin_tools_loaded
from agiwo.tool.reference import AgentToolReference

from server.services.agent_registry import AgentRegistry

BUILTIN_TOOL_DESCRIPTIONS: dict[str, str] = {
    "bash": "Execute shell commands in a terminal-style sandbox",
    "bash_process": "Manage background jobs started by the bash tool",
    "web_search": "Search the web for information",
    "web_reader": "Fetch and extract content from web URLs",
}


async def list_available_tools(
    registry: AgentRegistry,
    *,
    exclude_agent_id: str | None = None,
) -> list[dict[str, str]]:
    """List all available tools including builtin and agent-as-tools."""
    ensure_builtin_tools_loaded()
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
        # Use AgentToolReference for consistent reference formatting
        ref = AgentToolReference(agent_id=agent.id)
        tools.append(
            {
                "name": str(ref),
                "description": agent.description or f"Delegate tasks to {agent.name}",
                "type": "agent",
                "agent_name": agent.name,
            }
        )
    return tools
