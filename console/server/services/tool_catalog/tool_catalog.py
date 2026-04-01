"""Available tool listing for the Console API."""

from agiwo.tool.builtin.registry import BUILTIN_TOOLS, ensure_builtin_tools_loaded
from agiwo.utils.logging import get_logger

from server.services.agent_registry import AgentRegistry
from server.services.tool_catalog.tool_references import AGENT_TOOL_PREFIX

logger = get_logger(__name__)

BUILTIN_TOOL_DESCRIPTIONS: dict[str, str] = {
    "bash": "Execute shell commands in a terminal-style sandbox",
    "bash_process": "Inspect and control background shell processes",
    "memory_retrieval": "Search shared workspace memory for relevant context",
    "web_search": "Search the web for information",
    "web_reader": "Fetch and extract content from web URLs",
}


async def list_available_tools(
    registry: AgentRegistry,
    *,
    exclude_agent_id: str | None = None,
) -> list[dict[str, str]]:
    ensure_builtin_tools_loaded()
    tools: list[dict[str, str]] = []
    for name in BUILTIN_TOOLS:
        description = BUILTIN_TOOL_DESCRIPTIONS.get(name)
        if description is None:
            logger.warning("builtin_tool_description_missing", tool_name=name)
            description = ""
        tools.append(
            {
                "name": name,
                "description": description,
                "type": "builtin",
            }
        )
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
