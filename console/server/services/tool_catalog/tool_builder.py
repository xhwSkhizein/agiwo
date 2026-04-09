"""Runtime tool assembly for console agents."""

from collections.abc import Awaitable, Callable

from agiwo.tool.base import BaseTool
from agiwo.tool.builtin.registry import BUILTIN_TOOLS
from agiwo.tool.reference import (
    AgentToolReference,
    InvalidToolReferenceError,
    parse_tool_reference,
)
from agiwo.tool.storage.citation import CitationStoreConfig

from server.config import ConsoleConfig
from server.services.storage_wiring import build_citation_store_config

_CITATION_TOOLS: set[str] = {"web_search", "web_reader"}


async def build_tools(
    tool_refs: list[str],
    *,
    console_config: ConsoleConfig,
    build_agent_tool: Callable[[str], Awaitable[BaseTool | None]],
) -> list[BaseTool]:
    """Build tool instances from reference strings.

    Args:
        tool_refs: List of tool reference strings (e.g., "bash", "agent:id")
        console_config: Console configuration
        build_agent_tool: Async callable to build an agent-as-tool from reference

    Returns:
        List of built tool instances
    """
    citation_store_config = build_citation_store_config(console_config)
    tools: list[BaseTool] = []

    for ref_str in tool_refs:
        try:
            ref = parse_tool_reference(ref_str)
        except InvalidToolReferenceError:
            # Skip invalid references
            continue

        if isinstance(ref, AgentToolReference):
            agent_tool = await build_agent_tool(str(ref))
            if agent_tool is not None:
                tools.append(agent_tool)
        else:
            # BuiltinToolReference
            tools.append(_build_builtin_tool(ref.name, citation_store_config))

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
