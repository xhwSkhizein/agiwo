"""
Tool registry for the console — maps tool names to instances.
"""

from typing import Any

import agiwo.tool.builtin  # noqa: F401 — triggers load_builtin_tools() registration
from agiwo.tool.base import BaseTool
from agiwo.tool.builtin.registry import BUILTIN_TOOLS


TOOL_DESCRIPTIONS: dict[str, str] = {
    "bash": "Execute shell commands in a terminal-style sandbox",
    "web_search": "Search the web for information",
    "web_reader": "Fetch and extract content from web URLs",
}

TOOLS_REQUIRING_CITATION_STORE = frozenset({"web_search", "web_reader"})


def get_available_builtin_tools() -> list[dict[str, str]]:
    """Return list of builtin tools with name, description, and type."""
    return [
        {"name": name, "description": TOOL_DESCRIPTIONS.get(name, ""), "type": "builtin"}
        for name in BUILTIN_TOOLS
    ]


def create_tools(
    tool_names: list[str],
    citation_source_store: Any = None,
) -> list[BaseTool]:
    """Instantiate tool objects from a list of tool names."""
    tools: list[BaseTool] = []
    for name in tool_names:
        cls = BUILTIN_TOOLS.get(name)
        if cls is not None:
            kwargs: dict[str, Any] = {}
            if name in TOOLS_REQUIRING_CITATION_STORE and citation_source_store is not None:
                kwargs["citation_source_store"] = citation_source_store
            tools.append(cls(**kwargs))
    return tools
