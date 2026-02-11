"""
Tool registry for the console — maps tool names to instances.
"""

from agiwo.tool.base import BaseTool
from agiwo.tool.builtin import BUILTIN_TOOLS


TOOL_DESCRIPTIONS: dict[str, str] = {
    "current_time": "Get current date, time, and timezone",
    "calculator": "Evaluate mathematical expressions safely",
    "http_request": "Make HTTP GET/POST requests to URLs",
}


def get_available_builtin_tools() -> list[dict[str, str]]:
    """Return list of builtin tools with name, description, and type."""
    return [
        {"name": name, "description": TOOL_DESCRIPTIONS.get(name, ""), "type": "builtin"}
        for name in BUILTIN_TOOLS
    ]


def create_tools(tool_names: list[str]) -> list[BaseTool]:
    """Instantiate tool objects from a list of tool names."""
    tools: list[BaseTool] = []
    for name in tool_names:
        cls = BUILTIN_TOOLS.get(name)
        if cls is not None:
            tools.append(cls())
    return tools
