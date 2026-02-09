"""
Builtin tools package — auto-discovers all tool modules in this directory
and external tools registered via entry points.

To add a new builtin tool:
1. Create a new .py file in this directory
2. Decorate the tool class with @builtin_tool("tool_name")
3. Optionally add @default_enable to include it when Agent tools=None

External packages register tools via pyproject.toml:
    [project.entry-points."agiwo.tools"]
    my_tool = "my_package.tools:MyCustomTool"
"""

import importlib
import pkgutil
from pathlib import Path

from agiwo.tool.builtin.registry import BUILTIN_TOOLS, DEFAULT_TOOLS, discover_entry_point_tools

_pkg_dir = str(Path(__file__).parent)
for _info in pkgutil.iter_modules([_pkg_dir]):
    if _info.name != "registry":
        importlib.import_module(f"{__name__}.{_info.name}")

discover_entry_point_tools()

__all__ = ["BUILTIN_TOOLS", "DEFAULT_TOOLS"]
