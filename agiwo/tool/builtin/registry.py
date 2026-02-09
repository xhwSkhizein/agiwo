"""
Builtin tool auto-registration via decorators.

Usage:
    from agiwo.tool.builtin.registry import builtin_tool, default_enable

    @default_enable
    @builtin_tool("my_tool")
    class MyTool(BaseTool):
        ...

External packages can register tools via entry points in pyproject.toml:
    [project.entry-points."agiwo.tools"]
    my_tool = "my_package.tools:MyCustomTool"
"""

import importlib.metadata
from typing import Type

from agiwo.tool.base import BaseTool
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

BUILTIN_TOOLS: dict[str, Type[BaseTool]] = {}
DEFAULT_TOOLS: dict[str, Type[BaseTool]] = {}

_ENTRY_POINT_GROUP = "agiwo.tools"


def builtin_tool(name: str):
    """Register a BaseTool subclass as a builtin tool."""

    def decorator(cls: Type[BaseTool]) -> Type[BaseTool]:
        if name in BUILTIN_TOOLS:
            raise ValueError(f"Builtin tool name '{name}' is already registered")
        BUILTIN_TOOLS[name] = cls
        cls._builtin_name = name
        return cls

    return decorator


def default_enable(cls: Type[BaseTool]) -> Type[BaseTool]:
    """Mark a builtin tool as enabled by default when Agent tools=None."""
    name = getattr(cls, "_builtin_name", None)
    if name is None:
        raise ValueError(
            f"@default_enable must be applied AFTER @builtin_tool, "
            f"but {cls.__name__} has no _builtin_name"
        )
    DEFAULT_TOOLS[name] = cls
    return cls


def discover_entry_point_tools() -> None:
    """Scan 'agiwo.tools' entry points and register external tools."""
    eps = importlib.metadata.entry_points()
    group = eps.select(group=_ENTRY_POINT_GROUP) if hasattr(eps, "select") else eps.get(_ENTRY_POINT_GROUP, [])
    for ep in group:
        try:
            cls = ep.load()
            if not (isinstance(cls, type) and issubclass(cls, BaseTool)):
                logger.warning("entry_point_not_a_tool", name=ep.name, value=ep.value)
                continue
            if ep.name in BUILTIN_TOOLS:
                logger.warning("entry_point_name_conflict", name=ep.name)
                continue
            BUILTIN_TOOLS[ep.name] = cls
            cls._builtin_name = ep.name
            logger.info("entry_point_tool_registered", name=ep.name)
        except Exception as e:
            logger.warning("entry_point_load_failed", name=ep.name, error=str(e))
