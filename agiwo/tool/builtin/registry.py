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

import importlib
import importlib.metadata
import pkgutil
from typing import Type

from agiwo.tool.base import BaseTool
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

BUILTIN_TOOLS: dict[str, Type[BaseTool]] = {}
DEFAULT_TOOLS: dict[str, Type[BaseTool]] = {}

_ENTRY_POINT_GROUP = "agiwo.tools"
_BUILTIN_PACKAGE = "agiwo.tool.builtin"


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


def load_builtin_tools() -> None:
    """Auto-discover and import all builtin tool modules.

    Recursively searches the builtin package for modules named 'tool.py'
    and imports them to trigger @builtin_tool decorator registration.
    Each module is wrapped in try/except so optional-dependency tools
    don't block others from loading.
    """
    import agiwo.tool.builtin as builtin_pkg

    def _is_tool_module(name: str) -> bool:
        return name == "tool" or name.endswith("_tool")

    for _, name, ispkg in pkgutil.iter_modules(builtin_pkg.__path__, prefix=""):
        if ispkg:
            # Scan subpackage for tool modules (tool.py or *_tool.py)
            try:
                subpkg = importlib.import_module(f"{_BUILTIN_PACKAGE}.{name}")
            except Exception as e:
                logger.warning("builtin_subpkg_load_failed", subpkg=name, error=str(e))
                continue

            if not hasattr(subpkg, "__path__"):
                continue

            for _, subname, _ in pkgutil.iter_modules(subpkg.__path__, prefix=""):
                if _is_tool_module(subname):
                    sub_module = f"{_BUILTIN_PACKAGE}.{name}.{subname}"
                    try:
                        importlib.import_module(sub_module)
                        logger.debug("builtin_tool_loaded", module=sub_module)
                    except Exception as e:
                        logger.warning("builtin_tool_load_failed", module=sub_module, error=str(e))
        else:
            if _is_tool_module(name):
                module_path = f"{_BUILTIN_PACKAGE}.{name}"
                try:
                    importlib.import_module(module_path)
                    logger.debug("builtin_tool_loaded", module=module_path)
                except Exception as e:
                    logger.warning("builtin_tool_load_failed", module=module_path, error=str(e))


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
