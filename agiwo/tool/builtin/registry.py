"""
Builtin tool auto-registration via decorators.

Usage:
    from agiwo.tool.builtin.registry import builtin_tool, default_enable

    @default_enable
    @builtin_tool("my_tool")
    class MyTool(BaseTool):
        ...
"""

import importlib
import pkgutil
from typing import Type

from agiwo.tool.base import BaseTool
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

BUILTIN_TOOLS: dict[str, Type[BaseTool]] = {}
DEFAULT_TOOLS: dict[str, Type[BaseTool]] = {}
_BUILTIN_TOOLS_LOADED = False

_BUILTIN_PACKAGE = "agiwo.tool.builtin"


def _is_tool_module(name: str) -> bool:
    return name == "tool" or name.endswith("_tool")


def _safe_import_module(
    module_path: str, *, warning_event: str, warning_key: str
) -> None:
    try:
        importlib.import_module(module_path)
        logger.info("builtin_tool_loaded", module=module_path)
    except Exception as error:  # noqa: BLE001 - optional builtin dependency boundary
        logger.warning(warning_event, **{warning_key: module_path, "error": str(error)})


def _load_subpackage_tools(name: str) -> None:
    module_path = f"{_BUILTIN_PACKAGE}.{name}"
    try:
        subpkg = importlib.import_module(module_path)
    except Exception as error:  # noqa: BLE001 - optional builtin dependency boundary
        logger.warning("builtin_subpkg_load_failed", subpkg=name, error=str(error))
        return

    if not hasattr(subpkg, "__path__"):
        return

    for _, subname, _ in pkgutil.iter_modules(subpkg.__path__, prefix=""):
        if _is_tool_module(subname):
            _safe_import_module(
                f"{module_path}.{subname}",
                warning_event="builtin_tool_load_failed",
                warning_key="module",
            )


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
    builtin_pkg = importlib.import_module(_BUILTIN_PACKAGE)

    for _, name, ispkg in pkgutil.iter_modules(builtin_pkg.__path__, prefix=""):
        if ispkg:
            _load_subpackage_tools(name)
        elif _is_tool_module(name):
            _safe_import_module(
                f"{_BUILTIN_PACKAGE}.{name}",
                warning_event="builtin_tool_load_failed",
                warning_key="module",
            )


def ensure_builtin_tools_loaded() -> None:
    """Load builtin tool modules at most once per process."""
    global _BUILTIN_TOOLS_LOADED
    if _BUILTIN_TOOLS_LOADED:
        return
    load_builtin_tools()
    _BUILTIN_TOOLS_LOADED = True
