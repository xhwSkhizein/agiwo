"""Agent-level tool resolution: filtering, disabling, and default assembly.

Functions moved here from ``agiwo.agent.definition`` so that tool-assembly
logic lives closer to the rest of the tool subsystem.
"""

from agiwo.tool.base import BaseTool
from agiwo.tool.builtin import ensure_builtin_tools_loaded
from agiwo.tool.builtin.bash_tool import ensure_bash_tool_pair
from agiwo.tool.builtin.registry import DEFAULT_TOOLS

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agiwo.skill.manager import SkillManager


def normalize_disabled_sdk_tool_names(names: set[str] | None) -> set[str]:
    normalized = set(names or set())
    if "bash" in normalized:
        normalized.add("bash_process")
    if "bash_process" in normalized:
        normalized.add("bash")
    return normalized


def exact_tool_disable_set() -> set[str]:
    return {"skill", *DEFAULT_TOOLS.keys()}


def build_agent_tools(
    *,
    tools: list[BaseTool] | None,
    disabled_sdk_tool_names: set[str] | None,
    skill_manager: "SkillManager | None",
) -> tuple[BaseTool, ...]:
    ensure_builtin_tools_loaded()
    disabled_names = normalize_disabled_sdk_tool_names(disabled_sdk_tool_names)
    provided_tools = list(tools or [])
    base_tool_names = {tool.name for tool in provided_tools}
    default_tools: list[BaseTool] = []
    for name, tool_cls in DEFAULT_TOOLS.items():
        if name in disabled_names or name in base_tool_names:
            continue
        default_tools.append(tool_cls())

    resolved_base_tools = ensure_bash_tool_pair([*provided_tools, *default_tools])
    if skill_manager is not None and "skill" not in disabled_names:
        if all(tool.name != "skill" for tool in resolved_base_tools):
            resolved_base_tools.append(skill_manager.get_skill_tool())
    return tuple(resolved_base_tools)


__all__ = [
    "build_agent_tools",
    "exact_tool_disable_set",
    "normalize_disabled_sdk_tool_names",
]
