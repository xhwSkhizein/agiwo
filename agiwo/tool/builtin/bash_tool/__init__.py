"""Public exports and pairing helpers for bash tools."""

from agiwo.tool.base import BaseTool
from agiwo.tool.builtin.bash_tool.policy_adapter import (
    BashCommandPolicyAdapter,
    BashPermissionMode,
)
from agiwo.tool.builtin.bash_tool.process_tool import (
    BashProcessTool,
    BashProcessToolConfig,
)
from agiwo.tool.builtin.bash_tool.tool import BashTool, BashToolConfig


def ensure_bash_tool_pair(tools: list[BaseTool]) -> list[BaseTool]:
    """Ensure bash execution and bash process management tools share one sandbox."""
    resolved_tools = list(tools)
    bash_tool = next(
        (tool for tool in resolved_tools if tool.get_name() == "bash"), None
    )
    bash_process_tool = next(
        (tool for tool in resolved_tools if tool.get_name() == "bash_process"),
        None,
    )

    if isinstance(bash_tool, BashTool) and bash_process_tool is None:
        resolved_tools.append(
            BashProcessTool(
                BashProcessToolConfig(
                    sandbox=bash_tool.config.sandbox,
                    max_output_length=bash_tool.config.max_output_length,
                )
            )
        )
    elif isinstance(bash_process_tool, BashProcessTool) and bash_tool is None:
        resolved_tools.append(
            BashTool(
                BashToolConfig(
                    sandbox=bash_process_tool.config.sandbox,
                    cwd=".",
                    max_output_length=bash_process_tool.config.max_output_length,
                )
            )
        )

    return resolved_tools


__all__ = [
    "BashCommandPolicyAdapter",
    "BashPermissionMode",
    "BashProcessTool",
    "BashProcessToolConfig",
    "BashTool",
    "BashToolConfig",
    "ensure_bash_tool_pair",
]
