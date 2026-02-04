import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from agiwo.agent.base import AgiwoAgent
from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from agiwo.agent import Agent

# Default maximum nesting depth for Agent as Tool
DEFAULT_MAX_DEPTH = 5


class CircularReferenceError(Exception):
    """Raised when a circular reference is detected in Agent call chain."""

    pass


class MaxDepthExceededError(Exception):
    """Raised when the maximum nesting depth is exceeded."""

    pass


class AgentTool(BaseTool):
    """ """

    def __init__(
        self,
        agent: AgiwoAgent,
        name: str | None = None,
        description: str | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ):
        """
        Initialize AgentTool.

        Args:
            agent: AgiwoAgent instance
            name: Tool name
            description: Tool description
            max_depth: Maximum nesting depth for Agent as Tool
        """
        self.agent = agent
        self.name = name or agent.name
        self.description = description or agent.description
        self.max_depth = max_depth

    def get_name(self) -> str:
        """Return the tool name."""
        return self.name

    def get_description(self) -> str:
        """Return the tool description."""
        return self.description

    def get_parameters(self) -> dict[str, Any]:
        """Return the JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to delegate to this agent",
                },
                "context": {
                    "type": "string",
                    "description": "Optional additional context for the task",
                },
            },
            "required": ["task"],
        }
