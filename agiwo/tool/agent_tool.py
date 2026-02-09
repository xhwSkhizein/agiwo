import time
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from agiwo.agent.schema import RunOutput
from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from agiwo.agent.agent import Agent


# Default maximum nesting depth for Agent as Tool
DEFAULT_MAX_DEPTH = 5


class CircularReferenceError(Exception):
    """Raised when a circular reference is detected in Agent call chain."""

    pass


class MaxDepthExceededError(Exception):
    """Raised when the maximum nesting depth is exceeded."""

    pass


class AgentTool(BaseTool):
    """Wraps an Agent as a Tool for nested agent execution."""

    def __init__(
        self,
        agent: "Agent",
        name: str | None = None,
        description: str | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ):
        self._agent = agent
        self.name = name or agent.id
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

    def is_concurrency_safe(self) -> bool:
        """Return True if the tool can be executed concurrently."""
        return True

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        """
        Execute the wrapped Agent.

        Execution flow:
        1. Check depth limit and circular references
        2. Build input and context
        3. Execute Agent.run() directly
        4. Return final output as ToolResult

        Safety checks:
        - Raises MaxDepthExceededError if depth > max_depth
        - Raises CircularReferenceError if agent.id is in call stack
        """
        start_time = time.time()
        toolcall_id = parameters.get("tool_call_id", "")
        task = parameters.get("task", "")
        extra_context = parameters.get("context", "")

        # Get current depth and call stack from context metadata
        current_depth = context.depth + 1
        call_stack: list[str] = context.metadata.get("_call_stack", []).copy()

        # Safety check 1: Depth limit
        if current_depth > self.max_depth:
            error_msg = (
                f"Maximum nesting depth ({self.max_depth}) exceeded. "
                f"Current call chain: {' -> '.join(call_stack)} -> {self._agent.id}"
            )
            return ToolResult.error(
                tool_name=self.get_name(),
                error=error_msg,
                tool_call_id=toolcall_id,
                input_args=parameters,
                start_time=start_time,
            )

        # Safety check 2: Circular reference
        if self._agent.id in call_stack:
            error_msg = (
                f"Circular reference detected: {self._agent.id} is in call stack. "
                f"Current call chain: {' -> '.join(call_stack)}"
            )
            return ToolResult.error(
                tool_name=self.get_name(),
                error=error_msg,
                tool_call_id=toolcall_id,
                input_args=parameters,
                start_time=start_time,
            )

        # Add current agent to call stack
        call_stack.append(self._agent.id)

        # Build input and child context
        input_query = task
        if extra_context:
            input_query += f"\nAdditional context: {extra_context}"

        run_id = str(uuid4())
        child_ctx = context.new_child(run_id=run_id, agent_id=self._agent.id)
        child_ctx.metadata["_call_stack"] = call_stack

        response_text = ""
        error = None
        try:
            run_output: RunOutput = await self._agent.run(
                input_query,
                context=child_ctx,
                abort_signal=abort_signal,
            )
            response_text = run_output.response or ""
        except CircularReferenceError as e:
            error = str(e)
            response_text = f"Circular reference error: {error}"
        except MaxDepthExceededError as e:
            error = str(e)
            response_text = f"Max depth exceeded: {error}"
        except Exception as e:
            error = str(e)
            response_text = f"Error executing {self._agent.id}: {error}"

        end_time = time.time()

        return ToolResult(
            tool_name=self.get_name(),
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args={"task": task, "context": extra_context},
            content=response_text,
            output=response_text,
            error=error,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            is_success=error is None,
        )


def as_tool(
    agent: "Agent",
    description: str | None = None,
    name: str | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> AgentTool:
    """
    Convert an Agent to a Tool.

    This is a convenience factory function.

    Usage:
        research_tool = as_tool(research_agent, "Expert at research tasks")
        orchestra = Agent(model=gpt4, tools=[research_tool])

    Args:
        agent: Agent instance
        description: Tool description for LLM reference
        name: Tool name, defaults to call_{agent.id}
        max_depth: Maximum nesting depth allowed (default: 5)

    Returns:
        AgentTool instance
    """
    return AgentTool(
        agent,
        description,
        name,
        max_depth=max_depth,
    )
