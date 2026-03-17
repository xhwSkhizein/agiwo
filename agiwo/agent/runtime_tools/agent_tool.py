import time
from typing import TYPE_CHECKING, Any

from agiwo.agent.inner.context import AgentRunContext
from agiwo.agent.runtime import RunOutput
from agiwo.agent.runtime_tools.contracts import RuntimeToolOutcome
from agiwo.tool.base import ToolDefinition, ToolResult
from agiwo.utils.abort_signal import AbortSignal

if TYPE_CHECKING:
    from agiwo.agent.agent import Agent


DEFAULT_MAX_DEPTH = 5


class CircularReferenceError(Exception):
    """Raised when a circular reference is detected in Agent call chain."""

    pass


class MaxDepthExceededError(Exception):
    """Raised when the maximum nesting depth is exceeded."""

    pass


class AgentTool:
    """Wrap a child agent template as a tool for nested execution."""

    cacheable = False
    timeout_seconds = 30

    def __init__(
        self,
        agent: "Agent",
        *,
        name: str | None = None,
        description: str | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        self._agent = agent
        self._name = name or agent.name
        self._description = description or agent.description
        self.max_depth = max_depth

    def get_name(self) -> str:
        return self._name

    def get_description(self) -> str:
        return self._description

    def get_parameters(self) -> dict[str, Any]:
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
        return True

    def get_short_description(self) -> str:
        return self.get_description()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.get_name(),
            description=self.get_description(),
            parameters=self.get_parameters(),
            is_concurrency_safe=self.is_concurrency_safe(),
            timeout_seconds=self.timeout_seconds,
            cacheable=self.cacheable,
        )

    async def execute(
        self,
        parameters: dict[str, Any],
        context: AgentRunContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        outcome = await self.execute_for_agent(parameters, context, abort_signal)
        return outcome.result

    async def execute_for_agent(
        self,
        parameters: dict[str, Any],
        context: AgentRunContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome:
        start_time = time.time()
        toolcall_id = parameters.get("tool_call_id", "")
        task = parameters.get("task", "")
        extra_context = parameters.get("context", "")
        child_spec = self._agent.derive_child_spec(child_id=self._agent.id)
        current_depth = context.depth + 1
        call_stack: list[str] = context.metadata.get("_call_stack", []).copy()

        if current_depth > self.max_depth:
            error_msg = (
                f"Maximum nesting depth ({self.max_depth}) exceeded. "
                f"Current call chain: {' -> '.join(call_stack)} -> {child_spec.agent_id}"
            )
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_name=self.get_name(),
                    error=error_msg,
                    tool_call_id=toolcall_id,
                    input_args=parameters,
                    start_time=start_time,
                )
            )

        if child_spec.agent_id in call_stack:
            error_msg = (
                f"Circular reference detected: {child_spec.agent_id} is in call stack. "
                f"Current call chain: {' -> '.join(call_stack)}"
            )
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_name=self.get_name(),
                    error=error_msg,
                    tool_call_id=toolcall_id,
                    input_args=parameters,
                    start_time=start_time,
                )
            )

        call_stack.append(child_spec.agent_id)
        input_query = task
        if extra_context:
            input_query += f"\nAdditional context: {extra_context}"

        response_text = ""
        error = None
        try:
            run_output: RunOutput = await self._agent.run_child(
                input_query,
                spec=child_spec,
                parent_context=context,
                metadata_updates={"_call_stack": call_stack},
                abort_signal=abort_signal,
            )
            response_text = run_output.response or ""
        except CircularReferenceError as exc:
            error = str(exc)
            response_text = f"Circular reference error: {error}"
        except MaxDepthExceededError as exc:
            error = str(exc)
            response_text = f"Max depth exceeded: {error}"
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            response_text = f"Error executing {child_spec.agent_name}: {error}"

        if error is not None:
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_name=self.get_name(),
                    error=error,
                    tool_call_id=str(parameters.get("tool_call_id", "")),
                    input_args={"task": task, "context": extra_context},
                    start_time=start_time,
                    content=response_text,
                    output=response_text,
                )
            )

        return RuntimeToolOutcome(
            result=ToolResult.success(
                tool_name=self.get_name(),
                tool_call_id=str(parameters.get("tool_call_id", "")),
                input_args={"task": task, "context": extra_context},
                content=response_text,
                output=response_text,
                start_time=start_time,
            )
        )


def as_tool(
    agent: "Agent",
    name: str | None = None,
    description: str | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> AgentTool:
    return AgentTool(
        agent,
        name=name,
        description=description,
        max_depth=max_depth,
    )


__all__ = [
    "AgentTool",
    "CircularReferenceError",
    "DEFAULT_MAX_DEPTH",
    "MaxDepthExceededError",
    "as_tool",
]
