"""Child agent tool wrapper and nested execution."""

import time
from typing import TYPE_CHECKING, Any

from agiwo.agent.runtime.context import RunContext
from agiwo.agent.nested.context import AgentToolContext
from agiwo.tool.base import BaseTool, ToolResult, resolve_timeout
from agiwo.tool.context import ToolContext
from agiwo.utils.abort_signal import AbortSignal

if TYPE_CHECKING:
    from agiwo.agent.agent import Agent


DEFAULT_MAX_DEPTH = 5


class AgentTool(BaseTool):
    """Wrap a child agent template as a tool for nested execution."""

    cacheable = False
    timeout_seconds = 600

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
        super().__init__()

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def is_inheritable_by(self, parent_agent_id: str) -> bool:
        """Check if this AgentTool can be inherited by a parent agent.

        Returns False if this tool is bound to the parent agent itself
        (circular reference risk). Returns True for all other cases.
        """
        return self._agent.id != parent_agent_id

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

    def build_context(
        self, run_context: "RunContext", *, tool_call_id: str = ""
    ) -> ToolContext:
        timeout_at = resolve_timeout(run_context.timeout_at, self.timeout_seconds)
        return AgentToolContext.from_run_context(
            run_context,
            timeout_at=timeout_at,
            gate_checked=True,
            tool_call_id=tool_call_id,
        )

    async def execute(
        self,
        parameters: dict[str, Any],
        context: AgentToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        start_time = time.time()
        toolcall_id = context.tool_call_id
        task = parameters.get("task", "")
        extra_context = parameters.get("context", "")
        child_id = self._agent.id
        current_depth = context.depth + 1
        call_stack: list[str] = context.metadata.get("_call_stack", []).copy()

        if current_depth > self.max_depth:
            error_msg = (
                f"Maximum nesting depth ({self.max_depth}) exceeded. "
                f"Current call chain: {' -> '.join(call_stack)} -> {child_id}"
            )
            return ToolResult.failed(
                tool_name=self.name,
                error=error_msg,
                tool_call_id=toolcall_id,
                input_args=parameters,
                start_time=start_time,
            )

        if child_id in call_stack:
            error_msg = (
                f"Circular reference detected: {child_id} is in call stack. "
                f"Current call chain: {' -> '.join(call_stack)}"
            )
            return ToolResult.failed(
                tool_name=self.name,
                error=error_msg,
                tool_call_id=toolcall_id,
                input_args=parameters,
                start_time=start_time,
            )

        call_stack.append(child_id)
        input_query = task
        if extra_context:
            input_query += f"\nAdditional context: {extra_context}"

        response_text = ""
        error = None
        termination_reason = None
        try:
            run_output = await self._agent.run_child(
                input_query,
                session_runtime=context.session_runtime,
                parent_run_id=context.parent_run_id,
                parent_depth=context.depth,
                parent_user_id=context.user_id,
                parent_timeout_at=context.timeout_at,
                parent_metadata=context.metadata,
                metadata_updates={"_call_stack": call_stack},
                abort_signal=abort_signal,
            )
            response_text = run_output.response or ""
            termination_reason = run_output.termination_reason
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            response_text = f"Error executing {self._agent.name}: {error}"

        input_args = {"task": task, "context": extra_context}
        if error is not None:
            return ToolResult.failed(
                tool_name=self.name,
                error=error,
                tool_call_id=context.tool_call_id,
                input_args=input_args,
                start_time=start_time,
                content=response_text,
                output=response_text,
            )

        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=input_args,
            content=response_text,
            output=response_text,
            start_time=start_time,
            termination_reason=termination_reason,
        )


__all__ = ["AgentTool", "DEFAULT_MAX_DEPTH"]
