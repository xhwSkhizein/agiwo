from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any, Literal

from agiwo.tool.context import ToolContext
from agiwo.utils.abort_signal import AbortSignal


@dataclass
class ToolDefinition:
    """Tool definition for LLM-facing registration."""

    name: str
    description: str
    parameters: dict[str, Any]
    is_concurrency_safe: bool = True
    timeout_seconds: int = 30
    # Whether tool results can be cached within a session
    cacheable: bool = False


@dataclass(frozen=True)
class ToolGateDecision:
    """Minimal preflight decision returned before tool execution."""

    action: Literal["allow", "deny"]
    reason: str = ""

    @classmethod
    def allow(cls, reason: str = "") -> "ToolGateDecision":
        return cls(action="allow", reason=reason)

    @classmethod
    def deny(cls, reason: str) -> "ToolGateDecision":
        return cls(action="deny", reason=reason)


@dataclass
class ToolResult:
    """Result of a tool execution"""

    tool_name: str
    tool_call_id: str
    input_args: dict[str, Any]
    content: str  # Result for LLM (used when building messages)
    output: Any  # Raw execution result
    start_time: float
    end_time: float
    duration: float
    content_for_user: str | None = (
        None  # Display content for frontend (preferred for UI)
    )
    error: str | None = None
    is_success: bool = True
    termination_reason: Any = None

    @classmethod
    def success(
        cls,
        tool_name: str,
        content: str,
        tool_call_id: str = "",
        input_args: dict[str, Any] | None = None,
        start_time: float | None = None,
        output: Any = None,
        content_for_user: str | None = None,
        termination_reason: Any = None,
    ) -> "ToolResult":
        """Create a ToolResult representing a successful operation."""
        now = time.time()
        start = start_time if start_time is not None else now
        return cls(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            input_args=input_args or {},
            content=content,
            output=output,
            start_time=start,
            end_time=now,
            duration=now - start,
            content_for_user=content_for_user,
            is_success=True,
            termination_reason=termination_reason,
        )

    @classmethod
    def failed(
        cls,
        tool_name: str,
        error: str,
        tool_call_id: str = "",
        input_args: dict[str, Any] | None = None,
        start_time: float | None = None,
        content: str | None = None,
        output: Any = None,
    ) -> "ToolResult":
        """Create a ToolResult representing an error."""
        now = time.time()
        start = start_time if start_time is not None else now
        return cls(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            input_args=input_args or {},
            content=content or f"Error: {error}",
            output=output,
            error=error,
            start_time=start,
            end_time=now,
            duration=now - start,
            is_success=False,
        )

    @classmethod
    def aborted(
        cls,
        tool_name: str,
        tool_call_id: str = "",
        input_args: dict[str, Any] | None = None,
        start_time: float | None = None,
    ) -> "ToolResult":
        """Create a ToolResult representing an aborted operation."""
        now = time.time()
        start = start_time if start_time is not None else now
        return cls(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            input_args=input_args or {},
            content="Operation was aborted",
            output=None,
            error="Aborted",
            start_time=start,
            end_time=now,
            duration=now - start,
            is_success=False,
        )

    @classmethod
    def denied(
        cls,
        tool_name: str,
        reason: str,
        tool_call_id: str = "",
        input_args: dict[str, Any] | None = None,
        start_time: float | None = None,
        content: str | None = None,
        output: Any = None,
    ) -> "ToolResult":
        """Create a ToolResult representing a denied operation."""
        error = f"Permission denied: {reason}"
        return cls.failed(
            tool_name=tool_name,
            error=error,
            tool_call_id=tool_call_id,
            input_args=input_args,
            start_time=start_time,
            content=content
            or (
                f"Tool execution denied: {reason}. "
                "Please ask the user for permission or use a different approach."
            ),
            output=output,
        )


class BaseTool(ABC):
    """Common interface that every concrete tool must implement."""

    # Override to enable caching for expensive operations
    # Cached results are reused within the same session for identical arguments
    cacheable: bool = False
    timeout_seconds: int = 30

    def __init__(self) -> None:
        self.name = self.get_name()
        self.description = self.get_description()

    @abstractmethod
    def get_name(self) -> str:
        """Return the tool name."""

    @abstractmethod
    def get_description(self) -> str:
        """Return the tool description used for prompting."""

    def get_short_description(self) -> str:
        return self.get_description()

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return the JSON schema describing `execute` parameters."""

    @abstractmethod
    def is_concurrency_safe(self) -> bool:
        """Whether the tool can be executed concurrently."""

    async def gate(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
    ) -> ToolGateDecision:
        """Preflight hook for safety checks before execution."""
        del parameters, context
        return ToolGateDecision.allow()

    @abstractmethod
    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        """
        Execute the tool and return ToolResult directly.

        Args:
            parameters: Tool parameters (from LLM, user-defined)
            context: Execution context (required, provided by runtime)
            abort_signal: Optional abort signal for cancellation

        Returns:
            ToolResult: Tool execution result
        """

    def get_definition(self) -> ToolDefinition:
        """Construct a `ToolDefinition` for LLM-facing registration."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.get_parameters(),
            is_concurrency_safe=self.is_concurrency_safe(),
            timeout_seconds=self.timeout_seconds or 30,
            cacheable=self.cacheable,
        )

    def to_openai_schema(self) -> dict[str, object]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters(),
            },
        }
