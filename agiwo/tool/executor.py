import asyncio
from dataclasses import dataclass, replace
from typing import Any
import time

from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.cache import ToolResultCache
from agiwo.utils.abort_signal import AbortSignal
from agiwo.llm.helper import parse_json_tool_args

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolExecutionOptions:
    timeout_seconds: int = 30
    cache_max_items: int = 1000
    cache_ttl_seconds: int = 3600


class ToolExecutor:
    """
    ToolExecutor is responsible for executing tools and returning ToolResult.

    FIXME:
    1. cache is not implemented yet
    2. more options support

    """

    def __init__(
        self,
        tools: list[BaseTool],
        cache: ToolResultCache | None = None,
        options: ToolExecutionOptions | None = None,
    ):
        self.tools = tools
        self.tools_map = {t.get_name(): t for t in tools}
        self.cache = cache
        self.options = options
        if options is None:
            self.options = ToolExecutionOptions()

    async def execute_batch(
        self,
        tool_calls: list[dict[str, Any]],
        context: "ExecutionContext",
        abort_signal: "AbortSignal | None" = None,
    ) -> list[ToolResult]:
        """
        Execute multiple tool calls in parallel.

        Args:
            tool_calls: List of tool calls
            context: Execution context
            abort_signal: Abort signal

        Returns:
            list[ToolResult]: List of tool execution results
        """

        async def _run_single(tc: dict[str, Any]) -> ToolResult:
            try:
                return await self.aexecute(
                    tc, context=context, abort_signal=abort_signal
                )
            except asyncio.CancelledError:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                tool_name = fn.get("name", "unknown")
                call_id = tc.get("id", "") if isinstance(tc, dict) else ""
                return ToolResult.error(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    error="Tool execution was cancelled",
                    start_time=time.time(),
                )
            except Exception as e:  # Defensive: should not propagate
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                tool_name = fn.get("name", "unknown")
                call_id = tc.get("id", "") if isinstance(tc, dict) else ""
                logger.error(
                    "tool_batch_execute_error",
                    tool_name=tool_name,
                    tool_call_id=call_id,
                    error=str(e),
                    exc_info=True,
                )
                return ToolResult.error(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    error=f"Tool execution failed: {e}",
                    start_time=time.time(),
                )

        return await asyncio.gather(*(_run_single(tc) for tc in tool_calls))

    async def aexecute(
        self,
        tool_call: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        call_id = tool_call.get("id")
        fn_name = tool_call.get("function", {}).get("name")
        fn_args = tool_call.get("function", {}).get("arguments", {})

        start_time = time.time()
        if not fn_name:
            return ToolResult.error(
                tool_call_id=call_id,
                tool_name="unknown",
                error="Tool name missing in tool call",
                start_time=start_time,
            )

        tool: BaseTool | None = self.tools_map.get(fn_name)
        if not tool:
            return ToolResult.error(
                tool_call_id=call_id,
                tool_name=fn_name,
                error=f"Tool {fn_name} not found",
                start_time=start_time,
            )

        # Parse arguments
        try:
            args = parse_json_tool_args(fn_args)
        except ValueError as e:
            return ToolResult.error(
                tool_call_id=call_id,
                tool_name=fn_name,
                error=str(e),
                start_time=start_time,
            )

        args["tool_call_id"] = call_id

        # Set timeout
        timeout_seconds = (
            tool.timeout_seconds
            if tool and tool.timeout_seconds
            else self.options.timeout_seconds
        )
        execution_context = context
        if timeout_seconds:
            timeout_at = time.time() + timeout_seconds
            execution_context = replace(context, timeout_at=timeout_at)

        # Check cache
        if self.cache and tool.cacheable:
            cached_result = self.cache.get(context.session_id, fn_name, args)
            if cached_result:
                logger.info(
                    "tool_cache_hit",
                    tool_name=fn_name,
                    session_id=context.session_id,
                )
                # Return cached result with updated tool_call_id
                # We update timestamps to current time to reflect this specific "execution" (cache retrieval)
                now = time.time()
                return replace(
                    cached_result,
                    tool_call_id=call_id,
                    start_time=now,
                    end_time=now,
                    duration=0.0,
                )

        # Execute tool
        try:
            logger.debug("executing_tool", tool_name=fn_name, tool_call_id=call_id)
            result: ToolResult = await tool.execute(
                args, context=execution_context, abort_signal=abort_signal
            )
            logger.debug(
                "tool_execution_completed",
                tool_name=fn_name,
                success=result.is_success,
                duration=result.duration,
            )

            # Save to cache
            if self.cache and tool.cacheable and result.is_success:
                self.cache.set(context.session_id, fn_name, args, result)

            return result
        except asyncio.CancelledError:
            logger.info("tool_execution_cancelled", tool_name=fn_name)
            return ToolResult.error(
                tool_call_id=call_id,
                tool_name=fn_name,
                error="Tool execution was cancelled",
                start_time=start_time,
            )
        except Exception as e:
            logger.error(
                "tool_execution_exception",
                tool_name=fn_name,
                error=str(e),
                exc_info=True,
            )
            return ToolResult.error(
                tool_call_id=call_id,
                tool_name=fn_name,
                error=f"Tool execution failed: {e}",
                start_time=start_time,
            )

    def _create_denied_result(
        self,
        call_id: str,
        tool_name: str,
        reason: str,
        start_time: float,
    ) -> ToolResult:
        """
        Create authorization denied ToolResult.

        Critical: Must return explicit ToolResult with error information,
        so LLM can understand why tool call failed.
        """
        end_time = time.time()
        return ToolResult(
            tool_name=tool_name,
            tool_call_id=call_id,
            input_args={},
            content=(
                f"Tool execution denied: {reason}. "
                "Please ask the user for permission or use a different approach."
            ),
            output=None,
            error=f"Permission denied: {reason}",
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            is_success=False,
        )
