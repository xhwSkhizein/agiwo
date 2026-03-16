import asyncio
from dataclasses import dataclass, replace
import time
from typing import Any

from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.runtime_tools import (
    AgentRuntimeTool,
    RuntimeToolLike,
    RuntimeToolOutcome,
    adapt_runtime_tool,
)
from agiwo.agent.tool_auth import ToolAuthorizationRuntime
from agiwo.llm.helper import parse_json_tool_args
from agiwo.tool.base import ToolResult
from agiwo.tool.cache import ToolResultCache
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolRuntimeOptions:
    timeout_seconds: int = 30
    cache_max_items: int = 1000
    cache_ttl_seconds: int = 3600
    auth_timeout_seconds: float = 300.0


class ToolRuntime:
    """Execute agent-visible tools from LLM tool_call payloads."""

    def __init__(
        self,
        tools: list[RuntimeToolLike],
        cache: ToolResultCache | None = None,
        options: ToolRuntimeOptions | None = None,
        auth_runtime: ToolAuthorizationRuntime | None = None,
    ) -> None:
        runtime_tools = [adapt_runtime_tool(tool) for tool in tools]
        self.tools = runtime_tools
        self.tools_map = {tool.get_name(): tool for tool in runtime_tools}
        self.cache = cache
        self.options = options or ToolRuntimeOptions()
        self.auth_runtime = auth_runtime or ToolAuthorizationRuntime()

    async def execute_batch(
        self,
        tool_calls: list[dict[str, Any]],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> list[RuntimeToolOutcome]:
        async def _run_single(tc: dict[str, Any]) -> RuntimeToolOutcome:
            try:
                return await self.execute(
                    tc,
                    context=context,
                    abort_signal=abort_signal,
                )
            except asyncio.CancelledError:
                fn = _get_function_payload(tc)
                tool_name = _get_string(fn, "name", "unknown")
                call_id = _get_string(tc, "id")
                return RuntimeToolOutcome(
                    result=ToolResult.failed(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        error="Tool execution was cancelled",
                        start_time=time.time(),
                    )
                )
            except Exception as error:  # noqa: BLE001 - defensive runtime boundary
                fn = _get_function_payload(tc)
                tool_name = _get_string(fn, "name", "unknown")
                call_id = _get_string(tc, "id")
                logger.error(
                    "tool_batch_execute_error",
                    tool_name=tool_name,
                    tool_call_id=call_id,
                    error=str(error),
                    exc_info=True,
                )
                return RuntimeToolOutcome(
                    result=ToolResult.failed(
                        tool_call_id=call_id,
                        tool_name=tool_name,
                        error=f"Tool execution failed: {error}",
                        start_time=time.time(),
                    )
                )

        safe_indices: list[int] = []
        unsafe_indices: list[int] = []
        for index, tool_call in enumerate(tool_calls):
            fn = _get_function_payload(tool_call)
            name = _get_string(fn, "name")
            tool = self.tools_map.get(name)
            if tool is not None and not tool.is_concurrency_safe():
                unsafe_indices.append(index)
            else:
                safe_indices.append(index)

        results: list[RuntimeToolOutcome | None] = [None] * len(tool_calls)

        if safe_indices:
            safe_results = await asyncio.gather(
                *(_run_single(tool_calls[index]) for index in safe_indices)
            )
            for index, result in zip(safe_indices, safe_results):
                results[index] = result

        for index in unsafe_indices:
            results[index] = await _run_single(tool_calls[index])

        return results  # type: ignore[return-value]

    async def execute(
        self,
        tool_call: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome:
        start_time = time.time()
        resolved = self._resolve_tool_call(tool_call, start_time)
        if isinstance(resolved, ToolResult):
            return RuntimeToolOutcome(result=resolved)
        call_id, tool_name, tool, args = resolved

        auth = await self.auth_runtime.authorize(
            tool_call_id=call_id,
            tool_name=tool_name,
            tool_args=args,
            context=context,
            timeout=self.options.auth_timeout_seconds,
        )
        if not auth.allowed:
            return RuntimeToolOutcome(
                result=ToolResult.denied(
                    tool_name=tool_name,
                    reason=auth.reason,
                    tool_call_id=call_id,
                    input_args=args,
                    start_time=start_time,
                )
            )

        timeout_seconds = tool.timeout_seconds or self.options.timeout_seconds
        execution_context = context
        if timeout_seconds:
            execution_context = replace(
                context,
                timeout_at=time.time() + timeout_seconds,
            )

        if self.cache and tool.cacheable:
            cached_result = self.cache.get(context.session_id, tool_name, args)
            if cached_result is not None:
                now = time.time()
                return RuntimeToolOutcome(
                    result=replace(
                        cached_result,
                        tool_call_id=call_id,
                        start_time=now,
                        end_time=now,
                        duration=0.0,
                    )
                )

        try:
            logger.debug(
                "executing_tool",
                tool_name=tool_name,
                tool_call_id=call_id,
            )
            outcome = await tool.execute_for_agent(
                args,
                context=execution_context,
                abort_signal=abort_signal,
            )
            result = outcome.result
            result.tool_call_id = call_id or ""
            logger.debug(
                "tool_execution_completed",
                tool_name=tool_name,
                success=result.is_success,
                duration=result.duration,
                termination_reason=(
                    outcome.termination_reason.value
                    if outcome.termination_reason is not None
                    else None
                ),
            )
            if (
                self.cache
                and tool.cacheable
                and result.is_success
                and outcome.termination_reason is None
            ):
                self.cache.set(context.session_id, tool_name, args, result)
            return outcome
        except asyncio.CancelledError:
            logger.info("tool_execution_cancelled", tool_name=tool_name)
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    error="Tool execution was cancelled",
                    start_time=start_time,
                )
            )
        except Exception as error:  # noqa: BLE001 - runtime boundary
            logger.error(
                "tool_execution_exception",
                tool_name=tool_name,
                error=str(error),
                exc_info=True,
            )
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_call_id=call_id,
                    tool_name=tool_name,
                    error=f"Tool execution failed: {error}",
                    start_time=start_time,
                )
            )

    def _resolve_tool_call(
        self,
        tool_call: dict[str, Any],
        start_time: float,
    ) -> tuple[str, str, AgentRuntimeTool, dict[str, Any]] | ToolResult:
        call_id = _get_string(tool_call, "id")
        function_payload = _get_function_payload(tool_call)
        tool_name = _get_string(function_payload, "name")
        if not tool_name:
            return ToolResult.failed(
                tool_call_id=call_id,
                tool_name="unknown",
                error="Tool name missing in tool call",
                start_time=start_time,
            )

        tool = self.tools_map.get(tool_name)
        if tool is None:
            return ToolResult.failed(
                tool_call_id=call_id,
                tool_name=tool_name,
                error=f"Tool {tool_name} not found",
                start_time=start_time,
            )

        try:
            args = parse_json_tool_args(function_payload.get("arguments", {}))
        except ValueError as error:
            return ToolResult.failed(
                tool_call_id=call_id,
                tool_name=tool_name,
                error=str(error),
                start_time=start_time,
            )

        args["tool_call_id"] = call_id
        return call_id, tool_name, tool, args


def _get_function_payload(tool_call: dict[str, Any]) -> dict[str, Any]:
    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        return function_payload
    return {}


def _get_string(payload: dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else default
