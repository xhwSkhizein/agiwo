import asyncio
from dataclasses import dataclass, replace
import time
from typing import Any

from agiwo.agent.engine.context import AgentRunContext
from agiwo.agent.runtime_tools import (
    AgentRuntimeTool,
    RuntimeToolLike,
    RuntimeToolOutcome,
    adapt_runtime_tool,
)
from agiwo.llm.message_converter import parse_json_tool_args
from agiwo.tool.base import ToolGateDecision, ToolResult
from agiwo.tool.cache import ToolResultCache
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolRuntimeOptions:
    timeout_seconds: int = 30
    cache_max_items: int = 1000
    cache_ttl_seconds: int = 3600


@dataclass(frozen=True)
class ResolvedToolCall:
    raw_call: dict[str, Any]
    call_id: str
    tool_name: str
    tool: AgentRuntimeTool
    args: dict[str, Any]

    def with_args(self, args: dict[str, Any]) -> "ResolvedToolCall":
        updated = dict(args)
        updated["tool_call_id"] = self.call_id
        return replace(self, args=updated)


class ToolRuntime:
    """Execute agent-visible tools from LLM tool_call payloads."""

    def __init__(
        self,
        tools: list[RuntimeToolLike],
        cache: ToolResultCache | None = None,
        options: ToolRuntimeOptions | None = None,
    ) -> None:
        runtime_tools = [adapt_runtime_tool(tool) for tool in tools]
        self.tools = runtime_tools
        self.tools_map = {tool.get_name(): tool for tool in runtime_tools}
        self.cache = cache
        self.options = options or ToolRuntimeOptions()

    def resolve_tool_call(
        self,
        tool_call: dict[str, Any],
        *,
        start_time: float | None = None,
    ) -> ResolvedToolCall | ToolResult:
        resolved_start = start_time if start_time is not None else time.time()
        call_id = _get_string(tool_call, "id")
        function_payload = _get_function_payload(tool_call)
        tool_name = _get_string(function_payload, "name")
        if not tool_name:
            return ToolResult.failed(
                tool_call_id=call_id,
                tool_name="unknown",
                error="Tool name missing in tool call",
                start_time=resolved_start,
            )

        tool = self.tools_map.get(tool_name)
        if tool is None:
            return ToolResult.failed(
                tool_call_id=call_id,
                tool_name=tool_name,
                error=f"Tool {tool_name} not found",
                start_time=resolved_start,
            )

        try:
            args = parse_json_tool_args(function_payload.get("arguments", {}))
        except ValueError as error:
            return ToolResult.failed(
                tool_call_id=call_id,
                tool_name=tool_name,
                error=str(error),
                start_time=resolved_start,
            )

        args["tool_call_id"] = call_id
        return ResolvedToolCall(
            raw_call=tool_call,
            call_id=call_id,
            tool_name=tool_name,
            tool=tool,
            args=args,
        )

    async def execute_batch(
        self,
        tool_calls: list[dict[str, Any]],
        context: AgentRunContext,
        abort_signal: AbortSignal | None = None,
    ) -> list[RuntimeToolOutcome]:
        prepared: list[ResolvedToolCall | ToolResult] = [
            self.resolve_tool_call(tool_call) for tool_call in tool_calls
        ]
        return await self.execute_resolved_batch(
            prepared,
            context=context,
            abort_signal=abort_signal,
        )

    async def execute_resolved_batch(
        self,
        prepared_calls: list[ResolvedToolCall | ToolResult],
        *,
        context: AgentRunContext,
        abort_signal: AbortSignal | None = None,
    ) -> list[RuntimeToolOutcome]:
        async def _run_single(
            prepared: ResolvedToolCall | ToolResult,
        ) -> RuntimeToolOutcome:
            if isinstance(prepared, ToolResult):
                return RuntimeToolOutcome(result=prepared)
            try:
                return await self.execute_resolved(
                    prepared,
                    context=context,
                    abort_signal=abort_signal,
                )
            except asyncio.CancelledError:
                return RuntimeToolOutcome(
                    result=ToolResult.failed(
                        tool_call_id=prepared.call_id,
                        tool_name=prepared.tool_name,
                        error="Tool execution was cancelled",
                        start_time=time.time(),
                    )
                )
            except Exception as error:  # noqa: BLE001 - defensive runtime boundary
                logger.error(
                    "tool_batch_execute_error",
                    tool_name=prepared.tool_name,
                    tool_call_id=prepared.call_id,
                    error=str(error),
                    exc_info=True,
                )
                return RuntimeToolOutcome(
                    result=ToolResult.failed(
                        tool_call_id=prepared.call_id,
                        tool_name=prepared.tool_name,
                        error=f"Tool execution failed: {error}",
                        start_time=time.time(),
                    )
                )

        safe_indices: list[int] = []
        unsafe_indices: list[int] = []
        for index, prepared in enumerate(prepared_calls):
            if (
                isinstance(prepared, ResolvedToolCall)
                and not prepared.tool.is_concurrency_safe()
            ):
                unsafe_indices.append(index)
            else:
                safe_indices.append(index)

        results: list[RuntimeToolOutcome | None] = [None] * len(prepared_calls)
        if safe_indices:
            safe_results = await asyncio.gather(
                *(_run_single(prepared_calls[index]) for index in safe_indices)
            )
            for index, result in zip(safe_indices, safe_results):
                results[index] = result

        for index in unsafe_indices:
            results[index] = await _run_single(prepared_calls[index])

        return results  # type: ignore[return-value]

    async def execute(
        self,
        tool_call: dict[str, Any],
        context: AgentRunContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome:
        resolved = self.resolve_tool_call(tool_call)
        if isinstance(resolved, ToolResult):
            return RuntimeToolOutcome(result=resolved)
        return await self.execute_resolved(
            resolved,
            context=context,
            abort_signal=abort_signal,
        )

    async def execute_resolved(
        self,
        resolved: ResolvedToolCall,
        *,
        context: AgentRunContext,
        abort_signal: AbortSignal | None = None,
    ) -> RuntimeToolOutcome:
        start_time = time.time()
        gate_decision = await self._gate_tool_call(resolved, context)
        if gate_decision.action == "deny":
            return RuntimeToolOutcome(
                result=ToolResult.denied(
                    tool_name=resolved.tool_name,
                    reason=gate_decision.reason,
                    tool_call_id=resolved.call_id,
                    input_args=resolved.args,
                    start_time=start_time,
                )
            )

        timeout_seconds = resolved.tool.timeout_seconds or self.options.timeout_seconds
        execution_context = replace(
            context,
            metadata={
                **context.metadata,
                "_tool_gate_checked": True,
            },
        )
        if timeout_seconds:
            execution_context = replace(
                execution_context,
                timeout_at=time.time() + timeout_seconds,
            )

        if self.cache and resolved.tool.cacheable:
            cached_result = self.cache.get(
                context.session_id,
                resolved.tool_name,
                resolved.args,
            )
            if cached_result is not None:
                now = time.time()
                return RuntimeToolOutcome(
                    result=replace(
                        cached_result,
                        tool_call_id=resolved.call_id,
                        start_time=now,
                        end_time=now,
                        duration=0.0,
                    )
                )

        try:
            logger.debug(
                "executing_tool",
                tool_name=resolved.tool_name,
                tool_call_id=resolved.call_id,
            )
            outcome = await resolved.tool.execute_for_agent(
                resolved.args,
                context=execution_context,
                abort_signal=abort_signal,
            )
            result = outcome.result
            result.tool_call_id = resolved.call_id or ""
            logger.debug(
                "tool_execution_completed",
                tool_name=resolved.tool_name,
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
                and resolved.tool.cacheable
                and result.is_success
                and outcome.termination_reason is None
            ):
                self.cache.set(
                    context.session_id,
                    resolved.tool_name,
                    resolved.args,
                    result,
                )
            return outcome
        except asyncio.CancelledError:
            logger.info("tool_execution_cancelled", tool_name=resolved.tool_name)
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_call_id=resolved.call_id,
                    tool_name=resolved.tool_name,
                    error="Tool execution was cancelled",
                    start_time=start_time,
                )
            )
        except Exception as error:  # noqa: BLE001 - runtime boundary
            logger.error(
                "tool_execution_exception",
                tool_name=resolved.tool_name,
                error=str(error),
                exc_info=True,
            )
            return RuntimeToolOutcome(
                result=ToolResult.failed(
                    tool_call_id=resolved.call_id,
                    tool_name=resolved.tool_name,
                    error=f"Tool execution failed: {error}",
                    start_time=start_time,
                )
            )

    async def _gate_tool_call(
        self,
        resolved: ResolvedToolCall,
        context: AgentRunContext,
    ) -> ToolGateDecision:
        gate_for_agent = getattr(resolved.tool, "gate_for_agent", None)
        if gate_for_agent is None:
            return ToolGateDecision.allow()
        return await gate_for_agent(resolved.args, context)


def _get_function_payload(tool_call: dict[str, Any]) -> dict[str, Any]:
    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        return function_payload
    return {}


def _get_string(payload: dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else default


__all__ = ["ResolvedToolCall", "ToolRuntime", "ToolRuntimeOptions"]
