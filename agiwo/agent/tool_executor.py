"""Tool executor — execute tools directly from LLM tool_call payloads."""

import asyncio
import time
from typing import Any

from agiwo.agent.runtime.context import RunContext
from agiwo.llm.message_converter import parse_json_tool_args
from agiwo.tool.base import BaseTool, ToolGateDecision, ToolResult
from agiwo.tool.context import ToolContext
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


async def execute_tool_batch(
    tool_calls: list[dict[str, Any]],
    *,
    tools_map: dict[str, BaseTool],
    context: RunContext,
    abort_signal: AbortSignal | None = None,
) -> list[ToolResult]:
    results: dict[int, ToolResult] = {}
    safe_indices: list[int] = []
    unsafe_indices: list[int] = []

    for index, tool_call in enumerate(tool_calls):
        tool_name = _get_tool_name(tool_call)
        tool = tools_map.get(tool_name) if tool_name else None
        if tool is not None and not tool.concurrency_safe:
            unsafe_indices.append(index)
        else:
            safe_indices.append(index)

    if safe_indices:
        safe_results = await asyncio.gather(
            *(
                _execute_tool_call(
                    tool_calls[index],
                    tools_map=tools_map,
                    context=context,
                    abort_signal=abort_signal,
                )
                for index in safe_indices
            )
        )
        for index, result in zip(safe_indices, safe_results):
            results[index] = result

    for index in unsafe_indices:
        results[index] = await _execute_tool_call(
            tool_calls[index],
            tools_map=tools_map,
            context=context,
            abort_signal=abort_signal,
        )

    assert len(results) == len(tool_calls), "All tool calls must produce a result"
    return [results[i] for i in range(len(tool_calls))]


async def _execute_tool_call(
    tool_call: dict[str, Any],
    *,
    tools_map: dict[str, BaseTool],
    context: RunContext,
    abort_signal: AbortSignal | None = None,
) -> ToolResult:
    start_time = time.time()
    prepared = await _prepare_tool_call(
        tool_call,
        tools_map=tools_map,
        context=context,
        start_time=start_time,
    )
    if isinstance(prepared, ToolResult):
        return prepared
    call_id, tool_name, tool, args = prepared

    return await _execute_prepared_tool_call(
        call_id=call_id,
        tool_name=tool_name,
        tool=tool,
        args=args,
        context=context,
        abort_signal=abort_signal,
        start_time=start_time,
    )


async def _prepare_tool_call(
    tool_call: dict[str, Any],
    *,
    tools_map: dict[str, BaseTool],
    context: RunContext,
    start_time: float,
) -> ToolResult | tuple[str, str, BaseTool, dict[str, Any]]:
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

    tool = tools_map.get(tool_name)
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

    if context.hooks.on_before_tool_call:
        modified = await context.hooks.on_before_tool_call(
            call_id, tool_name, dict(args)
        )
        if modified is not None:
            args = dict(modified)

    return call_id, tool_name, tool, args


async def _execute_prepared_tool_call(
    *,
    call_id: str,
    tool_name: str,
    tool: BaseTool,
    args: dict[str, Any],
    context: RunContext,
    abort_signal: AbortSignal | None,
    start_time: float,
) -> ToolResult:
    try:
        tool_context = tool.build_context(context, tool_call_id=call_id)
        gate_decision = await _gate_tool_call(tool, args, tool_context)
        if gate_decision.action == "deny":
            return ToolResult.denied(
                tool_name=tool_name,
                reason=gate_decision.reason,
                tool_call_id=call_id,
                input_args=args,
                start_time=start_time,
            )

        logger.debug(
            "executing_tool",
            tool_name=tool_name,
            tool_call_id=call_id,
        )
        result = await tool.execute(
            args, context=tool_context, abort_signal=abort_signal
        )
        result.tool_call_id = call_id or ""
        logger.debug(
            "tool_execution_completed",
            tool_name=tool_name,
            success=result.is_success,
            duration=result.duration,
            termination_reason=(
                result.termination_reason.value
                if result.termination_reason is not None
                else None
            ),
        )
        return result
    except asyncio.CancelledError:
        logger.info("tool_execution_cancelled", tool_name=tool_name)
        return ToolResult.failed(
            tool_call_id=call_id,
            tool_name=tool_name,
            error="Tool execution was cancelled",
            start_time=start_time,
        )
    except Exception as error:  # noqa: BLE001 - runtime boundary
        logger.error(
            "tool_execution_exception",
            tool_name=tool_name,
            tool_call_id=call_id,
            error=str(error),
            exc_info=True,
        )
        return ToolResult.failed(
            tool_call_id=call_id,
            tool_name=tool_name,
            error=f"Tool execution failed: {error}",
            start_time=start_time,
        )


async def _gate_tool_call(
    tool: BaseTool,
    args: dict[str, Any],
    context: ToolContext,
) -> ToolGateDecision:
    return await tool.gate(args, context=context)


def _get_tool_name(tool_call: dict[str, Any]) -> str:
    return _get_string(_get_function_payload(tool_call), "name")


def _get_function_payload(tool_call: dict[str, Any]) -> dict[str, Any]:
    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        return function_payload
    return {}


def _get_string(payload: dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else default


__all__ = ["execute_tool_batch"]
