"""
CalculatorTool — safely evaluate mathematical expressions.
"""

import ast
import math
import operator
import time
from typing import Any

from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.utils.abort_signal import AbortSignal

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_func = SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_func = SAFE_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            func = SAFE_FUNCTIONS[node.func.id]
            if callable(func):
                args = [_safe_eval(arg) for arg in node.args]
                return func(*args)
            return func
        raise ValueError(f"Unsupported function: {ast.dump(node.func)}")
    if isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            val = SAFE_FUNCTIONS[node.id]
            if not callable(val):
                return val
        raise ValueError(f"Unsupported name: {node.id}")
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@default_enable
@builtin_tool("calculator")
class CalculatorTool(BaseTool):

    def get_name(self) -> str:
        return "calculator"

    def get_description(self) -> str:
        return (
            "Evaluate a mathematical expression safely. "
            "Supports: +, -, *, /, //, %, ** and functions: "
            "abs, round, min, max, sum, sqrt, log, log10, sin, cos, tan, pi, e."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. '2 ** 10 + sqrt(144)'.",
                },
            },
            "required": ["expression"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        start = time.time()
        expression = parameters.get("expression", "")

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            content = f"{expression} = {result}"
        except Exception as e:
            end = time.time()
            return ToolResult.error(
                tool_name=self.name,
                error=f"Failed to evaluate '{expression}': {e}",
                input_args=parameters,
                start_time=start,
            )

        end = time.time()
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            input_args=parameters,
            content=content,
            output=result,
            start_time=start,
            end_time=end,
            duration=end - start,
        )
