"""
CurrentTimeTool — returns the current date, time, and timezone.
"""

import time
from datetime import datetime
from typing import Any

from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.utils.abort_signal import AbortSignal


@default_enable
@builtin_tool("current_time")
class CurrentTimeTool(BaseTool):

    def get_name(self) -> str:
        return "current_time"

    def get_description(self) -> str:
        return "Get the current date, time, and timezone information."

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Output format: 'iso' for ISO-8601, 'human' for human-readable. Default: 'human'.",
                    "enum": ["iso", "human"],
                },
            },
            "required": [],
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
        fmt = parameters.get("format", "human")
        now = datetime.now().astimezone()

        if fmt == "iso":
            result = now.isoformat()
        else:
            tz_name = now.strftime("%Z") or str(now.tzinfo)
            result = (
                f"Date: {now.strftime('%Y-%m-%d')}\n"
                f"Time: {now.strftime('%H:%M:%S')}\n"
                f"Timezone: {tz_name}\n"
                f"Unix timestamp: {int(now.timestamp())}"
            )

        end = time.time()
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            input_args=parameters,
            content=result,
            output=result,
            start_time=start,
            end_time=end,
            duration=end - start,
        )
