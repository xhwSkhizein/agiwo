"""
HttpRequestTool — make HTTP GET/POST requests.
"""

import time
from typing import Any

import httpx

from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.utils.abort_signal import AbortSignal


@default_enable
@builtin_tool("http_request")
class HttpRequestTool(BaseTool):
    timeout_seconds: int = 30

    def get_name(self) -> str:
        return "http_request"

    def get_description(self) -> str:
        return (
            "Make an HTTP request (GET or POST) to a URL and return the response. "
            "Useful for fetching web pages, calling APIs, etc."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to request.",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method: GET or POST. Default: GET.",
                    "enum": ["GET", "POST"],
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers as key-value pairs.",
                },
                "body": {
                    "type": "string",
                    "description": "Optional request body (for POST requests).",
                },
            },
            "required": ["url"],
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
        url = parameters.get("url", "")
        method = parameters.get("method", "GET").upper()
        headers = parameters.get("headers") or {}
        body = parameters.get("body")

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                if method == "POST":
                    resp = await client.post(url, headers=headers, content=body)
                else:
                    resp = await client.get(url, headers=headers)

            text = resp.text
            if len(text) > 8000:
                text = text[:8000] + "\n... (truncated)"

            content = (
                f"Status: {resp.status_code}\n"
                f"Content-Type: {resp.headers.get('content-type', 'unknown')}\n\n"
                f"{text}"
            )
        except Exception as e:
            end = time.time()
            return ToolResult.error(
                tool_name=self.name,
                error=f"HTTP request failed: {e}",
                input_args=parameters,
                start_time=start,
            )

        end = time.time()
        return ToolResult(
            tool_name=self.name,
            tool_call_id="",
            input_args=parameters,
            content=content,
            output={"status_code": resp.status_code, "body": resp.text},
            start_time=start,
            end_time=end,
            duration=end - start,
        )
