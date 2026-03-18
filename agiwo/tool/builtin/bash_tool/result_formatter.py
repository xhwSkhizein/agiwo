"""Result formatting for BashTool."""

from typing import Any

from agiwo.tool.base import ToolResult
from agiwo.tool.builtin.bash_tool.types import CommandResult


def truncate_output(output: str, limit: int, stream: str) -> str:
    """Truncate output when it exceeds the configured size limit."""
    if len(output) <= limit:
        return output
    # Reserve space for the suffix
    suffix = f"\n[{stream} truncated: N characters removed]"
    suffix_len = len(suffix) + 10  # Extra space for the number
    available = max(1, limit - suffix_len)
    removed = len(output) - available
    suffix = f"\n[{stream} truncated: {removed} characters removed]"
    return output[:available] + suffix


class BashResultFormatter:
    """Builds ToolResult instances for bash command outcomes."""

    def __init__(self, tool_name: str, max_output_length: int) -> None:
        self._tool_name = tool_name
        self._max_output_length = max_output_length

    def from_command_result(
        self,
        parameters: dict[str, Any],
        command: str,
        result: CommandResult,
        **extra: Any,
    ) -> ToolResult:
        payload = {
            "ok": result.exit_code == 0,
            "command": command,
            "stdout": truncate_output(
                result.stdout, self._max_output_length, "stdout"
            ),
            "stderr": truncate_output(
                result.stderr, self._max_output_length, "stderr"
            ),
            "exit_code": result.exit_code,
        }
        payload.update(extra)

        content = f"exit_code: {payload['exit_code']}\nstdout: {payload['stdout']}\nstderr: {payload['stderr']}"
        if result.exit_code == 0:
            return ToolResult.success(
                tool_name=self._tool_name,
                content=content,
                tool_call_id=str(parameters.get("tool_call_id", "")),
                input_args=parameters,
                output=payload,
            )
        # Fall back to exit code message if stderr is empty
        error_msg = str(payload["stderr"])
        if not error_msg:
            error_msg = f"command exited with code {payload['exit_code']}"
        return ToolResult.failed(
            tool_name=self._tool_name,
            error=error_msg,
            tool_call_id=str(parameters.get("tool_call_id", "")),
            input_args=parameters,
            content=content,
            output=payload,
        )

    def ok(
        self,
        parameters: dict[str, Any],
        stdout: str = "",
        **extra: Any,
    ) -> ToolResult:
        payload = {
            "ok": True,
            "command": str(parameters.get("command", "")).strip(),
            "stdout": truncate_output(stdout, self._max_output_length, "stdout"),
            "stderr": "",
            "exit_code": 0,
        }
        payload.update(extra)

        content = f"exit_code: {payload['exit_code']}\nstdout: {payload['stdout']}"
        return ToolResult.success(
            tool_name=self._tool_name,
            tool_call_id=str(parameters.get("tool_call_id", "")),
            input_args=parameters,
            content=content,
            output=payload,
        )

    def error(
        self,
        parameters: dict[str, Any],
        message: str,
        exit_code: int = 1,
        **extra: Any,
    ) -> ToolResult:
        command = str(parameters.get("command", "")).strip()
        payload = {
            "ok": False,
            "command": command,
            "stdout": "",
            "stderr": truncate_output(message, self._max_output_length, "stderr"),
            "exit_code": exit_code,
        }
        payload.update(extra)

        content = f"exit_code: {payload['exit_code']}\nstderr: {payload['stderr']}"
        return ToolResult.failed(
            tool_name=self._tool_name,
            tool_call_id=str(parameters.get("tool_call_id", "")),
            input_args=parameters,
            content=content,
            output=payload,
            error=str(payload["stderr"]),
        )
