"""Unified terminal-style BashTool implementation."""

from dataclasses import dataclass
from typing import Any, Callable

from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.builtin.bash_tool.sandbox import get_shared_local_sandbox
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.tool.builtin.bash_tool.security import CommandSafetyValidator
from agiwo.tool.builtin.bash_tool.types import (
    AfterBashCallInput,
    AfterBashCallOutput,
    BeforeBashCallInput,
    BeforeBashCallOutput,
    CommandResult,
    Sandbox,
)
from agiwo.utils.abort_signal import AbortSignal


@dataclass
class BashToolConfig:
    """Configuration for BashTool."""

    sandbox: Sandbox
    cwd: str
    extra_instructions: str | None = None
    on_before_bash_call: (
        Callable[[BeforeBashCallInput], BeforeBashCallOutput | None] | None
    ) = None
    on_after_bash_call: (
        Callable[[AfterBashCallInput], AfterBashCallOutput | None] | None
    ) = None
    command_safety_validator: CommandSafetyValidator | None = None
    max_output_length: int = 30000


@dataclass
class BashExecutionRequest:
    """Normalized bash execution request."""

    command: str
    cwd: str | None
    timeout: float | None
    background: bool
    use_pty: bool
    stdin: str | None


def truncate_output(output: str, limit: int, stream: str) -> str:
    """Truncate output when it exceeds the configured size limit."""
    if len(output) <= limit:
        return output
    removed = len(output) - limit
    return f"{output[:limit]}\n[{stream} truncated: {removed} characters removed]"


@default_enable
@builtin_tool("bash")
class BashTool(BaseTool):
    """Single tool entry that behaves like a terminal command line."""

    cacheable: bool = False
    timeout_seconds: int = 30

    def __init__(self, config: BashToolConfig | None = None) -> None:
        if config is None:
            config = BashToolConfig(
                sandbox=get_shared_local_sandbox(),
                cwd=".",
                command_safety_validator=CommandSafetyValidator(),
            )
        elif config.command_safety_validator is None:
            config.command_safety_validator = CommandSafetyValidator()
        self.config = config

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        lines = (
            "Terminal-style bash tool. Pass one shell command via `command`. "
            "Set `background=true` to start a background job. "
            "Use the separate `bash_process` tool to inspect, stop, or feed background jobs. "
            "Set `pty=true` for interactive CLI commands that require a TTY. "
            "Built-in safety guard blocks destructive commands."
        )
        if self.config.extra_instructions:
            lines += " " + self.config.extra_instructions
        return lines

    @property
    def parameters(self) -> dict[str, Any]:
        return self.get_parameters()

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Full command line to run in bash, like in a terminal.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory relative to workspace root.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout seconds for foreground command execution.",
                },
                "background": {
                    "type": "boolean",
                    "description": "Run command as background job and return job_id immediately.",
                },
                "pty": {
                    "type": "boolean",
                    "description": "Enable PTY mode for commands that require a terminal.",
                },
                "stdin": {
                    "type": "string",
                    "description": "Input sent to stdin for foreground PTY commands.",
                },
            },
            "required": ["command"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal

        request = self._resolve_request(parameters)
        if isinstance(request, ToolResult):
            return request

        try:
            return await self._execute_shell(
                parameters=parameters,
                command=request.command,
                cwd=request.cwd,
                timeout=request.timeout,
                background=request.background,
                context=context,
                use_pty=request.use_pty,
                stdin=request.stdin,
            )
        except TimeoutError:
            return self._error(parameters, "command timed out")
        except (RuntimeError, ValueError, OSError) as exc:
            return self._error(parameters, str(exc))

    async def _execute_shell(
        self,
        parameters: dict[str, Any],
        command: str,
        cwd: str | None,
        timeout: float | None,
        background: bool,
        context: ExecutionContext,
        use_pty: bool,
        stdin: str | None,
    ) -> ToolResult:
        shell_command = self._apply_before_hook(command)
        foreground_command = shell_command.strip()
        if not foreground_command:
            return self._error(parameters, "command cannot be empty")

        if background and stdin is not None:
            return self._error(
                parameters,
                "stdin is only supported for foreground PTY execution",
            )
        if stdin is not None and not use_pty:
            return self._error(parameters, "stdin requires pty=true")

        safety = self.config.command_safety_validator
        if safety is not None:
            safety_decision = await safety.validate(foreground_command)
            if not safety_decision.allowed:
                return self._error(
                    parameters,
                    safety_decision.message,
                    exit_code=126,
                    security=safety_decision.to_dict(),
                )

        if background:
            agent_id = getattr(context, "agent_id", None)
            job_id = await self.config.sandbox.start_process(
                foreground_command,
                cwd=cwd,
                agent_id=agent_id,
                use_pty=use_pty,
            )
            return self._ok(
                parameters=parameters,
                stdout=f"started background job {job_id}\n",
                job_id=job_id,
                state="running",
                background=True,
                mode="pty" if use_pty else "pipe",
            )

        result: CommandResult = await self.config.sandbox.execute_command(
            foreground_command,
            cwd=cwd,
            timeout=timeout,
            use_pty=use_pty,
            stdin=stdin,
        )
        result: CommandResult = self._apply_after_hook(foreground_command, result)
        return self._from_command_result(
            parameters,
            command,
            result,
            mode="pty" if use_pty else "pipe",
        )

    def _from_command_result(
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
                result.stdout, self.config.max_output_length, "stdout"
            ),
            "stderr": truncate_output(
                result.stderr, self.config.max_output_length, "stderr"
            ),
            "exit_code": result.exit_code,
        }
        payload.update(extra)

        content = f"exit_code: {payload['exit_code']}\nstdout: {payload['stdout']}\nstderr: {payload['stderr']}"
        if result.exit_code == 0:
            return ToolResult.success(
                tool_name=self.name,
                content=content,
                tool_call_id=str(parameters.get("tool_call_id", "")),
                input_args=parameters,
                output=payload,
            )
        return ToolResult.failed(
            tool_name=self.name,
            error=str(payload["stderr"]),
            tool_call_id=str(parameters.get("tool_call_id", "")),
            input_args=parameters,
            content=content,
            output=payload,
        )

    def _ok(
        self,
        parameters: dict[str, Any],
        stdout: str = "",
        **extra: Any,
    ) -> ToolResult:
        payload = {
            "ok": True,
            "command": str(parameters.get("command", "")).strip(),
            "stdout": truncate_output(stdout, self.config.max_output_length, "stdout"),
            "stderr": "",
            "exit_code": 0,
        }
        payload.update(extra)

        content = f"exit_code: {payload['exit_code']}\nstdout: {payload['stdout']}"
        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=str(parameters.get("tool_call_id", "")),
            input_args=parameters,
            content=content,
            output=payload,
        )

    def _error(
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
            "stderr": truncate_output(message, self.config.max_output_length, "stderr"),
            "exit_code": exit_code,
        }
        payload.update(extra)

        content = f"exit_code: {payload['exit_code']}\nstderr: {payload['stderr']}"
        return ToolResult.failed(
            tool_name=self.name,
            tool_call_id=str(parameters.get("tool_call_id", "")),
            input_args=parameters,
            content=content,
            output=payload,
            error=str(payload["stderr"]),
        )

    def _apply_before_hook(self, command: str) -> str:
        callback = self.config.on_before_bash_call
        if callback is None:
            return command
        output = callback(BeforeBashCallInput(command=command))
        return command if output is None else output.command

    def _apply_after_hook(self, command: str, result: CommandResult) -> CommandResult:
        callback = self.config.on_after_bash_call
        if callback is None:
            return result
        output = callback(AfterBashCallInput(command=command, result=result))
        return result if output is None else output.result

    @staticmethod
    def _parse_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        return None

    def _parse_timeout(self, parameters: dict[str, Any]) -> float | None | ToolResult:
        timeout_value = parameters.get("timeout")
        if timeout_value is None:
            return None
        try:
            return float(timeout_value)
        except (TypeError, ValueError):
            return self._error(parameters, "timeout must be a number")

    def _parse_flag(
        self,
        parameters: dict[str, Any],
        *,
        key: str,
    ) -> bool | ToolResult:
        value = parameters.get(key)
        parsed = self._parse_bool(value)
        if value is not None and parsed is None:
            return self._error(parameters, f"{key} must be a boolean")
        return bool(parsed)

    def _parse_stdin(self, parameters: dict[str, Any]) -> str | None | ToolResult:
        stdin_value = parameters.get("stdin")
        if stdin_value is None:
            return None
        if not isinstance(stdin_value, str):
            return self._error(parameters, "stdin must be a string")
        return stdin_value

    def _resolve_request(
        self,
        parameters: dict[str, Any],
    ) -> BashExecutionRequest | ToolResult:
        command = str(parameters.get("command", "")).strip()
        if not command:
            return self._error(parameters, "command is required")

        timeout = self._parse_timeout(parameters)
        if isinstance(timeout, ToolResult):
            return timeout

        modes = self._parse_modes(parameters)
        if isinstance(modes, ToolResult):
            return modes

        stdin = self._parse_stdin(parameters)
        if isinstance(stdin, ToolResult):
            return stdin

        if command.rstrip().endswith("&"):
            return self._error(
                parameters,
                "trailing '&' is not supported; use background=true",
                exit_code=2,
            )

        return BashExecutionRequest(
            command=command,
            cwd=str(parameters.get("cwd") or self.config.cwd or "").strip() or None,
            timeout=timeout,
            background=modes[0],
            use_pty=modes[1],
            stdin=stdin,
        )

    def _parse_modes(
        self,
        parameters: dict[str, Any],
    ) -> tuple[bool, bool] | ToolResult:
        background = self._parse_flag(parameters, key="background")
        if isinstance(background, ToolResult):
            return background

        use_pty = self._parse_flag(parameters, key="pty")
        if isinstance(use_pty, ToolResult):
            return use_pty

        return background, use_pty
