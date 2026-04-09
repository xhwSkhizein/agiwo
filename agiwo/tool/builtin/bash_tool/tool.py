"""Unified terminal-style BashTool implementation."""

from dataclasses import dataclass
from typing import Any, Callable

from agiwo.tool.base import BaseTool, ToolGateDecision, ToolResult
from agiwo.tool.builtin.bash_tool.parameter_parser import (
    BashParameterParser,
    ParseError,
)
from agiwo.tool.builtin.bash_tool.result_formatter import BashResultFormatter
from agiwo.tool.builtin.bash_tool.security import CommandSafetyValidator
from agiwo.tool.builtin.bash_tool.sandbox import get_shared_local_sandbox
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.tool.builtin.bash_tool.types import (
    AfterBashCallInput,
    AfterBashCallOutput,
    BeforeBashCallInput,
    BeforeBashCallOutput,
    CommandResult,
    Sandbox,
)
from agiwo.tool.context import ToolContext
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


@default_enable
@builtin_tool("bash")
class BashTool(BaseTool):
    """Single tool entry that behaves like a terminal command line."""

    name = "bash"
    cacheable: bool = False
    timeout_seconds: int = 30

    def __init__(self, config: BashToolConfig | None = None) -> None:
        if config is None:
            config = BashToolConfig(
                sandbox=get_shared_local_sandbox(),
                cwd=".",
            )
        self.config = config
        self._parser = BashParameterParser()
        self._formatter = BashResultFormatter("bash", config.max_output_length)
        self._safety_validator = CommandSafetyValidator()

    @property
    def description(self) -> str:
        lines = (
            "Terminal-style bash tool. Pass one shell command via `command`. "
            "Set `background=true` to start a background job. "
            "Use the separate `bash_process` tool to inspect, stop, or feed background jobs. "
            "Set `pty=true` for interactive CLI commands that require a TTY. "
            "Built-in safety guard blocks destructive commands, and risky commands may require confirmation."
        )
        if self.config.extra_instructions:
            lines += " " + self.config.extra_instructions
        return lines

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

    async def gate(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
    ) -> ToolGateDecision:
        del context
        command = str(parameters.get("command", "")).strip()
        if not command:
            return ToolGateDecision.allow()
        decision = await self._safety_validator.validate(command)
        return ToolGateDecision(action=decision.action, reason=decision.reason)

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        tool_call_id = context.tool_call_id

        request = self._resolve_request(parameters, tool_call_id=tool_call_id)
        if isinstance(request, ToolResult):
            return request

        if not context.gate_checked:
            safety_decision = await self._safety_validator.validate(request.command)
            if safety_decision.action == "deny":
                return self._formatter.error(
                    parameters, safety_decision.reason, tool_call_id=tool_call_id
                )

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
                abort_signal=abort_signal,
            )
        except TimeoutError:
            return self._formatter.error(
                parameters, "command timed out", tool_call_id=tool_call_id
            )
        except (RuntimeError, ValueError, OSError) as exc:
            return self._formatter.error(
                parameters, str(exc), tool_call_id=tool_call_id
            )

    async def _execute_shell(
        self,
        parameters: dict[str, Any],
        command: str,
        cwd: str | None,
        timeout: float | None,
        background: bool,
        context: ToolContext,
        use_pty: bool,
        stdin: str | None,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        tool_call_id = context.tool_call_id
        shell_command = self._apply_before_hook(command)
        foreground_command = shell_command.strip()
        if not foreground_command:
            return self._formatter.error(
                parameters, "command cannot be empty", tool_call_id=tool_call_id
            )

        if background and stdin is not None:
            return self._formatter.error(
                parameters,
                "stdin is only supported for foreground PTY execution",
                tool_call_id=tool_call_id,
            )
        if stdin is not None and not use_pty:
            return self._formatter.error(
                parameters, "stdin requires pty=true", tool_call_id=tool_call_id
            )

        if background:
            agent_id = context.agent_id
            job_id = await self.config.sandbox.start_process(
                foreground_command,
                cwd=cwd,
                env=self._build_agent_env(context),
                agent_id=agent_id,
                use_pty=use_pty,
            )
            return self._formatter.ok(
                parameters,
                tool_call_id=tool_call_id,
                stdout=f"started background job {job_id}\n",
                job_id=job_id,
                state="running",
                background=True,
                mode="pty" if use_pty else "pipe",
            )

        result: CommandResult = await self.config.sandbox.execute_command(
            foreground_command,
            cwd=cwd,
            env=self._build_agent_env(context),
            timeout=timeout,
            use_pty=use_pty,
            stdin=stdin,
            abort_signal=abort_signal,
        )
        result = self._apply_after_hook(foreground_command, result)
        return self._formatter.from_command_result(
            parameters,
            foreground_command,
            result,
            tool_call_id=tool_call_id,
            mode="pty" if use_pty else "pipe",
        )

    def _build_agent_env(self, context: ToolContext) -> dict[str, str] | None:
        """Build environment variables with X_AGENT_ID for the command."""
        if context.agent_id is None:
            return None
        return {"X_AGENT_ID": context.agent_id}

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

    def _resolve_request(
        self,
        parameters: dict[str, Any],
        *,
        tool_call_id: str = "",
    ) -> BashExecutionRequest | ToolResult:
        command = str(parameters.get("command", "")).strip()
        if not command:
            return self._formatter.error(
                parameters, "command is required", tool_call_id=tool_call_id
            )

        timeout = self._parser.parse_timeout(parameters)
        if isinstance(timeout, ParseError):
            return self._formatter.error(
                parameters, timeout.message, tool_call_id=tool_call_id
            )

        modes = self._parser.parse_modes(parameters)
        if isinstance(modes, ParseError):
            return self._formatter.error(
                parameters, modes.message, tool_call_id=tool_call_id
            )

        stdin = self._parser.parse_stdin(parameters)
        if isinstance(stdin, ParseError):
            return self._formatter.error(
                parameters, stdin.message, tool_call_id=tool_call_id
            )

        if command.rstrip().endswith("&"):
            return self._formatter.error(
                parameters,
                "trailing '&' is not supported; use background=true",
                tool_call_id=tool_call_id,
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
