"""Unified terminal-style BashTool implementation."""

from __future__ import annotations

import time
import shlex
from dataclasses import dataclass
from typing import Any, Callable, Literal

from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.utils.abort_signal import AbortSignal
from agiwo.tool.builtin.bash_tool.security import CommandSafetyValidator
from agiwo.tool.builtin.bash_tool.types import (
    AfterBashCallInput,
    AfterBashCallOutput,
    BeforeBashCallInput,
    BeforeBashCallOutput,
    CommandResult,
    Sandbox,
)


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
            from agiwo.tool.builtin.bash_tool.sandbox.local import LocalSandbox

            config = BashToolConfig(
                sandbox=LocalSandbox(),
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
        lines = [
            "Terminal-style bash tool.",
            "Pass one shell command via `command`.",
            "Set `background=true` to start a background job.",
            "Set `pty=true` for interactive CLI commands that require a TTY.",
            "Use `bashctl` for job management (help/jobs/status/logs/stop/paths/input).",
            "Built-in safety guard blocks destructive commands."
        ]
        if self.config.extra_instructions:
            lines.append(self.config.extra_instructions)
        return "\n".join(lines)

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
        """Execute a terminal command or a built-in bashctl command."""
        start = time.time()
        command = str(parameters.get("command", "")).strip()
        if not command:
            return self._error(
                start=start, parameters=parameters, message="command is required"
            )

        cwd = str(parameters.get("cwd") or self.config.cwd or "").strip() or None

        timeout_value = parameters.get("timeout")
        timeout: float | None = None
        if timeout_value is not None:
            try:
                timeout = float(timeout_value)
            except (TypeError, ValueError):
                return self._error(
                    start=start,
                    parameters=parameters,
                    message="timeout must be a number",
                )

        background_value = parameters.get("background")
        background_enabled = self._parse_bool(background_value)
        if background_value is not None and background_enabled is None:
            return self._error(
                start=start,
                parameters=parameters,
                message="background must be a boolean",
            )
        background = bool(background_enabled)

        pty_value = parameters.get("pty")
        pty_enabled = self._parse_bool(pty_value)
        if pty_value is not None and pty_enabled is None:
            return self._error(
                start=start,
                parameters=parameters,
                message="pty must be a boolean",
            )
        use_pty = bool(pty_enabled)

        stdin_value = parameters.get("stdin")
        stdin: str | None = None
        if stdin_value is not None and not isinstance(stdin_value, str):
            return self._error(
                start=start,
                parameters=parameters,
                message="stdin must be a string",
            )
        if isinstance(stdin_value, str):
            stdin = stdin_value

        try:
            if self._is_bashctl(command):
                return await self._execute_bashctl(start, parameters, command)
            if command.rstrip().endswith("&"):
                return self._error(
                    start=start,
                    parameters=parameters,
                    message="trailing '&' is not supported; use background=true",
                    exit_code=2,
                )
            return await self._execute_shell(
                start=start,
                parameters=parameters,
                command=command,
                cwd=cwd,
                timeout=timeout,
                background=background,
                context=context,
                use_pty=use_pty,
                stdin=stdin,
            )
        except TimeoutError:
            return self._error(
                start=start, parameters=parameters, message="command timed out"
            )
        except Exception as exc:
            return self._error(start=start, parameters=parameters, message=str(exc))

    async def _execute_shell(
        self,
        start: float,
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
            return self._error(
                start=start,
                parameters=parameters,
                message="command cannot be empty",
            )

        if background and stdin is not None:
            return self._error(
                start=start,
                parameters=parameters,
                message="stdin is only supported for foreground PTY execution",
            )
        if stdin is not None and not use_pty:
            return self._error(
                start=start,
                parameters=parameters,
                message="stdin requires pty=true",
            )

        safety = self.config.command_safety_validator
        if safety is not None:
            safety_decision = await safety.validate(foreground_command)
            if not safety_decision.allowed:
                return self._error(
                    start=start,
                    parameters=parameters,
                    message=safety_decision.message,
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
                start=start,
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
            start,
            parameters,
            command,
            result,
            mode="pty" if use_pty else "pipe",
        )

    async def _execute_bashctl(
        self, start: float, parameters: dict[str, Any], command: str
    ) -> ToolResult:
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            return self._error(
                start=start,
                parameters=parameters,
                message=f"bashctl parse error: {exc}",
                exit_code=2,
            )

        if len(tokens) == 1:
            return self._ok(
                start=start,
                parameters=parameters,
                command=command,
                stdout=self._bashctl_help(),
            )

        subcommand = tokens[1]
        args = tokens[2:]

        if subcommand in {"help", "--help", "-h"}:
            topic = args[0] if args else None
            return self._ok(
                start=start,
                parameters=parameters,
                command=command,
                stdout=self._bashctl_help(topic),
            )

        handlers = {
            "jobs": self._bashctl_jobs,
            "status": self._bashctl_status,
            "logs": self._bashctl_logs,
            "stop": self._bashctl_stop,
            "paths": self._bashctl_paths,
            "input": self._bashctl_input,
        }
        handler = handlers.get(subcommand)
        if handler is None:
            return self._error(
                start=start,
                parameters=parameters,
                message=f"unknown bashctl subcommand: {subcommand}\n\n{self._bashctl_help()}",
                exit_code=2,
            )
        return await handler(start, parameters, args)

    async def _bashctl_jobs(
        self, start: float, parameters: dict[str, Any], args: list[str]
    ) -> ToolResult:
        running_only = False
        for arg in args:
            if arg in {"-r", "--running"}:
                running_only = True
                continue
            if arg in {"-h", "--help"}:
                return self._ok(start, parameters, stdout=self._bashctl_help("jobs"))
            return self._error(start, parameters, f"unknown flag for jobs: {arg}", exit_code=2)

        state: Literal["running", "all"] = "running" if running_only else "all"
        jobs = await self.config.sandbox.list_processes(state=state)

        if not jobs:
            return self._ok(start, parameters, stdout="no jobs\n", count=0)

        header = "JOB ID    STATE    MODE   EXIT   COMMAND"
        lines = [header]
        for job in jobs:
            exit_code = "-" if job.exit_code is None else str(job.exit_code)
            summary = " ".join(job.command.split())
            if len(summary) > 80:
                summary = f"{summary[:77]}..."
            lines.append(
                f"{job.process_id:<8}  {job.state:<7}  {job.mode:<5}  {exit_code:<4}  {summary}"
            )

        output = "\n".join(lines) + "\n"
        return self._ok(start, parameters, stdout=output, count=len(jobs))

    async def _bashctl_status(
        self, start: float, parameters: dict[str, Any], args: list[str]
    ) -> ToolResult:
        if not args or args[0] in {"-h", "--help"}:
            return self._ok(start, parameters, stdout=self._bashctl_help("status"))

        job_id = args[0]
        if len(args) > 1:
            return self._error(
                start, parameters, "status accepts exactly one <job_id>", exit_code=2
            )

        try:
            status = await self.config.sandbox.get_process_status(job_id)
            info = await self.config.sandbox.attach_process(job_id)
        except KeyError:
            return self._error(start, parameters, f"job not found: {job_id}", exit_code=1)

        output = (
            "\n".join(
                [
                    f"job_id: {job_id}",
                    f"state: {status.state}",
                    f"mode: {status.mode}",
                    f"exit_code: {status.exit_code if status.exit_code is not None else '-'}",
                    f"started_at: {status.started_at or '-'}",
                    f"command: {info.command}",
                ]
            )
            + "\n"
        )
        return self._ok(
            start,
            parameters,
            stdout=output,
            job_id=job_id,
            state=status.state,
            mode=status.mode,
            exit_code_value=status.exit_code,
        )

    async def _bashctl_paths(
        self, start: float, parameters: dict[str, Any], args: list[str]
    ) -> ToolResult:
        if not args or args[0] in {"-h", "--help"}:
            return self._ok(start, parameters, stdout=self._bashctl_help("paths"))

        job_id = args[0]
        if len(args) > 1:
            return self._error(
                start, parameters, "paths accepts exactly one <job_id>", exit_code=2
            )

        try:
            logs = await self.config.sandbox.get_process_logs_info(job_id)
        except KeyError:
            return self._error(start, parameters, f"job not found: {job_id}", exit_code=1)

        output = (
            f"mode: {logs.mode}\n"
            f"stdout: {logs.stdout_path}\n"
            f"stderr: {logs.stderr_path}\n"
        )
        return self._ok(
            start,
            parameters,
            stdout=output,
            job_id=job_id,
            stdout_path=logs.stdout_path,
            stderr_path=logs.stderr_path,
            mode=logs.mode,
        )

    async def _bashctl_stop(
        self, start: float, parameters: dict[str, Any], args: list[str]
    ) -> ToolResult:
        force = False
        job_id: str | None = None

        for arg in args:
            if arg in {"-h", "--help"}:
                return self._ok(start, parameters, stdout=self._bashctl_help("stop"))
            if arg in {"-f", "--force"}:
                force = True
                continue
            if arg.startswith("-"):
                return self._error(
                    start, parameters, f"unknown flag for stop: {arg}", exit_code=2
                )
            if job_id is None:
                job_id = arg
                continue
            return self._error(
                start, parameters, "stop accepts exactly one <job_id>", exit_code=2
            )

        if not job_id:
            return self._error(start, parameters, "stop requires <job_id>", exit_code=2)

        signal = "KILL" if force else "TERM"
        try:
            await self.config.sandbox.stop_process(job_id, signal=signal)
        except KeyError:
            return self._error(start, parameters, f"job not found: {job_id}", exit_code=1)

        return self._ok(
            start, parameters, stdout=f"stopped {job_id} with {signal}\n", job_id=job_id
        )

    async def _bashctl_input(
        self, start: float, parameters: dict[str, Any], args: list[str]
    ) -> ToolResult:
        if not args or args[0] in {"-h", "--help"}:
            return self._ok(start, parameters, stdout=self._bashctl_help("input"))

        job_id = args[0]
        append_newline = True
        payload_parts: list[str] = []
        for arg in args[1:]:
            if arg in {"--no-newline", "-n"}:
                append_newline = False
                continue
            payload_parts.append(arg)

        if not payload_parts:
            return self._error(
                start,
                parameters,
                "input requires text payload after <job_id>",
                exit_code=2,
            )

        payload = " ".join(payload_parts)
        if append_newline:
            payload += "\n"

        try:
            await self.config.sandbox.write_process_stdin(job_id, payload)
        except KeyError:
            return self._error(start, parameters, f"job not found: {job_id}", exit_code=1)
        except (RuntimeError, ValueError) as exc:
            return self._error(start, parameters, str(exc), exit_code=1)

        return self._ok(
            start,
            parameters,
            stdout=f"sent input to {job_id}\n",
            job_id=job_id,
            bytes=len(payload.encode("utf-8")),
        )

    async def _bashctl_logs(
        self, start: float, parameters: dict[str, Any], args: list[str]
    ) -> ToolResult:
        if not args or args[0] in {"-h", "--help"}:
            return self._ok(start, parameters, stdout=self._bashctl_help("logs"))

        job_id = args[0]
        stream = "all"
        tail = 200
        grep: str | None = None
        context = 0
        ignore_case = False

        index = 1
        while index < len(args):
            arg = args[index]

            if arg in {"-i", "--ignore-case"}:
                ignore_case = True
                index += 1
                continue

            if arg in {"-n", "--tail"}:
                index += 1
                if index >= len(args):
                    return self._error(start, parameters, "--tail requires a value", exit_code=2)
                parsed_tail = self._parse_positive_int(args[index])
                if parsed_tail is None:
                    return self._error(
                        start, parameters, "--tail expects a positive integer", exit_code=2
                    )
                tail = parsed_tail
                index += 1
                continue

            if arg.startswith("--tail="):
                value = arg.split("=", 1)[1]
                parsed_tail = self._parse_positive_int(value)
                if parsed_tail is None:
                    return self._error(
                        start, parameters, "--tail expects a positive integer", exit_code=2
                    )
                tail = parsed_tail
                index += 1
                continue

            if arg in {"-C", "--context"}:
                index += 1
                if index >= len(args):
                    return self._error(
                        start, parameters, "--context requires a value", exit_code=2
                    )
                parsed_context = self._parse_non_negative_int(args[index])
                if parsed_context is None:
                    return self._error(
                        start, parameters, "--context expects a non-negative integer", exit_code=2
                    )
                context = parsed_context
                index += 1
                continue

            if arg.startswith("--context="):
                value = arg.split("=", 1)[1]
                parsed_context = self._parse_non_negative_int(value)
                if parsed_context is None:
                    return self._error(
                        start, parameters, "--context expects a non-negative integer", exit_code=2
                    )
                context = parsed_context
                index += 1
                continue

            if arg == "--grep":
                index += 1
                if index >= len(args):
                    return self._error(start, parameters, "--grep requires a value", exit_code=2)
                grep = args[index]
                index += 1
                continue

            if arg.startswith("--grep="):
                grep = arg.split("=", 1)[1]
                index += 1
                continue

            if arg == "--stream":
                index += 1
                if index >= len(args):
                    return self._error(
                        start, parameters, "--stream requires a value", exit_code=2
                    )
                stream = args[index]
                index += 1
                continue

            if arg.startswith("--stream="):
                stream = arg.split("=", 1)[1]
                index += 1
                continue

            return self._error(start, parameters, f"unknown logs flag: {arg}", exit_code=2)

        if stream not in {"all", "stdout", "stderr"}:
            return self._error(
                start, parameters, "--stream must be one of: all, stdout, stderr", exit_code=2
            )

        try:
            log_info = await self.config.sandbox.get_process_logs_info(job_id)
        except KeyError:
            return self._error(start, parameters, f"job not found: {job_id}", exit_code=1)

        source = self._log_source_command(
            log_info.stdout_path, log_info.stderr_path, stream, mode=log_info.mode
        )
        if grep:
            grep_flags = "-n"
            if ignore_case:
                grep_flags += " -i"
            context_flags = f" -C {context}" if context > 0 else ""
            query = shlex.quote(grep)
            log_command = (
                f"{source} | grep {grep_flags}{context_flags} -- {query} || true"
            )
        else:
            log_command = f"{source} | tail -n {tail}"

        result = await self.config.sandbox.execute_command(log_command)
        return self._from_command_result(
            start,
            parameters,
            str(parameters.get("command", "")).strip(),
            result,
            job_id=job_id,
            stream=stream,
            logs_command=log_command,
            mode=log_info.mode,
        )

    @staticmethod
    def _log_source_command(
        stdout_path: str, stderr_path: str, stream: str, mode: str = "pipe"
    ) -> str:
        quoted_stdout = shlex.quote(stdout_path)
        quoted_stderr = shlex.quote(stderr_path)
        if mode == "pty":
            if stream == "stderr":
                return "cat /dev/null"
            return f"cat {quoted_stdout}"
        if stream == "all":
            return f"cat {quoted_stdout} {quoted_stderr}"
        if stream == "stdout":
            return f"cat {quoted_stdout}"
        return f"cat {quoted_stderr}"

    @staticmethod
    def _is_bashctl(command: str) -> bool:
        stripped = command.lstrip()
        return stripped == "bashctl" or stripped.startswith("bashctl ")

    def _from_command_result(
        self,
        start: float,
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
        end = time.time()
        return ToolResult(
            tool_name=self.name,
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content=content,
            output=payload,
            start_time=start,
            end_time=end,
            duration=end - start,
        )

    def _ok(
        self, start: float, parameters: dict[str, Any], stdout: str = "", **extra: Any
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
        end = time.time()
        return ToolResult(
            tool_name=self.name,
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content=content,
            output=payload,
            start_time=start,
            end_time=end,
            duration=end - start,
        )

    def _error(
        self,
        start: float,
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
        end = time.time()
        return ToolResult(
            tool_name=self.name,
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content=content,
            output=payload,
            start_time=start,
            end_time=end,
            duration=end - start,
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
    def _parse_positive_int(value: str) -> int | None:
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _parse_non_negative_int(value: str) -> int | None:
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None

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

    @staticmethod
    def _bashctl_help(topic: str | None = None) -> str:
        general = """bashctl - manage background jobs started by this tool

Usage:
  bashctl help [command]
  bashctl jobs [--running]
  bashctl status <job_id>
  bashctl logs <job_id> [--stream all|stdout|stderr] [-n N] [--grep TEXT] [-C N] [-i]
  bashctl stop <job_id> [--force]
  bashctl paths <job_id>
  bashctl input <job_id> <text...> [--no-newline]

Examples:
  bashctl jobs --running
  bashctl status a1b2c3d4
  bashctl logs a1b2c3d4 --stream stderr --grep molt -C 10
  bashctl stop a1b2c3d4 --force
  bashctl input a1b2c3d4 "help"
"""

        details = {
            "jobs": """Usage: bashctl jobs [--running]\nList tracked jobs; use --running to filter active jobs only.\n""",
            "status": """Usage: bashctl status <job_id>\nShow one job's state, exit code, start time, and command.\n""",
            "logs": """Usage: bashctl logs <job_id> [options]\nOptions:\n  --stream all|stdout|stderr\n  -n, --tail N\n  --grep TEXT\n  -C, --context N\n  -i, --ignore-case\n""",
            "stop": """Usage: bashctl stop <job_id> [--force]\nSend TERM by default; use --force to send KILL.\n""",
            "paths": """Usage: bashctl paths <job_id>\nShow stdout/stderr log file paths for a job.\n""",
            "input": """Usage: bashctl input <job_id> <text...> [--no-newline]\nWrite text to a running PTY job stdin. Appends newline by default.\n""",
        }

        if topic is None:
            return general
        return details.get(topic, f"Unknown help topic: {topic}\n\n{general}")
