"""Background process management tool for jobs started by BashTool."""

import shlex
from dataclasses import dataclass
from typing import Any

from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.context import ToolContext
from agiwo.tool.process import AgentProcessRegistry
from agiwo.tool.builtin.bash_tool.sandbox import get_shared_local_sandbox
from agiwo.tool.builtin.bash_tool.parameter_parser import BashParameterParser
from agiwo.tool.builtin.bash_tool.result_formatter import truncate_output
from agiwo.tool.builtin.bash_tool.types import CommandResult, ProcessInfo, Sandbox
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.utils.abort_signal import AbortSignal


@dataclass
class BashProcessToolConfig:
    """Configuration for BashProcessTool."""

    sandbox: Sandbox
    max_output_length: int = 30000


@dataclass
class BashLogsRequest:
    """Normalized request for reading job logs."""

    job_id: str
    stream: str
    tail: int
    grep: str | None
    context_lines: int
    ignore_case: bool


@default_enable
@builtin_tool("bash_process")
class BashProcessTool(BaseTool, AgentProcessRegistry):
    """Manage background processes started by the bash tool."""

    cacheable: bool = False
    timeout_seconds: int = 30

    def __init__(self, config: BashProcessToolConfig | None = None) -> None:
        self.config = config or BashProcessToolConfig(
            sandbox=get_shared_local_sandbox(),
        )
        super().__init__()

    def get_name(self) -> str:
        return "bash_process"

    def get_description(self) -> str:
        return (
            "Manage background jobs started by the `bash` tool. "
            "Choose an `action` to list jobs, inspect status, read logs, stop a job, "
            "show log paths, or write stdin to a running PTY job."
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["jobs", "status", "logs", "stop", "paths", "input"],
                    "description": "Management action to perform.",
                },
                "job_id": {
                    "type": "string",
                    "description": "Tracked process id. Required for all actions except `jobs`.",
                },
                "running_only": {
                    "type": "boolean",
                    "description": "When action=`jobs`, only return running jobs.",
                },
                "stream": {
                    "type": "string",
                    "enum": ["all", "stdout", "stderr"],
                    "description": "When action=`logs`, select which stream to read.",
                },
                "tail": {
                    "type": "integer",
                    "description": "When action=`logs`, tail this many lines. Default 200.",
                },
                "grep": {
                    "type": "string",
                    "description": "When action=`logs`, filter matching lines.",
                },
                "context": {
                    "type": "integer",
                    "description": "When action=`logs`, include N surrounding lines per match.",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "When action=`logs`, ignore case for grep matches.",
                },
                "force": {
                    "type": "boolean",
                    "description": "When action=`stop`, send KILL instead of TERM.",
                },
                "text": {
                    "type": "string",
                    "description": "When action=`input`, text to write to stdin.",
                },
                "append_newline": {
                    "type": "boolean",
                    "description": "When action=`input`, append a trailing newline. Default true.",
                },
            },
            "required": ["action"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        tool_call_id = context.tool_call_id

        action = parameters.get("action")
        if not isinstance(action, str):
            return self._error(
                parameters,
                "action is required",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        handlers = {
            "jobs": self._jobs,
            "status": self._status,
            "logs": self._logs,
            "stop": self._stop,
            "paths": self._paths,
            "input": self._input,
        }
        handler = handlers.get(action)
        if handler is None:
            return self._error(
                parameters,
                ("action must be one of: jobs, status, logs, stop, paths, input"),
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        try:
            return await handler(parameters, tool_call_id)
        except TimeoutError:
            return self._error(
                parameters, "log retrieval timed out", tool_call_id=tool_call_id
            )
        except (RuntimeError, ValueError, OSError) as exc:
            return self._error(parameters, str(exc), tool_call_id=tool_call_id)

    async def list_agent_processes(
        self,
        agent_id: str,
        *,
        state: str = "running",
    ) -> list[dict[str, object]]:
        if state not in {"running", "all"}:
            raise ValueError("state must be one of: running, all")
        processes = await self.config.sandbox.list_processes_by_agent(
            agent_id, state=state
        )
        return [self._serialize_process(process) for process in processes]

    async def _jobs(
        self, parameters: dict[str, Any], tool_call_id: str = ""
    ) -> ToolResult:
        running_only = self._parse_bool(parameters.get("running_only"))
        if parameters.get("running_only") is not None and running_only is None:
            return self._error(
                parameters,
                "running_only must be a boolean",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        state = "running" if running_only else "all"
        jobs = await self.config.sandbox.list_processes(state=state)
        if not jobs:
            return self._ok(
                parameters,
                tool_call_id=tool_call_id,
                stdout="no jobs\n",
                count=0,
                jobs=[],
            )

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

        return self._ok(
            parameters,
            tool_call_id=tool_call_id,
            stdout="\n".join(lines) + "\n",
            count=len(jobs),
            jobs=[self._serialize_process(job) for job in jobs],
        )

    async def _status(
        self, parameters: dict[str, Any], tool_call_id: str = ""
    ) -> ToolResult:
        job_id = self._require_job_id(parameters)
        if job_id is None:
            return self._error(
                parameters,
                "status requires job_id",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        try:
            status = await self.config.sandbox.get_process_status(job_id)
            info = await self.config.sandbox.attach_process(job_id)
        except KeyError:
            return self._error(
                parameters,
                f"job not found: {job_id}",
                tool_call_id=tool_call_id,
                exit_code=1,
            )

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
            parameters,
            tool_call_id=tool_call_id,
            stdout=output,
            job_id=job_id,
            state=status.state,
            mode=status.mode,
            exit_code_value=status.exit_code,
            command=info.command,
        )

    async def _paths(
        self, parameters: dict[str, Any], tool_call_id: str = ""
    ) -> ToolResult:
        job_id = self._require_job_id(parameters)
        if job_id is None:
            return self._error(
                parameters,
                "paths requires job_id",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        try:
            logs = await self.config.sandbox.get_process_logs_info(job_id)
        except KeyError:
            return self._error(
                parameters,
                f"job not found: {job_id}",
                tool_call_id=tool_call_id,
                exit_code=1,
            )

        output = (
            f"mode: {logs.mode}\n"
            f"stdout: {logs.stdout_path}\n"
            f"stderr: {logs.stderr_path}\n"
        )
        return self._ok(
            parameters,
            tool_call_id=tool_call_id,
            stdout=output,
            job_id=job_id,
            stdout_path=logs.stdout_path,
            stderr_path=logs.stderr_path,
            mode=logs.mode,
        )

    async def _stop(
        self, parameters: dict[str, Any], tool_call_id: str = ""
    ) -> ToolResult:
        job_id = self._require_job_id(parameters)
        if job_id is None:
            return self._error(
                parameters,
                "stop requires job_id",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        force = self._parse_bool(parameters.get("force"))
        if parameters.get("force") is not None and force is None:
            return self._error(
                parameters,
                "force must be a boolean",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        signal_name = "KILL" if force else "TERM"
        try:
            await self.config.sandbox.stop_process(job_id, signal=signal_name)
        except KeyError:
            return self._error(
                parameters,
                f"job not found: {job_id}",
                tool_call_id=tool_call_id,
                exit_code=1,
            )

        return self._ok(
            parameters,
            tool_call_id=tool_call_id,
            stdout=f"stopped {job_id} with {signal_name}\n",
            job_id=job_id,
            signal=signal_name,
        )

    async def _input(
        self, parameters: dict[str, Any], tool_call_id: str = ""
    ) -> ToolResult:
        job_id = self._require_job_id(parameters)
        if job_id is None:
            return self._error(
                parameters,
                "input requires job_id",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        payload = parameters.get("text")
        if not isinstance(payload, str) or not payload:
            return self._error(
                parameters,
                "input requires non-empty text",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        append_newline = self._parse_bool(parameters.get("append_newline"))
        if parameters.get("append_newline") is not None and append_newline is None:
            return self._error(
                parameters,
                "append_newline must be a boolean",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        data = payload
        if append_newline is not False:
            data += "\n"

        try:
            await self.config.sandbox.write_process_stdin(job_id, data)
        except KeyError:
            return self._error(
                parameters,
                f"job not found: {job_id}",
                tool_call_id=tool_call_id,
                exit_code=1,
            )
        except (RuntimeError, ValueError) as exc:
            return self._error(
                parameters, str(exc), tool_call_id=tool_call_id, exit_code=1
            )

        return self._ok(
            parameters,
            tool_call_id=tool_call_id,
            stdout=f"sent input to {job_id}\n",
            job_id=job_id,
            bytes=len(data.encode("utf-8")),
        )

    async def _logs(
        self, parameters: dict[str, Any], tool_call_id: str = ""
    ) -> ToolResult:
        request = self._resolve_logs_request(parameters, tool_call_id=tool_call_id)
        if isinstance(request, ToolResult):
            return request

        try:
            log_info = await self.config.sandbox.get_process_logs_info(request.job_id)
        except KeyError:
            return self._error(
                parameters,
                f"job not found: {request.job_id}",
                tool_call_id=tool_call_id,
                exit_code=1,
            )

        source = self._log_source_command(
            log_info.stdout_path,
            log_info.stderr_path,
            request.stream,
            mode=log_info.mode,
        )
        if request.grep:
            grep_flags = "-n"
            if request.ignore_case:
                grep_flags += " -i"
            context_flags = (
                f" -C {request.context_lines}" if request.context_lines > 0 else ""
            )
            query = shlex.quote(request.grep)
            log_command = (
                f"{source} | grep {grep_flags}{context_flags} -- {query} || true"
            )
        else:
            log_command = f"{source} | tail -n {request.tail}"

        result = await self.config.sandbox.execute_command(log_command)
        return self._from_command_result(
            parameters,
            result,
            tool_call_id=tool_call_id,
            job_id=request.job_id,
            stream=request.stream,
            logs_command=log_command,
            mode=log_info.mode,
        )

    def _from_command_result(
        self,
        parameters: dict[str, Any],
        result: CommandResult,
        *,
        tool_call_id: str = "",
        **extra: Any,
    ) -> ToolResult:
        payload = {
            "ok": result.exit_code == 0,
            "action": str(parameters.get("action", "")),
            "stdout": truncate_output(
                result.stdout,
                self.config.max_output_length,
                "stdout",
            ),
            "stderr": truncate_output(
                result.stderr,
                self.config.max_output_length,
                "stderr",
            ),
            "exit_code": result.exit_code,
        }
        payload.update(extra)

        content = (
            f"exit_code: {payload['exit_code']}\n"
            f"stdout: {payload['stdout']}\n"
            f"stderr: {payload['stderr']}"
        )
        if result.exit_code == 0:
            return self._success(
                parameters, tool_call_id=tool_call_id, content=content, output=payload
            )
        return ToolResult.failed(
            tool_name=self.name,
            error=str(payload["stderr"]),
            tool_call_id=tool_call_id,
            input_args=parameters,
            content=content,
            output=payload,
        )

    def _ok(
        self,
        parameters: dict[str, Any],
        *,
        tool_call_id: str = "",
        stdout: str = "",
        **extra: Any,
    ) -> ToolResult:
        payload = {
            "ok": True,
            "action": str(parameters.get("action", "")),
            "stdout": truncate_output(
                stdout,
                self.config.max_output_length,
                "stdout",
            ),
            "stderr": "",
            "exit_code": 0,
        }
        payload.update(extra)
        content = f"exit_code: {payload['exit_code']}\nstdout: {payload['stdout']}"
        return self._success(
            parameters, tool_call_id=tool_call_id, content=content, output=payload
        )

    def _error(
        self,
        parameters: dict[str, Any],
        message: str,
        *,
        tool_call_id: str = "",
        exit_code: int = 1,
        **extra: Any,
    ) -> ToolResult:
        payload = {
            "ok": False,
            "action": str(parameters.get("action", "")),
            "stdout": "",
            "stderr": truncate_output(
                message,
                self.config.max_output_length,
                "stderr",
            ),
            "exit_code": exit_code,
        }
        payload.update(extra)
        content = f"exit_code: {payload['exit_code']}\nstderr: {payload['stderr']}"
        return ToolResult.failed(
            tool_name=self.name,
            error=str(payload["stderr"]),
            tool_call_id=tool_call_id,
            input_args=parameters,
            content=content,
            output=payload,
        )

    def _success(
        self,
        parameters: dict[str, Any],
        *,
        tool_call_id: str = "",
        content: str,
        output: dict[str, Any],
    ) -> ToolResult:
        return ToolResult.success(
            tool_name=self.name,
            content=content,
            tool_call_id=tool_call_id,
            input_args=parameters,
            output=output,
        )

    @staticmethod
    def _require_job_id(parameters: dict[str, Any]) -> str | None:
        job_id = parameters.get("job_id")
        if isinstance(job_id, str) and job_id.strip():
            return job_id.strip()
        return None

    @staticmethod
    def _parse_bool(value: object) -> bool | None:
        return BashParameterParser.parse_bool(value)

    @staticmethod
    def _coerce_int(
        value: object,
        *,
        positive: bool,
        default: int,
    ) -> int | None:
        if value is None:
            return default
        if not isinstance(value, int):
            return None
        if positive and value <= 0:
            return None
        if not positive and value < 0:
            return None
        return value

    def _resolve_logs_request(
        self,
        parameters: dict[str, Any],
        *,
        tool_call_id: str = "",
    ) -> BashLogsRequest | ToolResult:
        job_id = self._require_job_id(parameters)
        if job_id is None:
            return self._error(
                parameters,
                "logs requires job_id",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        stream = parameters.get("stream", "all")
        if not isinstance(stream, str) or stream not in {"all", "stdout", "stderr"}:
            return self._error(
                parameters,
                "stream must be one of: all, stdout, stderr",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        tail = self._coerce_int(parameters.get("tail"), positive=True, default=200)
        if tail is None:
            return self._error(
                parameters,
                "tail must be a positive integer",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        context_lines = self._coerce_int(
            parameters.get("context"),
            positive=False,
            default=0,
        )
        if context_lines is None:
            return self._error(
                parameters,
                "context must be a non-negative integer",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        filters = self._parse_log_filters(parameters, tool_call_id=tool_call_id)
        if isinstance(filters, ToolResult):
            return filters

        return BashLogsRequest(
            job_id=job_id,
            stream=stream,
            tail=tail,
            grep=filters[0],
            context_lines=context_lines,
            ignore_case=filters[1],
        )

    def _parse_log_filters(
        self,
        parameters: dict[str, Any],
        *,
        tool_call_id: str = "",
    ) -> tuple[str | None, bool] | ToolResult:
        grep = parameters.get("grep")
        if grep is not None and not isinstance(grep, str):
            return self._error(
                parameters,
                "grep must be a string",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        ignore_case = self._parse_bool(parameters.get("ignore_case"))
        if parameters.get("ignore_case") is not None and ignore_case is None:
            return self._error(
                parameters,
                "ignore_case must be a boolean",
                tool_call_id=tool_call_id,
                exit_code=2,
            )

        return grep, bool(ignore_case)

    @staticmethod
    def _log_source_command(
        stdout_path: str,
        stderr_path: str,
        stream: str,
        *,
        mode: str = "pipe",
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
    def _serialize_process(process: ProcessInfo) -> dict[str, object]:
        return {
            "process_id": process.process_id,
            "command": process.command,
            "state": process.state,
            "mode": process.mode,
            "started_at": process.started_at,
            "exit_code": process.exit_code,
        }


__all__ = ["BashProcessTool", "BashProcessToolConfig"]
