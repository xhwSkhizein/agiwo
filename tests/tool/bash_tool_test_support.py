"""Shared test support for bash tool and bash process tool."""

import time
import pytest

from agiwo.tool.builtin.bash_tool.process_tool import (
    BashProcessTool,
    BashProcessToolConfig,
)
from agiwo.tool.builtin.bash_tool.tool import BashTool, BashToolConfig
from agiwo.tool.builtin.bash_tool.types import (
    CommandResult,
    ProcessInfo,
    ProcessLogInfo,
    ProcessStatus,
)
from tests.utils.agent_context import build_agent_context


class MockSandbox:
    """Mock sandbox for testing bash tools."""

    def __init__(self):
        self._processes: dict[str, dict] = {}
        self._process_counter = 0
        self.executed_commands: list[str] = []
        self.execute_calls: list[dict[str, object]] = []
        self.started_process_calls: list[dict[str, str | None]] = []
        self.stdin_writes: list[dict[str, str]] = []

    async def execute_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float | None = None,
        use_pty: bool = False,
        pty_cols: int = 120,
        pty_rows: int = 40,
        stdin: str | None = None,
    ) -> CommandResult:
        self.executed_commands.append(command)
        self.execute_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "timeout": timeout,
                "use_pty": use_pty,
                "pty_cols": pty_cols,
                "pty_rows": pty_rows,
                "stdin": stdin,
            }
        )

        if command.startswith("echo "):
            output = command[5:].strip().strip("\"'")
            return CommandResult(exit_code=0, stdout=output + "\n", stderr="")
        if command.startswith("exit "):
            code = int(command[5:])
            return CommandResult(
                exit_code=code, stdout="", stderr=f"error: exit {code}"
            )
        if "tail -n" in command or "grep" in command:
            return CommandResult(
                exit_code=0, stdout="log line 1\nlog line 2\n", stderr=""
            )
        return CommandResult(exit_code=0, stdout="success\n", stderr="")

    async def start_process(
        self,
        command: str,
        cwd: str | None = None,
        env: dict | None = None,
        agent_id: str | None = None,
        use_pty: bool = False,
        pty_cols: int = 120,
        pty_rows: int = 40,
    ) -> str:
        del env
        self._process_counter += 1
        process_id = f"job_{self._process_counter}"
        self.started_process_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "agent_id": agent_id,
                "mode": "pty" if use_pty else "pipe",
                "pty_cols": str(pty_cols),
                "pty_rows": str(pty_rows),
            }
        )
        self._processes[process_id] = {
            "command": command,
            "state": "running",
            "mode": "pty" if use_pty else "pipe",
            "started_at": time.time(),
            "exit_code": None,
            "agent_id": agent_id,
        }
        return process_id

    async def attach_process(self, process_id: str) -> ProcessInfo:
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")
        process = self._processes[process_id]
        return ProcessInfo(
            process_id=process_id,
            command=process["command"],
            state=process["state"],
            mode=process["mode"],
            started_at=process["started_at"],
            exit_code=process["exit_code"],
        )

    async def get_process_status(self, process_id: str) -> ProcessStatus:
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")
        process = self._processes[process_id]
        return ProcessStatus(
            state=process["state"],
            mode=process["mode"],
            started_at=process["started_at"],
            exit_code=process["exit_code"],
        )

    async def stop_process(self, process_id: str, signal: str = "TERM") -> None:
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")
        self._processes[process_id]["state"] = "exited"
        self._processes[process_id]["exit_code"] = -9 if signal == "KILL" else -15

    async def list_processes(self, state: str = "all") -> list[ProcessInfo]:
        results = []
        for process_id, process in self._processes.items():
            if state == "running" and process["state"] != "running":
                continue
            results.append(
                ProcessInfo(
                    process_id=process_id,
                    command=process["command"],
                    state=process["state"],
                    mode=process["mode"],
                    started_at=process["started_at"],
                    exit_code=process["exit_code"],
                )
            )
        return results

    async def list_processes_by_agent(
        self,
        agent_id: str,
        state: str = "all",
    ) -> list[ProcessInfo]:
        results = []
        for process_id, process in self._processes.items():
            if process.get("agent_id") != agent_id:
                continue
            if state == "running" and process["state"] != "running":
                continue
            results.append(
                ProcessInfo(
                    process_id=process_id,
                    command=process["command"],
                    state=process["state"],
                    mode=process["mode"],
                    started_at=process["started_at"],
                    exit_code=process["exit_code"],
                )
            )
        return results

    async def get_process_logs_info(self, process_id: str) -> ProcessLogInfo:
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")
        return ProcessLogInfo(
            stdout_path=f"/tmp/{process_id}.stdout",
            stderr_path=f"/tmp/{process_id}.stderr",
            mode=self._processes[process_id]["mode"],
        )

    async def write_process_stdin(self, process_id: str, data: str) -> None:
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")
        if self._processes[process_id]["mode"] != "pty":
            raise RuntimeError("Process does not support stdin writes (not PTY mode)")
        self.stdin_writes.append({"process_id": process_id, "data": data})


@pytest.fixture
def mock_context():
    return build_agent_context(agent_id="agent_1", agent_name="agent_1")


@pytest.fixture
def mock_sandbox():
    return MockSandbox()


@pytest.fixture
def bash_tool(mock_sandbox):
    return BashTool(
        BashToolConfig(
            sandbox=mock_sandbox,
            cwd="/workspace",
        )
    )


@pytest.fixture
def bash_process_tool(mock_sandbox):
    return BashProcessTool(
        BashProcessToolConfig(
            sandbox=mock_sandbox,
        )
    )


