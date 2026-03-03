"""Type definitions for BashTool."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

if TYPE_CHECKING:
    from agiwo.tool.builtin.bash_tool.security import (
        CommandRiskEvaluator,
        CommandSafetyPolicy,
    )


@dataclass
class CommandResult:
    """Result of a command execution."""

    exit_code: int
    stdout: str
    stderr: str


@dataclass
class WriteFileSpec:
    """Specification for writing a file."""

    path: str
    content: str | bytes


@dataclass
class ProcessInfo:
    """Information about a running or completed process."""

    process_id: str
    command: str
    state: Literal["running", "exited", "unknown"]
    started_at: float | None = None
    exit_code: int | None = None


@dataclass
class ProcessStatus:
    """Status of a process."""

    state: Literal["running", "exited", "unknown"]
    started_at: float | None = None
    exit_code: int | None = None


@dataclass
class ProcessLogInfo:
    """Log file paths for a process."""

    stdout_path: str
    stderr_path: str


class Sandbox(Protocol):
    """Abstract protocol for sandbox implementations."""

    def execute_command(
        self, command: str, cwd: str | None = None, timeout: float | None = None
    ) -> CommandResult:
        """
        Execute a command in the sandbox.

        Args:
            command: The command to execute.
            cwd: The current working directory.
            timeout: The timeout in seconds.

        Returns:
            CommandResult: The result of the command execution.

        Raises:
            TimeoutError: If command execution exceeds timeout.
            Exception: If the command execution fails.
        """
        ...

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox.

        Args:
            path: Relative path to the file within the sandbox.

        Returns:
            File content as string.
        """
        ...

    def write_file(self, files: list[WriteFileSpec]) -> None:
        """Write multiple files to the sandbox.

        Args:
            files: List of file specifications to write.
        """
        ...

    async def start_process(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Start a long-running process."""
        ...

    async def attach_process(self, process_id: str) -> ProcessInfo:
        """Attach to an existing process."""
        ...

    async def get_process_status(self, process_id: str) -> ProcessStatus:
        """Get the status of a process."""
        ...

    async def stop_process(self, process_id: str, signal: str = "TERM") -> None:
        """Stop a running process."""
        ...

    async def list_processes(
        self,
        state: Literal["running", "all"] = "all",
    ) -> list[ProcessInfo]:
        """List processes managed by this sandbox."""
        ...

    async def get_process_logs_info(self, process_id: str) -> ProcessLogInfo:
        """Get log file paths for a process."""
        ...


@dataclass
class BeforeBashCallInput:
    """Input for on_before_bash_call callback."""

    command: str


@dataclass
class BeforeBashCallOutput:
    """Output from on_before_bash_call callback."""

    command: str


@dataclass
class AfterBashCallInput:
    """Input for on_after_bash_call callback."""

    command: str
    result: CommandResult


@dataclass
class AfterBashCallOutput:
    """Output from on_after_bash_call callback."""

    result: CommandResult


@dataclass
class CreateBashToolkitOptions:
    """Options for creating a BashToolkit."""

    destination: str | None = None
    files: dict[str, str | bytes] | None = None
    sandbox: Sandbox | None = None
    extra_instructions: str | None = None
    on_before_bash_call: (
        Callable[[BeforeBashCallInput], BeforeBashCallOutput | None] | None
    ) = None
    on_after_bash_call: (
        Callable[[AfterBashCallInput], AfterBashCallOutput | None] | None
    ) = None
    command_risk_evaluator: CommandRiskEvaluator | None = None
    command_safety_policy: CommandSafetyPolicy | None = None
    max_output_length: int = 30000
    max_files: int = 1000
    max_processes: int = 10


class BashToolkit:
    """Container for bash tools and sandbox."""

    def __init__(
        self,
        tools: dict[str, Any],
        sandbox: Sandbox,
    ) -> None:
        self.tools = tools
        self.sandbox = sandbox

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to tools."""
        return self.tools[key]
