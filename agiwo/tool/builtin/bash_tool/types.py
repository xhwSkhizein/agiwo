"""Type definitions for BashTool."""

from dataclasses import dataclass
from typing import Literal, Protocol


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
    mode: Literal["pipe", "pty"] = "pipe"
    started_at: float | None = None
    exit_code: int | None = None


@dataclass
class ProcessStatus:
    """Status of a process."""

    state: Literal["running", "exited", "unknown"]
    mode: Literal["pipe", "pty"] = "pipe"
    started_at: float | None = None
    exit_code: int | None = None


@dataclass
class ProcessLogInfo:
    """Log file paths for a process."""

    stdout_path: str
    stderr_path: str
    mode: Literal["pipe", "pty"] = "pipe"


class Sandbox(Protocol):
    """Abstract protocol for sandbox implementations."""

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

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox.

        Args:
            path: Relative path to the file within the sandbox.

        Returns:
            File content as string.
        """
        ...

    async def write_files(self, files: list[WriteFileSpec]) -> None:
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
        agent_id: str | None = None,
        use_pty: bool = False,
        pty_cols: int = 120,
        pty_rows: int = 40,
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

    async def write_process_stdin(
        self,
        process_id: str,
        data: str,
    ) -> None:
        """Write data to a process stdin (PTY mode)."""
        ...

    async def list_processes(
        self,
        state: Literal["running", "all"] = "all",
    ) -> list[ProcessInfo]:
        """List processes managed by this sandbox."""
        ...

    async def list_processes_by_agent(
        self,
        agent_id: str,
        state: Literal["running", "all"] = "all",
    ) -> list[ProcessInfo]:
        """List processes associated with a specific agent_id."""
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
