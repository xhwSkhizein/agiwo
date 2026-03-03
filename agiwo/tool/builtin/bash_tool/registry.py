"""Process registry for managing long-running processes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from datetime import datetime
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal

from .types import ProcessInfo, ProcessLogInfo, ProcessStatus


@dataclass
class ProcessRecord:
    """Record of a managed process."""

    process_id: str
    command: str
    cwd: str | None
    pid: int
    started_at: float
    state: Literal["running", "exited"]
    exit_code: int | None = None
    stdout_path: str = ""
    stderr_path: str = ""


class ProcessRegistry:
    """Registry for managing long-running processes."""

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._registry_file = logs_dir / "registry.json"
        self._processes: dict[str, ProcessRecord] = {}
        self._process_refs: dict[str, subprocess.Popen[Any]] = {}
        self._file_handles: dict[str, tuple[IO[str], IO[str]]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load process registry from disk."""
        if self._registry_file.exists():
            try:
                data = json.loads(self._registry_file.read_text())
                allowed_fields = ProcessRecord.__dataclass_fields__.keys()
                for proc_id, record in data.items():
                    filtered = {k: v for k, v in record.items() if k in allowed_fields}
                    self._processes[proc_id] = ProcessRecord(**filtered)
            except (json.JSONDecodeError, TypeError):
                self._processes = {}

    def _save_registry(self) -> None:
        """Save process registry to disk."""
        data = {proc_id: asdict(proc) for proc_id, proc in self._processes.items()}
        self._registry_file.write_text(json.dumps(data, indent=2))

    def start_process(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Start a new process.

        Args:
            command: Command to execute.
            cwd: Working directory.
            env: Environment variables.

        Returns:
            process_id: Unique identifier for the process.
        """
        process_id = str(uuid.uuid4())

        # Create log files
        stdout_path = self.logs_dir / f"{process_id}.stdout"
        stderr_path = self.logs_dir / f"{process_id}.stderr"
        # Open log files - must keep them open while process is running
        stdout_fp = open(stdout_path, "w")
        stderr_fp = open(stderr_path, "w")

        # Start process
        process_env = {**os.environ, **env} if env else None
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=stdout_fp,
            stderr=stderr_fp,
            cwd=cwd,
            env=process_env,
        )

        # Create record
        record = ProcessRecord(
            process_id=process_id,
            command=command,
            cwd=cwd,
            pid=process.pid,
            started_at=datetime.now().timestamp(),
            state="running",
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

        # Add to registry
        self._processes[process_id] = record
        self._process_refs[process_id] = process  # Store Popen reference separately
        self._file_handles[process_id] = (
            stdout_fp,
            stderr_fp,
        )  # Keep file handles open
        self._save_registry()

        return process_id

    def attach_process(self, process_id: str) -> ProcessInfo:
        """Attach to an existing process.

        Args:
            process_id: Process ID to attach to.

        Returns:
            ProcessInfo with process metadata.

        Raises:
            KeyError: If process not found.
        """
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")

        record = self._processes[process_id]
        self._update_process_status(record)

        return ProcessInfo(
            process_id=record.process_id,
            state=record.state,  # type: ignore[arg-type]
            command=record.command,
            started_at=record.started_at,
            exit_code=record.exit_code,
        )

    def get_process_status(self, process_id: str) -> ProcessStatus:
        """Get status of a process.

        Args:
            process_id: Process ID to query.

        Returns:
            ProcessStatus with current state.

        Raises:
            KeyError: If process not found.
        """
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")

        record = self._processes[process_id]
        self._update_process_status(record)

        return ProcessStatus(
            state=record.state,  # type: ignore[arg-type]
            exit_code=record.exit_code,
            started_at=record.started_at,
        )

    def stop_process(
        self, process_id: str, signal_name: Literal["TERM", "KILL"]
    ) -> None:
        """Stop a running process.

        Args:
            process_id: Process ID to stop.
            signal_name: Signal to send ("TERM" or "KILL").

        Raises:
            KeyError: If process not found.
        """
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")

        record = self._processes[process_id]
        self._update_process_status(record)

        if record.state == "running":
            sig = signal.SIGTERM if signal_name == "TERM" else signal.SIGKILL
            try:
                os.kill(record.pid, sig)
                record.state = "exited"
                record.exit_code = -1
                self._close_file_handles(process_id)
                self._save_registry()
            except OSError:
                pass  # Process already gone

    def list_processes(
        self,
        state: Literal["all", "running"] = "all",
    ) -> list[ProcessInfo]:
        """List managed processes.

        Args:
            state: Filter by state ("running" or "all").

        Returns:
            List of ProcessInfo objects.
        """
        results = []
        for record in self._processes.values():
            self._update_process_status(record)

            if state == "all" or record.state == state:
                results.append(
                    ProcessInfo(
                        process_id=record.process_id,
                        state=record.state,  # type: ignore[arg-type]
                        command=record.command,
                        started_at=record.started_at,
                        exit_code=record.exit_code,
                    )
                )

        return results

    def get_process_logs_info(self, process_id: str) -> ProcessLogInfo:
        """Get log file paths for a process.

        Args:
            process_id: Process ID to query.

        Returns:
            ProcessLogInfo with log file paths.

        Raises:
            KeyError: If process not found.
        """
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")

        record = self._processes[process_id]
        return ProcessLogInfo(
            stdout_path=record.stdout_path,
            stderr_path=record.stderr_path,
        )

    def _update_process_status(self, record: ProcessRecord) -> None:
        """Update process status by checking if still running."""
        if record.state == "running":
            process = self._process_refs.get(record.process_id)
            if process is not None:
                # Check if process has finished
                ret = process.poll()
                if ret is not None:
                    # Process finished, get exit code and close file handles
                    record.state = "exited"
                    record.exit_code = ret
                    self._close_file_handles(record.process_id)
                    self._save_registry()
                    return
            # Fallback to os.kill check
            try:
                os.kill(record.pid, 0)
            except OSError:
                # Process no longer exists
                record.state = "exited"
                if process is not None:
                    ret = process.poll()
                    record.exit_code = ret if ret is not None else -1
                else:
                    record.exit_code = -1
                self._close_file_handles(record.process_id)
                self._save_registry()

    def _close_file_handles(self, process_id: str) -> None:
        """Close file handles for a process."""
        handles = self._file_handles.pop(process_id, None)
        if handles:
            for fp in handles:
                try:
                    fp.close()
                except Exception:
                    pass  # Ignore close errors
