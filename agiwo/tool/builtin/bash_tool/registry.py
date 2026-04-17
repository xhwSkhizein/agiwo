"""Process registry for managing long-running processes.

Ownership model:
    Records carry ``agent_id`` from ``start_process``. All per-process
    operations accept a keyword-only ``owner_agent_id`` and raise ``KeyError``
    (indistinguishable from "not found") when the caller does not own the
    record. ``owner_agent_id=None`` bypasses the check (admin/CLI path).
"""

import errno
import json
import os
import pty
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Literal

from .pty_utils import set_pty_size
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
    agent_id: str | None = None
    pid_start_time: str | None = None
    mode: Literal["pipe", "pty"] = "pipe"


class ProcessRegistry:
    """Registry for managing long-running processes."""

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._registry_file = logs_dir / "registry.json"
        self._processes: dict[str, ProcessRecord] = {}
        self._process_refs: dict[str, subprocess.Popen[Any]] = {}
        self._file_handles: dict[str, tuple[IO[str], IO[str]]] = {}
        self._pty_master_fds: dict[str, int] = {}
        self._pty_reader_threads: dict[str, threading.Thread] = {}
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
        agent_id: str | None = None,
        use_pty: bool = False,
        pty_cols: int = 120,
        pty_rows: int = 40,
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
        if use_pty:
            return self._start_pty_process(
                process_id=process_id,
                command=command,
                cwd=cwd,
                env=env,
                agent_id=agent_id,
                pty_cols=pty_cols,
                pty_rows=pty_rows,
            )
        return self._start_pipe_process(
            process_id=process_id,
            command=command,
            cwd=cwd,
            env=env,
            agent_id=agent_id,
        )

    def _start_pipe_process(
        self,
        process_id: str,
        command: str,
        cwd: str | None,
        env: dict[str, str] | None,
        agent_id: str | None,
    ) -> str:
        stdout_path = self.logs_dir / f"{process_id}.stdout"
        stderr_path = self.logs_dir / f"{process_id}.stderr"
        stdout_fp = open(stdout_path, "w")
        stderr_fp = open(stderr_path, "w")

        process_env = {**os.environ, **env} if env else None
        # start_new_session=True makes the shell a session/process-group leader
        # with pgid == pid, so stop_process can signal the whole group and avoid
        # leaking reparented descendants (e.g. python -c 'subprocess.Popen(...)').
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=stdout_fp,
            stderr=stderr_fp,
            cwd=cwd,
            env=process_env,
            start_new_session=True,
        )

        record = ProcessRecord(
            process_id=process_id,
            command=command,
            cwd=cwd,
            pid=process.pid,
            started_at=datetime.now().timestamp(),
            state="running",
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            agent_id=agent_id,
            pid_start_time=self._get_pid_start_time(process.pid),
            mode="pipe",
        )

        self._processes[process_id] = record
        self._process_refs[process_id] = process
        self._file_handles[process_id] = (stdout_fp, stderr_fp)
        self._save_registry()
        return process_id

    @staticmethod
    def _cleanup_pty_resources(
        stdout_fp: IO[str],
        stderr_fp: IO[str],
        master_fd: int | None,
        slave_fd: int | None,
    ) -> None:
        for fp in (stdout_fp, stderr_fp):
            try:
                fp.close()
            except Exception:  # noqa: BLE001
                pass
        for fd in (master_fd, slave_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _start_pty_process(
        self,
        process_id: str,
        command: str,
        cwd: str | None,
        env: dict[str, str] | None,
        agent_id: str | None,
        pty_cols: int,
        pty_rows: int,
    ) -> str:
        stdout_path = self.logs_dir / f"{process_id}.stdout"
        stderr_path = self.logs_dir / f"{process_id}.stderr"
        master_fd: int | None = None
        slave_fd: int | None = None
        stdout_fp = open(stdout_path, "w", encoding="utf-8")
        try:
            stderr_fp = open(stderr_path, "w", encoding="utf-8")
        except Exception:
            stdout_fp.close()
            raise
        try:
            master_fd, slave_fd = pty.openpty()
            set_pty_size(slave_fd, cols=pty_cols, rows=pty_rows)

            process_env = {**os.environ, **env} if env else None
            process = subprocess.Popen(
                command,
                shell=True,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=cwd,
                env=process_env,
                start_new_session=True,
                close_fds=True,
            )
        except Exception:  # noqa: BLE001
            self._cleanup_pty_resources(stdout_fp, stderr_fp, master_fd, slave_fd)
            raise

        if slave_fd is not None:
            os.close(slave_fd)
        if master_fd is None:
            raise RuntimeError("Failed to allocate PTY master fd")

        reader = threading.Thread(
            target=self._pty_reader_loop,
            args=(master_fd, stdout_fp),
            daemon=True,
        )
        reader.start()

        record = ProcessRecord(
            process_id=process_id,
            command=command,
            cwd=cwd,
            pid=process.pid,
            started_at=datetime.now().timestamp(),
            state="running",
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            agent_id=agent_id,
            pid_start_time=self._get_pid_start_time(process.pid),
            mode="pty",
        )

        self._processes[process_id] = record
        self._process_refs[process_id] = process
        self._file_handles[process_id] = (stdout_fp, stderr_fp)
        self._pty_master_fds[process_id] = master_fd
        self._pty_reader_threads[process_id] = reader
        self._save_registry()
        return process_id

    def attach_process(
        self,
        process_id: str,
        *,
        owner_agent_id: str | None = None,
    ) -> ProcessInfo:
        """Attach to an existing process.

        Args:
            process_id: Process ID to attach to.
            owner_agent_id: Caller's agent id. ``None`` bypasses the owner
                check (admin/CLI); otherwise the record's ``agent_id`` must
                match.

        Returns:
            ProcessInfo with process metadata.

        Raises:
            KeyError: If the process does not exist or is not owned by the
                caller (same exception to avoid id enumeration leakage).
        """
        record = self._require_record(process_id, owner_agent_id)
        self._update_process_status(record)

        return ProcessInfo(
            process_id=record.process_id,
            state=record.state,  # type: ignore[arg-type]
            command=record.command,
            mode=record.mode,
            started_at=record.started_at,
            exit_code=record.exit_code,
        )

    def get_process_status(
        self,
        process_id: str,
        *,
        owner_agent_id: str | None = None,
    ) -> ProcessStatus:
        """Get status of a process.

        Raises:
            KeyError: If the process does not exist or is not owned by the
                caller.
        """
        record = self._require_record(process_id, owner_agent_id)
        self._update_process_status(record)

        return ProcessStatus(
            state=record.state,  # type: ignore[arg-type]
            mode=record.mode,
            exit_code=record.exit_code,
            started_at=record.started_at,
        )

    def stop_process(
        self,
        process_id: str,
        signal_name: Literal["TERM", "KILL"],
        *,
        owner_agent_id: str | None = None,
    ) -> None:
        """Stop a running process (and its whole process group).

        The pipe/pty launchers use ``start_new_session=True``, so the record's
        ``pid`` is the process-group leader. This method signals the entire
        group via :func:`os.killpg` and waits for the whole process group to
        disappear before committing ``state = "exited"``. TERM escalates to
        KILL after a grace period if any descendant refuses to die.

        Raises:
            KeyError: If the process does not exist or is not owned by the
                caller.
        """
        record = self._require_record(process_id, owner_agent_id)
        self._update_process_status(record)

        if record.state != "running":
            return

        sig = signal.SIGTERM if signal_name == "TERM" else signal.SIGKILL
        process = self._process_refs.get(process_id)

        # Without an in-memory handle, only trust the pid if start-time still
        # matches. Otherwise another process may have reclaimed the pid.
        if process is None and not self._is_same_process(record):
            self._finalize_exited(record, process=None)
            return

        self._kill_process_group(record, sig)
        group_exited = self._wait_for_process_group_exit(record, process)
        if not group_exited and signal_name == "TERM":
            self._kill_process_group(record, signal.SIGKILL)
            group_exited = self._wait_for_process_group_exit(record, process)
        if not group_exited:
            raise RuntimeError(f"Process group did not exit: {process_id}")
        if process is not None:
            process.poll()
        self._finalize_exited(record, process)

    def write_process_stdin(
        self,
        process_id: str,
        data: str,
        *,
        owner_agent_id: str | None = None,
    ) -> None:
        """Write stdin to a running PTY process."""
        record = self._require_record(process_id, owner_agent_id)
        self._update_process_status(record)
        if record.state != "running":
            raise RuntimeError(f"Process is not running: {process_id}")
        if record.mode != "pty":
            raise RuntimeError(
                f"Process does not support stdin writes (not PTY mode): {process_id}"
            )

        master_fd = self._pty_master_fds.get(process_id)
        if master_fd is None:
            raise RuntimeError(f"PTY handle is not available: {process_id}")
        os.write(master_fd, data.encode("utf-8"))

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
                        mode=record.mode,
                        started_at=record.started_at,
                        exit_code=record.exit_code,
                    )
                )

        return results

    def list_processes_by_agent(
        self,
        agent_id: str,
        state: Literal["all", "running"] = "all",
    ) -> list[ProcessInfo]:
        """List processes associated with a specific agent_id."""
        results = []
        for record in self._processes.values():
            if record.agent_id != agent_id:
                continue
            self._update_process_status(record)
            if state == "all" or record.state == state:
                results.append(
                    ProcessInfo(
                        process_id=record.process_id,
                        state=record.state,  # type: ignore[arg-type]
                        command=record.command,
                        mode=record.mode,
                        started_at=record.started_at,
                        exit_code=record.exit_code,
                    )
                )
        return results

    def get_process_logs_info(
        self,
        process_id: str,
        *,
        owner_agent_id: str | None = None,
    ) -> ProcessLogInfo:
        """Get log file paths for a process.

        Raises:
            KeyError: If the process does not exist or is not owned by the
                caller.
        """
        record = self._require_record(process_id, owner_agent_id)
        return ProcessLogInfo(
            stdout_path=record.stdout_path,
            stderr_path=record.stderr_path,
            mode=record.mode,
        )

    def _require_record(
        self,
        process_id: str,
        owner_agent_id: str | None,
    ) -> ProcessRecord:
        """Look up a record and enforce the owner boundary.

        ``owner_agent_id is None`` means the caller has no agent identity
        (CLI, tests, admin); owner is not enforced in that case. Otherwise the
        record's ``agent_id`` must match exactly; mismatch raises the same
        ``KeyError`` as 'not found' to avoid leaking existence.
        """
        record = self._processes.get(process_id)
        if record is None:
            raise KeyError(f"Process not found: {process_id}")
        if owner_agent_id is not None and record.agent_id != owner_agent_id:
            raise KeyError(f"Process not found: {process_id}")
        return record

    def _kill_process_group(self, record: ProcessRecord, sig: int) -> None:
        """Signal the record's process group; fall back to pid on OSError.

        With ``start_new_session=True`` the leader's pgid equals its pid.
        ``ProcessLookupError`` is treated as already-dead (idempotent).
        """
        try:
            os.killpg(record.pid, sig)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
        try:
            os.kill(record.pid, sig)
        except OSError:
            pass

    @staticmethod
    def _process_group_alive(pgid: int) -> bool:
        """Return whether a process group still has live members."""
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return True
        return True

    def _wait_for_process_group_exit(
        self,
        record: ProcessRecord,
        process: subprocess.Popen[Any] | None,
        *,
        timeout: float = 3.0,
    ) -> bool:
        """Block until the whole process group exits or ``timeout`` elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process is not None:
                process.poll()
            if not self._process_group_alive(record.pid):
                return True
            time.sleep(0.05)
        if process is not None:
            process.poll()
        return not self._process_group_alive(record.pid)

    def _finalize_exited(
        self,
        record: ProcessRecord,
        process: subprocess.Popen[Any] | None,
    ) -> None:
        """Commit the exited state with a best-effort real exit code."""
        record.state = "exited"
        if process is not None and process.returncode is not None:
            record.exit_code = process.returncode
        else:
            record.exit_code = -1
        self._close_file_handles(record.process_id)
        self._close_pty_handle(record.process_id)
        self._save_registry()

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
                    self._close_pty_handle(record.process_id)
                    self._save_registry()
                    return

            # Without an in-memory handle (e.g. after restart), guard against PID
            # reuse before trusting the pid.
            if not self._is_same_process(record):
                record.state = "exited"
                if process is not None:
                    ret = process.poll()
                    record.exit_code = ret if ret is not None else -1
                else:
                    record.exit_code = -1
                self._close_file_handles(record.process_id)
                self._close_pty_handle(record.process_id)
                self._save_registry()

    def _close_file_handles(self, process_id: str) -> None:
        """Close file handles for a process."""
        handles = self._file_handles.pop(process_id, None)
        if handles:
            for fp in handles:
                try:
                    fp.close()
                except Exception:  # noqa: BLE001
                    pass  # Ignore close errors

    def _close_pty_handle(self, process_id: str) -> None:
        master_fd = self._pty_master_fds.pop(process_id, None)
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass
        self._pty_reader_threads.pop(process_id, None)

    @staticmethod
    def _pty_reader_loop(master_fd: int, stdout_fp: IO[str]) -> None:
        while True:
            try:
                chunk = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                break

            if not chunk:
                break

            text = chunk.decode("utf-8", errors="replace")
            try:
                stdout_fp.write(text)
                stdout_fp.flush()
            except Exception:  # noqa: BLE001
                break

    def _is_same_process(self, record: ProcessRecord) -> bool:
        current_start = self._get_pid_start_time(record.pid)
        if current_start is None:
            return False
        if record.pid_start_time is None:
            return False
        return current_start == record.pid_start_time

    @staticmethod
    def _get_pid_start_time(pid: int) -> str | None:
        try:
            output = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "lstart="],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
            return None

        value = output.strip()
        return value if value else None
