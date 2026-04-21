"""Process registry for managing long-running processes."""

import errno
import json
import os
import pty
import signal
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Any, Literal, TextIO

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
        # Protects all mutable dicts below plus the JSON-on-disk view. RLock
        # allows nested acquisition by helpers that are called both directly
        # and from already-locked sections.
        self._lock = threading.RLock()
        self._processes: dict[str, ProcessRecord] = {}
        self._process_refs: dict[str, subprocess.Popen[Any]] = {}
        self._file_handles: dict[str, tuple[IO[str], IO[str]]] = {}
        self._pty_master_fds: dict[str, int] = {}
        self._pty_reader_threads: dict[str, threading.Thread] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load process registry from disk."""
        if not self._registry_file.exists():
            return
        try:
            data = json.loads(self._registry_file.read_text())
        except (json.JSONDecodeError, OSError):
            with self._lock:
                self._processes = {}
            return
        allowed_fields = ProcessRecord.__dataclass_fields__.keys()
        with self._lock:
            try:
                for proc_id, record in data.items():
                    filtered = {k: v for k, v in record.items() if k in allowed_fields}
                    self._processes[proc_id] = ProcessRecord(**filtered)
            except TypeError:
                self._processes = {}

    def _save_registry(self) -> None:
        """Save process registry to disk atomically (tmp + os.replace).

        Caller must hold ``self._lock`` to avoid interleaving with concurrent
        mutations.
        """
        data = {proc_id: asdict(proc) for proc_id, proc in self._processes.items()}
        payload = json.dumps(data, indent=2)
        tmp_path = self._registry_file.with_name(
            self._registry_file.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            tmp_path.write_text(payload)
            os.replace(tmp_path, self._registry_file)
        except OSError:
            # Best-effort cleanup; never leave orphan tmp files on failure.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

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
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=stdout_fp,
            stderr=stderr_fp,
            cwd=cwd,
            env=process_env,
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

        with self._lock:
            self._processes[process_id] = record
            self._process_refs[process_id] = process
            self._file_handles[process_id] = (stdout_fp, stderr_fp)
            self._save_registry()
        return process_id

    @staticmethod
    def _cleanup_pty_resources(
        stdout_fp: "TextIO | None",
        stderr_fp: "TextIO | None",
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

        with self._lock:
            self._processes[process_id] = record
            self._process_refs[process_id] = process
            self._file_handles[process_id] = (stdout_fp, stderr_fp)
            self._pty_master_fds[process_id] = master_fd
            self._pty_reader_threads[process_id] = reader
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
        with self._lock:
            if process_id not in self._processes:
                raise KeyError(f"Process not found: {process_id}")
            record = self._processes[process_id]
            self._update_process_status(record)
            return ProcessInfo(
                process_id=record.process_id,
                state=record.state,  # type: ignore[arg-type]
                command=record.command,
                mode=record.mode,
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
        with self._lock:
            if process_id not in self._processes:
                raise KeyError(f"Process not found: {process_id}")
            record = self._processes[process_id]
            self._update_process_status(record)
            return ProcessStatus(
                state=record.state,  # type: ignore[arg-type]
                mode=record.mode,
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
        with self._lock:
            if process_id not in self._processes:
                raise KeyError(f"Process not found: {process_id}")
            record = self._processes[process_id]
            self._update_process_status(record)

            if record.state != "running":
                return

            sig = signal.SIGTERM if signal_name == "TERM" else signal.SIGKILL
            process = self._process_refs.get(process_id)

            # Prefer stopping through in-memory Popen handle. Only fall back to
            # raw PID if it is still the same OS process we originally started.
            if process is None and not self._is_same_process(record):
                self._finalize_exited(record, exit_code=-1)
                return

            try:
                if process is not None:
                    process.send_signal(sig)
                else:
                    os.kill(record.pid, sig)
            except OSError:
                pass
            self._finalize_exited(record, exit_code=-1)

    def write_process_stdin(self, process_id: str, data: str) -> None:
        """Write stdin to a running PTY process."""
        with self._lock:
            if process_id not in self._processes:
                raise KeyError(f"Process not found: {process_id}")

            record = self._processes[process_id]
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
        # Release the lock before performing the blocking OS write so concurrent
        # readers can continue inspecting registry state.
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
        with self._lock:
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
        with self._lock:
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

    def get_process_logs_info(self, process_id: str) -> ProcessLogInfo:
        """Get log file paths for a process.

        Args:
            process_id: Process ID to query.

        Returns:
            ProcessLogInfo with log file paths.

        Raises:
            KeyError: If process not found.
        """
        with self._lock:
            if process_id not in self._processes:
                raise KeyError(f"Process not found: {process_id}")
            record = self._processes[process_id]
            return ProcessLogInfo(
                stdout_path=record.stdout_path,
                stderr_path=record.stderr_path,
                mode=record.mode,
            )

    def _update_process_status(self, record: ProcessRecord) -> None:
        """Update process status by checking if still running.

        Caller must hold ``self._lock``.
        """
        if record.state != "running":
            return

        process = self._process_refs.get(record.process_id)
        if process is not None:
            ret = process.poll()
            if ret is not None:
                self._finalize_exited(record, exit_code=ret)
                return

        # Without an in-memory handle (e.g. after restart), guard against PID
        # reuse before trusting the pid.
        if not self._is_same_process(record):
            exit_code = -1
            if process is not None:
                ret = process.poll()
                if ret is not None:
                    exit_code = ret
            self._finalize_exited(record, exit_code=exit_code)

    def _finalize_exited(self, record: ProcessRecord, *, exit_code: int) -> None:
        """Mark a record as exited and release its OS resources in a safe order.

        Closing the PTY master first forces the reader thread to wake with EIO
        and exit; joining the reader thread guarantees we do not close the sink
        stdout fp while it is mid-write. Caller must hold ``self._lock``.
        """
        record.state = "exited"
        record.exit_code = exit_code
        self._close_pty_handle(record.process_id)
        self._close_file_handles(record.process_id)
        self._save_registry()

    def _close_file_handles(self, process_id: str) -> None:
        """Close file handles for a process. Caller must hold ``self._lock``."""
        handles = self._file_handles.pop(process_id, None)
        if handles:
            for fp in handles:
                try:
                    fp.close()
                except Exception:  # noqa: BLE001
                    pass  # Ignore close errors

    def _close_pty_handle(self, process_id: str) -> None:
        """Close the PTY master fd and wait for its reader thread to exit.

        Must be called before ``_close_file_handles`` so the reader thread
        cannot race with closing its sink stdout fp. Caller must hold
        ``self._lock``.
        """
        master_fd = self._pty_master_fds.pop(process_id, None)
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass
        reader = self._pty_reader_threads.pop(process_id, None)
        if reader is not None and reader.is_alive():
            reader.join(timeout=1.0)

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
