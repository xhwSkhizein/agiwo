"""
Local executor for bash tool
"""

import asyncio
import errno
import os
import tempfile
from pathlib import Path
from typing import Literal

from agiwo.tool.builtin.bash_tool.pty_utils import set_pty_size
from agiwo.tool.builtin.bash_tool.registry import ProcessRegistry
from agiwo.tool.builtin.bash_tool.types import (
    CommandResult,
    ProcessInfo,
    ProcessLogInfo,
    ProcessStatus,
    Sandbox,
    WriteFileSpec,
)
from agiwo.utils.abort_signal import AbortSignal


class LocalExecutor(Sandbox):
    def __init__(
        self,
        workspace_dir: str | None = None,
        max_processes: int = 10,
    ) -> None:
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if workspace_dir:
            self.workspace = Path(workspace_dir).resolve()
            self.workspace.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="bash_tool_")
            self.workspace = Path(self._temp_dir.name).resolve()

        self.max_processes = max_processes
        logs_dir = self.workspace / ".bash_tool" / "logs"
        self._registry = ProcessRegistry(logs_dir)
        # Serializes start_process so the `running < max_processes` check and
        # the subsequent Popen form one atomic step. Without this, parallel
        # bash(background=true) tool calls can both observe count<max and both
        # spawn, turning max_processes into a soft budget rather than a cap.
        self._start_lock = asyncio.Lock()

    async def _graceful_terminate_process(
        self, process: asyncio.subprocess.Process
    ) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            process.kill()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    @property
    def workspace_path(self) -> Path:
        return self.workspace

    async def _cancel_gather_tasks(self, pending: set[asyncio.Task[object]]) -> None:
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _pipe_communicate_with_abort(
        self,
        process: asyncio.subprocess.Process,
        command: str,
        timeout: float | None,
        abort_signal: AbortSignal | None,
    ) -> CommandResult | tuple[bytes, bytes]:
        if abort_signal is None:
            if timeout is None:
                return await process.communicate()
            return await asyncio.wait_for(process.communicate(), timeout=timeout)
        communicate_task = asyncio.create_task(process.communicate())
        waiters: set[asyncio.Task[object]] = {communicate_task}
        abort_task = asyncio.create_task(abort_signal.wait())
        waiters.add(abort_task)
        if timeout is not None:
            waiters.add(asyncio.create_task(asyncio.sleep(timeout)))
        done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        if communicate_task in done:
            await self._cancel_gather_tasks(pending)
            return communicate_task.result()
        if abort_task in done:
            communicate_task.cancel()
            try:
                await communicate_task
            except asyncio.CancelledError:
                pass
            await self._cancel_gather_tasks(pending)
            await self._graceful_terminate_process(process)
            reason = abort_signal.reason or "Operation cancelled"
            return CommandResult(stdout="", stderr=reason, exit_code=1)
        communicate_task.cancel()
        try:
            await communicate_task
        except asyncio.CancelledError:
            pass
        await self._cancel_gather_tasks(pending)
        process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        raise TimeoutError(
            f"Command timed out after {timeout} seconds: {command}"
        ) from None

    async def execute_command(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        use_pty: bool = False,
        pty_cols: int = 120,
        pty_rows: int = 40,
        stdin: str | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> CommandResult:
        working_dir = self.workspace if not cwd else self.workspace / cwd
        working_dir = working_dir.resolve()
        try:
            working_dir.relative_to(self.workspace)
        except ValueError:
            working_dir = self.workspace

        if use_pty:
            return await self._execute_command_with_pty(
                command=command,
                working_dir=working_dir,
                env=env,
                timeout=timeout,
                pty_cols=pty_cols,
                pty_rows=pty_rows,
                stdin=stdin,
                abort_signal=abort_signal,
            )

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(working_dir),
                env=self._build_env(env),
            )
            try:
                outcome = await self._pipe_communicate_with_abort(
                    process, command, timeout, abort_signal
                )
                if isinstance(outcome, CommandResult):
                    return outcome
                stdout_bytes, stderr_bytes = outcome
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                raise TimeoutError(
                    f"Command timed out after {timeout} seconds: {command}"
                ) from None

            return CommandResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                exit_code=process.returncode or 0,
            )
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001 - executor execution boundary
            return CommandResult(
                stdout="",
                stderr=f"Failed to execute command: {exc}",
                exit_code=1,
            )

    async def _execute_command_with_pty(  # noqa: C901, PLR0912, PLR0915 - PTY lifecycle boundary
        self,
        command: str,
        working_dir: Path,
        env: dict[str, str] | None,
        timeout: float | None,
        pty_cols: int,
        pty_rows: int,
        stdin: str | None,
        abort_signal: AbortSignal | None = None,
    ) -> CommandResult:
        try:
            master_fd, slave_fd = os.openpty()
        except OSError as exc:
            return CommandResult(
                stdout="",
                stderr=f"Failed to allocate PTY: {exc}",
                exit_code=1,
            )

        set_pty_size(slave_fd, cols=pty_cols, rows=pty_rows)
        os.set_blocking(master_fd, False)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=str(working_dir),
                start_new_session=True,
                env=self._build_env(env),
            )
        except Exception as exc:  # noqa: BLE001 - executor PTY start boundary
            os.close(master_fd)
            os.close(slave_fd)
            return CommandResult(
                stdout="",
                stderr=f"Failed to execute PTY command: {exc}",
                exit_code=1,
            )
        finally:
            try:
                os.close(slave_fd)
            except OSError:
                pass

        if stdin:
            try:
                os.write(master_fd, stdin.encode("utf-8"))
            except OSError:
                pass

        chunks: list[bytes] = []
        loop = asyncio.get_running_loop()
        process_done = asyncio.Event()

        def on_readable() -> None:
            try:
                chunk = os.read(master_fd, 4096)
            except BlockingIOError:
                return
            except OSError as exc:
                if exc.errno == errno.EIO:
                    chunk = b""
                else:
                    loop.remove_reader(master_fd)
                    process_done.set()
                    raise
            if not chunk:
                loop.remove_reader(master_fd)
                process_done.set()
                return
            chunks.append(chunk)

        loop.add_reader(master_fd, on_readable)

        waiters: set[asyncio.Task[object]] = {asyncio.create_task(process.wait())}
        abort_task: asyncio.Task[None] | None = None
        if abort_signal is not None:
            abort_task = asyncio.create_task(abort_signal.wait())
            waiters.add(abort_task)
        timeout_task: asyncio.Task[None] | None = None
        if timeout is not None:
            timeout_task = asyncio.create_task(asyncio.sleep(timeout))
            waiters.add(timeout_task)

        try:
            done, pending = await asyncio.wait(
                waiters,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

            if abort_signal is not None and abort_task in done:
                await self._graceful_terminate_process(process)
                for _ in range(3):
                    try:
                        chunk = os.read(master_fd, 4096)
                    except (BlockingIOError, OSError):
                        break
                    if not chunk:
                        break
                    chunks.append(chunk)
                reason = abort_signal.reason or "Operation cancelled"
                return CommandResult(
                    stdout=b"".join(chunks).decode("utf-8", errors="replace"),
                    stderr=reason,
                    exit_code=1,
                )

            if timeout_task is not None and timeout_task in done:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                raise TimeoutError(
                    f"Command timed out after {timeout} seconds: {command}"
                )

            if (
                abort_signal is not None
                and abort_task is not None
                and not abort_task.done()
            ):
                abort_task.cancel()
                try:
                    await abort_task
                except asyncio.CancelledError:
                    pass

            for _ in range(3):
                try:
                    chunk = os.read(master_fd, 4096)
                except (BlockingIOError, OSError):
                    break
                if not chunk:
                    break
                chunks.append(chunk)
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001 - executor PTY read boundary
            return CommandResult(
                stdout=b"".join(chunks).decode("utf-8", errors="replace"),
                stderr=f"Failed to execute PTY command: {exc}",
                exit_code=1,
            )
        finally:
            # `process.wait()` can finish before the PTY reader observes EOF.
            # If we wait on `process_done` unconditionally, short-lived PTY
            # commands can hang forever during cleanup.
            if not process_done.is_set():
                loop.remove_reader(master_fd)
                process_done.set()
            try:
                os.close(master_fd)
            except OSError:
                pass

        if process.returncode is None:
            await process.wait()
        return CommandResult(
            stdout=b"".join(chunks).decode("utf-8", errors="replace"),
            stderr="",
            exit_code=process.returncode or 0,
        )

    async def read_file(self, path: str) -> str:
        file_path = self._resolve_path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise IsADirectoryError(f"Path is a directory: {path}")
        return file_path.read_text(encoding="utf-8")

    async def write_files(self, files: list[WriteFileSpec]) -> None:
        for spec in files:
            file_path = self._resolve_path(spec.path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(spec.content, str):
                file_path.write_text(spec.content, encoding="utf-8")
            else:
                file_path.write_bytes(spec.content)

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
        async with self._start_lock:
            running = await self.list_processes(state="running")
            if len(running) >= self.max_processes:
                sample = running[:5]
                running_list = ", ".join([p.process_id for p in sample])
                if len(running) > 5:
                    running_list += f" and {len(running) - 5} more"
                raise RuntimeError(
                    f"Too many running processes (limit: {self.max_processes}). "
                    f"Stop one first. Currently running: {running_list}. "
                    "Use `bash_process` with `action=jobs` to inspect and "
                    "`action=stop` to terminate."
                )

            working_dir = str(self.workspace)
            if cwd:
                target = self._resolve_path(cwd)
                target.mkdir(parents=True, exist_ok=True)
                working_dir = str(target)

            return self._registry.start_process(
                command=command,
                cwd=working_dir,
                env=env,
                agent_id=agent_id,
                use_pty=use_pty,
                pty_cols=pty_cols,
                pty_rows=pty_rows,
            )

    async def attach_process(self, process_id: str) -> ProcessInfo:
        return self._registry.attach_process(process_id)

    async def get_process_status(self, process_id: str) -> ProcessStatus:
        return self._registry.get_process_status(process_id)

    async def stop_process(self, process_id: str, signal: str = "TERM") -> None:
        self._registry.stop_process(process_id, signal)

    async def write_process_stdin(
        self,
        process_id: str,
        data: str,
    ) -> None:
        self._registry.write_process_stdin(process_id, data)

    async def list_processes(
        self,
        state: Literal["running", "all"] = "all",
    ) -> list[ProcessInfo]:
        return self._registry.list_processes(state)

    async def list_processes_by_agent(
        self,
        agent_id: str,
        state: Literal["running", "all"] = "all",
    ) -> list[ProcessInfo]:
        return self._registry.list_processes_by_agent(agent_id, state)

    async def get_process_logs_info(self, process_id: str) -> ProcessLogInfo:
        return self._registry.get_process_logs_info(process_id)

    def _build_env(self, extra_env: dict[str, str] | None) -> dict[str, str] | None:
        """Merge extra environment variables with current process environment."""
        if extra_env is None:
            return None
        merged = dict(os.environ)
        merged.update(extra_env)
        return merged

    def _resolve_path(self, path: str) -> Path:
        if path.startswith("/"):
            path = path.lstrip("/")
        target = (self.workspace / path).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError:
            raise ValueError(f"Path escapes workspace: {path}") from None
        return target

    async def cleanup(self) -> None:
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __del__(self) -> None:
        if self._temp_dir:
            self._temp_dir.cleanup()


__all__ = ["LocalExecutor"]
