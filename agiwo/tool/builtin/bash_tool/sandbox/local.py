"""
Local sandbox for bash tool
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Literal

from agiwo.tool.builtin.bash_tool.registry import ProcessRegistry
from agiwo.tool.builtin.bash_tool.types import (
    CommandResult,
    ProcessInfo,
    ProcessLogInfo,
    ProcessStatus,
    Sandbox,
    WriteFileSpec,
)


class LocalSandbox(Sandbox):
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

    @property
    def workspace_path(self) -> Path:
        return self.workspace

    async def execute_command(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        working_dir = self.workspace if not cwd else self.workspace / cwd
        working_dir = working_dir.resolve()
        try:
            working_dir.relative_to(self.workspace)
        except ValueError:
            working_dir = self.workspace

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(working_dir),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                raise TimeoutError(
                    f"Command timed out after {timeout} seconds: {command}"
                )

            return CommandResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                exit_code=process.returncode or 0,
            )
        except TimeoutError:
            raise
        except Exception as exc:
            return CommandResult(
                stdout="",
                stderr=f"Failed to execute command: {exc}",
                exit_code=1,
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
        self, command: str, cwd: str | None = None, env: dict[str, str] | None = None
    ) -> str:
        running = await self.list_processes()
        if len(running) >= self.max_processes:
            sample = running[:5]
            running_list = ", ".join([p.process_id for p in sample])
            if len(running) > 5:
                running_list += f" and {len(running) - 5} more"
            raise RuntimeError(
                f"Too many running processes (limit: {self.max_processes}). "
                f"Stop one first. Currently running: {running_list}. "
                "Use `bashctl jobs` to inspect and `bashctl stop <job_id>` to terminate."
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
        )

    async def attach_process(self, process_id: str) -> ProcessInfo:
        return self._registry.attach_process(process_id)

    async def get_process_status(self, process_id: str) -> ProcessStatus:
        return self._registry.get_process_status(process_id)

    async def stop_process(self, process_id: str, signal: str = "TERM") -> None:
        self._registry.stop_process(process_id, signal)

    async def list_processes(
        self,
        state: Literal["running", "all"] = "all",
    ) -> list[ProcessInfo]:
        return self._registry.list_processes(state)

    async def get_process_logs_info(self, process_id: str) -> ProcessLogInfo:
        return self._registry.get_process_logs_info(process_id)

    def _resolve_path(self, path: str) -> Path:
        if path.startswith("/"):
            path = path.lstrip("/")
        target = (self.workspace / path).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError:
            raise ValueError(f"Path escapes workspace: {path}")
        return target

    async def cleanup(self) -> None:
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def __del__(self) -> None:
        if self._temp_dir:
            self._temp_dir.cleanup()
