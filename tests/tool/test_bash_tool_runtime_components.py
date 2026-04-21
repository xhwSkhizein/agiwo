"""Runtime component tests for BashTool internals."""

from datetime import datetime

import pytest
from unittest.mock import AsyncMock

from agiwo.tool.builtin.bash_tool import registry as registry_module
from agiwo.tool.builtin.bash_tool.registry import ProcessRecord, ProcessRegistry
from agiwo.tool.builtin.bash_tool.local_executor import LocalExecutor
from agiwo.tool.builtin.bash_tool.types import ProcessInfo


class TestProcessRegistrySafety:
    def test_update_marks_exited_when_pid_is_reused(self, tmp_path):
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)
        record = ProcessRecord(
            process_id="job_1",
            command="echo test",
            cwd=str(tmp_path),
            pid=42424,
            started_at=datetime.now().timestamp(),
            state="running",
            pid_start_time="Wed Mar  4 10:00:00 2026",
        )
        registry._processes[record.process_id] = record

        registry._get_pid_start_time = lambda _: "Wed Mar  4 10:10:00 2026"  # type: ignore[method-assign]

        jobs = registry.list_processes(state="all")

        assert len(jobs) == 1
        assert jobs[0].state == "exited"
        assert record.state == "exited"
        assert record.exit_code == -1

    def test_stop_does_not_kill_when_pid_mismatch(self, tmp_path, monkeypatch):
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)
        record = ProcessRecord(
            process_id="job_2",
            command="echo test",
            cwd=str(tmp_path),
            pid=52525,
            started_at=datetime.now().timestamp(),
            state="running",
            pid_start_time="Wed Mar  4 10:00:00 2026",
        )
        registry._processes[record.process_id] = record

        registry._get_pid_start_time = lambda _: "Wed Mar  4 11:00:00 2026"  # type: ignore[method-assign]

        def fail_kill(_pid, _sig):
            raise AssertionError("os.kill should not be called when pid does not match")

        monkeypatch.setattr(registry_module.os, "kill", fail_kill)

        registry.stop_process("job_2", signal_name="TERM")

        assert record.state == "exited"
        assert record.exit_code == -1

    def test_list_processes_by_agent_filters_records(self, tmp_path):
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)
        registry._processes["job_a"] = ProcessRecord(
            process_id="job_a",
            command="echo a",
            cwd=str(tmp_path),
            pid=60001,
            started_at=datetime.now().timestamp(),
            state="exited",
            exit_code=0,
            agent_id="agent_a",
            pid_start_time=None,
        )
        registry._processes["job_b"] = ProcessRecord(
            process_id="job_b",
            command="echo b",
            cwd=str(tmp_path),
            pid=60002,
            started_at=datetime.now().timestamp(),
            state="exited",
            exit_code=0,
            agent_id="agent_b",
            pid_start_time=None,
        )

        jobs = registry.list_processes_by_agent("agent_a", state="all")

        assert len(jobs) == 1
        assert jobs[0].process_id == "job_a"


class TestLocalExecutorProcessLimit:
    @pytest.mark.asyncio
    async def test_start_process_checks_running_only(self, tmp_path):
        sandbox = LocalExecutor(workspace_dir=str(tmp_path), max_processes=2)
        captured: dict[str, object] = {}
        sandbox.list_processes = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                ProcessInfo(
                    process_id="running_1",
                    command="sleep 1",
                    state="running",
                )
            ]
        )
        sandbox._registry.start_process = (  # type: ignore[method-assign]
            lambda **kwargs: captured.update(kwargs) or "job_10"
        )

        job_id = await sandbox.start_process("echo hello")

        assert job_id == "job_10"
        assert sandbox.list_processes.await_count == 1
        kwargs = sandbox.list_processes.await_args.kwargs
        assert kwargs["state"] == "running"
        assert captured["use_pty"] is False

    @pytest.mark.asyncio
    async def test_start_process_rejects_when_running_limit_reached(self, tmp_path):
        sandbox = LocalExecutor(workspace_dir=str(tmp_path), max_processes=1)
        sandbox.list_processes = AsyncMock(  # type: ignore[method-assign]
            return_value=[
                ProcessInfo(
                    process_id="running_1",
                    command="sleep 1",
                    state="running",
                )
            ]
        )

        with pytest.raises(RuntimeError, match="Too many running processes"):
            await sandbox.start_process("echo hello")

    @pytest.mark.asyncio
    async def test_start_process_for_pty_passes_tty_size(self, tmp_path):
        sandbox = LocalExecutor(workspace_dir=str(tmp_path), max_processes=2)
        captured: dict[str, object] = {}
        sandbox.list_processes = AsyncMock(return_value=[])  # type: ignore[method-assign]
        sandbox._registry.start_process = (  # type: ignore[method-assign]
            lambda **kwargs: captured.update(kwargs) or "job_pty"
        )

        job_id = await sandbox.start_process(
            "codex",
            use_pty=True,
            pty_cols=160,
            pty_rows=50,
            agent_id="agent-1",
        )

        assert job_id == "job_pty"
        assert captured["use_pty"] is True
        assert captured["pty_cols"] == 160
        assert captured["pty_rows"] == 50

    @pytest.mark.asyncio
    async def test_write_process_stdin_delegates_to_registry(self, tmp_path):
        sandbox = LocalExecutor(workspace_dir=str(tmp_path))
        called: dict[str, object] = {}
        sandbox._registry.write_process_stdin = (  # type: ignore[method-assign]
            lambda process_id, data: called.update(
                {"process_id": process_id, "data": data}
            )
        )

        await sandbox.write_process_stdin("job_1", "input\n")
        assert called["process_id"] == "job_1"
        assert called["data"] == "input\n"
