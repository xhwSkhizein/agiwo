"""Runtime component tests for BashTool internals."""

import os
import shlex
import signal
import sys
import time
from datetime import datetime

import pytest
from unittest.mock import AsyncMock

from agiwo.tool.builtin.bash_tool import registry as registry_module
from agiwo.tool.builtin.bash_tool.registry import ProcessRecord, ProcessRegistry
from agiwo.tool.builtin.bash_tool.sandbox.local import LocalSandbox
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

    def test_owner_check_hides_record_as_not_found(self, tmp_path):
        """Cross-agent lookups must raise KeyError (indistinguishable from
        'not found') so callers cannot enumerate other agents' job ids.
        """
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)
        record = ProcessRecord(
            process_id="job_owned",
            command="sleep 10",
            cwd=str(tmp_path),
            pid=0,
            started_at=datetime.now().timestamp(),
            state="exited",
            exit_code=0,
            agent_id="agent_a",
            pid_start_time=None,
        )
        registry._processes[record.process_id] = record

        with pytest.raises(KeyError):
            registry.get_process_status("job_owned", owner_agent_id="agent_b")
        with pytest.raises(KeyError):
            registry.attach_process("job_owned", owner_agent_id="agent_b")
        with pytest.raises(KeyError):
            registry.get_process_logs_info("job_owned", owner_agent_id="agent_b")
        with pytest.raises(KeyError):
            registry.stop_process(
                "job_owned", signal_name="TERM", owner_agent_id="agent_b"
            )
        with pytest.raises(KeyError):
            registry.write_process_stdin("job_owned", "hi", owner_agent_id="agent_b")

        # Owner still sees the record.
        status = registry.get_process_status("job_owned", owner_agent_id="agent_a")
        assert status.state == "exited"

    def test_owner_check_bypassed_for_none_caller(self, tmp_path):
        """Admin/CLI path (owner_agent_id=None) must keep working."""
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)
        record = ProcessRecord(
            process_id="job_legacy",
            command="echo",
            cwd=str(tmp_path),
            pid=0,
            started_at=datetime.now().timestamp(),
            state="exited",
            exit_code=0,
            agent_id="agent_a",
            pid_start_time=None,
        )
        registry._processes[record.process_id] = record

        status = registry.get_process_status("job_legacy", owner_agent_id=None)
        assert status.state == "exited"

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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TestProcessRegistryStopProcessGroup:
    """Regression for Issue 2: stop_process must kill the whole process group.

    Without ``start_new_session=True`` + ``os.killpg``, signalling just the
    shell leader would leave descendants reparented to init, so ``status``
    would claim the job exited while the real workload keeps running.
    """

    def test_stop_process_kills_reparented_child(self, tmp_path):
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)

        # Run a bash command that launches a bash child (the worker). The
        # child will inherit the shell's new session/process-group, so
        # killpg must reach it.
        # Single-quoted so the outer sh does not expand ``$!``; bash expands
        # it after backgrounding ``sleep``.
        command = "bash -c 'sleep 30 & echo $! ; wait'"
        job_id = registry.start_process(command=command, cwd=str(tmp_path))
        record = registry._processes[job_id]

        # Wait until the child PID has been printed to stdout log.
        worker_pid: int | None = None
        deadline = time.monotonic() + 5.0
        stdout_path = logs_dir / f"{job_id}.stdout"
        while time.monotonic() < deadline:
            if stdout_path.exists():
                text = stdout_path.read_text().strip()
                if text.isdigit():
                    worker_pid = int(text)
                    break
            time.sleep(0.05)
        assert worker_pid is not None, "worker pid was not captured in time"
        assert _pid_alive(worker_pid), "worker should still be running before stop"

        registry.stop_process(job_id, signal_name="TERM")

        # After stop_process returns: state must be exited AND worker dead.
        assert record.state == "exited"
        for _ in range(40):  # up to 2s for reaping
            if not _pid_alive(worker_pid):
                break
            time.sleep(0.05)
        assert not _pid_alive(worker_pid), (
            "worker process survived stop_process; process-group kill missing"
        )

    def test_stop_waits_for_process_group_not_just_leader(self, tmp_path):
        """If the leader exits on TERM but a child ignores TERM, stop_process
        must keep waiting for the process group and escalate to SIGKILL.
        """
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)

        child_script = tmp_path / "child.py"
        child_script.write_text(
            "\n".join(
                [
                    "import os",
                    "import signal",
                    "import time",
                    "",
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                    "print(os.getpid(), flush=True)",
                    "time.sleep(60)",
                ]
            )
            + "\n"
        )
        parent_script = tmp_path / "parent.py"
        parent_script.write_text(
            "\n".join(
                [
                    "import signal",
                    "import subprocess",
                    "import sys",
                    "import time",
                    "",
                    "signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))",
                    f"subprocess.Popen([{sys.executable!r}, {str(child_script)!r}])",
                    "time.sleep(60)",
                ]
            )
            + "\n"
        )

        command = f"{shlex.quote(sys.executable)} {shlex.quote(str(parent_script))}"
        job_id = registry.start_process(command=command, cwd=str(tmp_path))
        record = registry._processes[job_id]

        child_pid: int | None = None
        deadline = time.monotonic() + 5.0
        stdout_path = logs_dir / f"{job_id}.stdout"
        while time.monotonic() < deadline:
            if stdout_path.exists():
                text = stdout_path.read_text().strip()
                if text.isdigit():
                    child_pid = int(text)
                    break
            time.sleep(0.05)
        assert child_pid is not None, "child pid was not captured in time"
        assert _pid_alive(child_pid), "child should still be running before stop"

        start = time.monotonic()
        registry.stop_process(job_id, signal_name="TERM")
        elapsed = time.monotonic() - start

        assert record.state == "exited"
        assert not _pid_alive(child_pid), (
            "child process survived stop_process; process-group wait is incomplete"
        )
        assert elapsed >= 2.5, (
            f"stop_process returned in {elapsed:.2f}s; "
            "leader exit was treated as group exit"
        )

    def test_pipe_launcher_uses_new_session(self, tmp_path):
        """``_start_pipe_process`` must create a new session so pgid == pid."""
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)
        job_id = registry.start_process(command="sleep 30", cwd=str(tmp_path))
        record = registry._processes[job_id]
        try:
            assert os.getpgid(record.pid) == record.pid
        finally:
            try:
                os.killpg(record.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    def test_pty_launcher_uses_new_session(self, tmp_path):
        """``_start_pty_process`` is an independent code path and must also
        produce a session leader (pgid == pid)."""
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)
        job_id = registry.start_process(
            command="sleep 30", cwd=str(tmp_path), use_pty=True
        )
        record = registry._processes[job_id]
        assert record.mode == "pty"
        try:
            assert os.getpgid(record.pid) == record.pid
        finally:
            try:
                os.killpg(record.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

    def test_stop_escalates_to_sigkill_when_term_ignored(self, tmp_path):
        """Regression: when the shell leader itself ignores SIGTERM,
        ``stop_process`` must escalate to SIGKILL after the grace period and
        still commit ``state == 'exited'`` with a real exit_code.

        The command is ``trap '' TERM; while true; do sleep 1; done`` so the
        top-level ``sh`` (which is what ``subprocess.Popen`` actually waits
        on with ``shell=True``) installs ``SIG_IGN`` for SIGTERM in its own
        process. Without escalation, ``stop_process`` would hang / never
        complete.
        """
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)
        command = "trap '' TERM; while true; do sleep 1; done"
        job_id = registry.start_process(command=command, cwd=str(tmp_path))
        record = registry._processes[job_id]

        time.sleep(0.3)  # let sh install the trap and enter the loop
        assert _pid_alive(record.pid)

        start = time.monotonic()
        registry.stop_process(job_id, signal_name="TERM")
        elapsed = time.monotonic() - start

        assert record.state == "exited"
        assert not _pid_alive(record.pid)
        # Escalation path waits ~3s for TERM before sending SIGKILL; returning
        # in <2.5s means we never actually paid the grace period.
        assert elapsed >= 2.5, (
            f"stop_process returned in {elapsed:.2f}s; "
            "SIGKILL escalation branch was not exercised"
        )
        assert elapsed < 10.0
        assert record.exit_code == -signal.SIGKILL

    def test_stop_records_real_exit_code(self, tmp_path):
        """Regression: after stop_process, ``exit_code`` must reflect the
        real ``process.returncode`` (Python returns ``-signum`` for signals),
        not a hard-coded -1.
        """
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)
        job_id = registry.start_process(command="sleep 60", cwd=str(tmp_path))
        record = registry._processes[job_id]
        assert _pid_alive(record.pid)

        registry.stop_process(job_id, signal_name="KILL")

        assert record.state == "exited"
        assert record.exit_code == -signal.SIGKILL

    def test_stop_is_idempotent_on_exited_record(self, tmp_path):
        """Regression: stopping an already-exited record must be a no-op,
        must not raise, and must preserve the recorded exit_code.
        """
        logs_dir = tmp_path / "logs"
        registry = ProcessRegistry(logs_dir)
        record = ProcessRecord(
            process_id="job_done",
            command="echo",
            cwd=str(tmp_path),
            pid=0,
            started_at=datetime.now().timestamp(),
            state="exited",
            exit_code=0,
            agent_id=None,
            pid_start_time=None,
        )
        registry._processes[record.process_id] = record

        registry.stop_process("job_done", signal_name="TERM")
        registry.stop_process("job_done", signal_name="KILL")

        assert record.state == "exited"
        assert record.exit_code == 0


class TestLocalSandboxProcessLimit:
    @pytest.mark.asyncio
    async def test_start_process_checks_running_only(self, tmp_path):
        sandbox = LocalSandbox(workspace_dir=str(tmp_path), max_processes=2)
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
        sandbox = LocalSandbox(workspace_dir=str(tmp_path), max_processes=1)
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
        sandbox = LocalSandbox(workspace_dir=str(tmp_path), max_processes=2)
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
        sandbox = LocalSandbox(workspace_dir=str(tmp_path))
        called: dict[str, object] = {}

        def fake_write(process_id, data, *, owner_agent_id=None):
            called["process_id"] = process_id
            called["data"] = data
            called["owner_agent_id"] = owner_agent_id

        sandbox._registry.write_process_stdin = fake_write  # type: ignore[method-assign]

        await sandbox.write_process_stdin("job_1", "input\n", owner_agent_id="agent_1")
        assert called["process_id"] == "job_1"
        assert called["data"] == "input\n"
        assert called["owner_agent_id"] == "agent_1"
