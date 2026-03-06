"""Coding-agent style BashTool tests with real Codex smoke coverage."""

import asyncio
import shlex
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.builtin.bash_tool.sandbox.local import LocalSandbox
from agiwo.tool.builtin.bash_tool.tool import BashTool, BashToolConfig


def has_codex_binary() -> bool:
    """Return True if Codex CLI is available in current PATH."""
    return shutil.which("codex") is not None


skip_without_codex = pytest.mark.skipif(
    not has_codex_binary(),
    reason="codex binary not found in PATH",
)


@pytest.fixture
def mock_context():
    context = MagicMock(spec=ExecutionContext)
    context.agent_id = "coding-agent"
    return context


@pytest.fixture
def local_bash_tool(tmp_path: Path) -> BashTool:
    sandbox = LocalSandbox(workspace_dir=str(tmp_path), max_processes=10)
    return BashTool(
        BashToolConfig(
            sandbox=sandbox,
            cwd=".",
        )
    )


async def _wait_for_job_exit(
    tool: BashTool,
    context: ExecutionContext,
    job_id: str,
    timeout_seconds: float = 8.0,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status_result = await tool.execute(
            {"command": f"bashctl status {job_id}", "tool_call_id": "tc_wait_status"},
            context,
        )
        if status_result.output.get("state") != "running":
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"job did not exit in time: {job_id}")


async def _wait_for_log_contains(
    tool: BashTool,
    context: ExecutionContext,
    job_id: str,
    expected: str,
    timeout_seconds: float = 8.0,
) -> str:
    deadline = time.time() + timeout_seconds
    latest = ""
    while time.time() < deadline:
        logs_result = await tool.execute(
            {
                "command": f"bashctl logs {job_id} --stream stdout -n 400",
                "tool_call_id": "tc_wait_logs",
            },
            context,
        )
        latest = logs_result.output.get("stdout", "")
        if expected in latest:
            return latest
        await asyncio.sleep(0.1)
    raise AssertionError(f"did not find '{expected}' in logs for {job_id}: {latest}")


class TestBashToolCodingAgentUsage:
    """Validate simplified coding-agent flows against BashTool."""

    @skip_without_codex
    @pytest.mark.asyncio
    async def test_codex_foreground_with_cwd_and_pty(
        self,
        local_bash_tool: BashTool,
        mock_context: ExecutionContext,
        tmp_path: Path,
    ):
        project_dir = tmp_path / "project_fg"
        project_dir.mkdir(parents=True, exist_ok=True)

        result = await local_bash_tool.execute(
            {
                "command": "pwd && codex --version",
                "cwd": "project_fg",
                "pty": True,
                "tool_call_id": "tc_codex_fg",
            },
            mock_context,
        )

        assert result.output["ok"] is True
        assert result.output["mode"] == "pty"
        assert str(project_dir) in result.output["stdout"]
        combined = f"{result.output['stdout']}\n{result.output['stderr']}".lower()
        assert "codex" in combined

    @skip_without_codex
    @pytest.mark.asyncio
    async def test_codex_background_and_bashctl_flow(
        self,
        local_bash_tool: BashTool,
        mock_context: ExecutionContext,
        tmp_path: Path,
    ):
        project_dir = tmp_path / "project_bg"
        project_dir.mkdir(parents=True, exist_ok=True)

        start_result = await local_bash_tool.execute(
            {
                "command": "pwd && codex --help",
                "cwd": "project_bg",
                "pty": True,
                "background": True,
                "tool_call_id": "tc_codex_bg_start",
            },
            mock_context,
        )
        assert start_result.output["ok"] is True
        assert start_result.output["background"] is True
        assert start_result.output["mode"] == "pty"
        job_id = start_result.output["job_id"]

        status_result = await local_bash_tool.execute(
            {"command": f"bashctl status {job_id}", "tool_call_id": "tc_codex_bg_status"},
            mock_context,
        )
        assert status_result.output["ok"] is True
        assert status_result.output["mode"] == "pty"
        assert status_result.output["state"] in {"running", "exited"}

        await _wait_for_job_exit(local_bash_tool, mock_context, job_id)

        logs_result = await local_bash_tool.execute(
            {
                "command": f"bashctl logs {job_id} --stream stdout -n 400",
                "tool_call_id": "tc_codex_bg_logs",
            },
            mock_context,
        )
        assert logs_result.output["ok"] is True
        assert logs_result.output["mode"] == "pty"
        assert str(project_dir) in logs_result.output["stdout"]
        assert "codex" in logs_result.output["stdout"].lower()

        paths_result = await local_bash_tool.execute(
            {"command": f"bashctl paths {job_id}", "tool_call_id": "tc_codex_bg_paths"},
            mock_context,
        )
        assert paths_result.output["ok"] is True
        assert paths_result.output["mode"] == "pty"
        assert paths_result.output["stdout_path"].endswith(".stdout")
        assert paths_result.output["stderr_path"].endswith(".stderr")

        jobs_result = await local_bash_tool.execute(
            {"command": "bashctl jobs", "tool_call_id": "tc_codex_bg_jobs"},
            mock_context,
        )
        assert jobs_result.output["ok"] is True
        assert job_id in jobs_result.output["stdout"]

        running_jobs_result = await local_bash_tool.execute(
            {"command": "bashctl jobs --running", "tool_call_id": "tc_codex_bg_jobs_running"},
            mock_context,
        )
        assert running_jobs_result.output["ok"] is True

        stop_result = await local_bash_tool.execute(
            {"command": f"bashctl stop {job_id} --force", "tool_call_id": "tc_codex_bg_stop"},
            mock_context,
        )
        assert stop_result.output["ok"] is True
        assert stop_result.output["job_id"] == job_id

    @pytest.mark.asyncio
    async def test_bashctl_input_for_background_pty_job(
        self,
        local_bash_tool: BashTool,
        mock_context: ExecutionContext,
        tmp_path: Path,
    ):
        project_dir = tmp_path / "project_input"
        project_dir.mkdir(parents=True, exist_ok=True)

        script = (
            "import os,sys,time;"
            "print(os.getcwd(), flush=True);"
            "line=sys.stdin.readline().strip();"
            "print('ACK:'+line, flush=True);"
            "time.sleep(0.2)"
        )
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

        start_result = await local_bash_tool.execute(
            {
                "command": command,
                "cwd": "project_input",
                "pty": True,
                "background": True,
                "tool_call_id": "tc_input_start",
            },
            mock_context,
        )
        assert start_result.output["ok"] is True
        assert start_result.output["mode"] == "pty"
        job_id = start_result.output["job_id"]

        input_result = await local_bash_tool.execute(
            {
                "command": f"bashctl input {job_id} hello-from-coding-agent",
                "tool_call_id": "tc_input_send",
            },
            mock_context,
        )
        assert input_result.output["ok"] is True
        assert input_result.output["bytes"] > 0

        logs_text = await _wait_for_log_contains(
            local_bash_tool,
            mock_context,
            job_id,
            "ACK:hello-from-coding-agent",
        )
        assert str(project_dir) in logs_text
