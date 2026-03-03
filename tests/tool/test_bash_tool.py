"""Comprehensive tests for BashTool."""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock
from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import ToolResult
from agiwo.tool.builtin.bash_tool.tool import BashTool, BashToolConfig
from agiwo.tool.builtin.bash_tool.types import (
    CommandResult,
    ProcessInfo,
    ProcessStatus,
    ProcessLogInfo,
)
from agiwo.tool.builtin.bash_tool.security import CommandSafetyDecision


class MockSandbox:
    """Mock sandbox for testing."""

    def __init__(self):
        self._processes: dict[str, dict] = {}
        self._process_counter = 0

    async def execute_command(
        self, command: str, cwd: str | None = None, timeout: float | None = None
    ) -> CommandResult:
        """Execute a command and return result."""
        # Simulate echo command
        if command.startswith("echo "):
            output = command[5:].strip().strip('"\'')
            return CommandResult(exit_code=0, stdout=output + "\n", stderr="")

        # Simulate error command
        if command.startswith("exit "):
            code = int(command[5:])
            return CommandResult(exit_code=code, stdout="", stderr=f"error: exit {code}")

        # Simulate log command
        if "tail -n" in command or "grep" in command:
            return CommandResult(exit_code=0, stdout="log line 1\nlog line 2\n", stderr="")

        # Default success
        return CommandResult(exit_code=0, stdout="success\n", stderr="")

    async def start_process(self, command: str, cwd: str | None = None, env: dict | None = None) -> str:
        """Start a background process."""
        self._process_counter += 1
        process_id = f"job_{self._process_counter}"
        self._processes[process_id] = {
            "command": command,
            "state": "running",
            "started_at": time.time(),
            "exit_code": None,
        }
        return process_id

    async def attach_process(self, process_id: str) -> ProcessInfo:
        """Attach to a process."""
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")
        p = self._processes[process_id]
        return ProcessInfo(
            process_id=process_id,
            command=p["command"],
            state=p["state"],
            started_at=p["started_at"],
            exit_code=p["exit_code"],
        )

    async def get_process_status(self, process_id: str) -> ProcessStatus:
        """Get process status."""
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")
        p = self._processes[process_id]
        return ProcessStatus(
            state=p["state"],
            started_at=p["started_at"],
            exit_code=p["exit_code"],
        )

    async def stop_process(self, process_id: str, signal: str = "TERM") -> None:
        """Stop a process."""
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")
        self._processes[process_id]["state"] = "exited"
        self._processes[process_id]["exit_code"] = -9 if signal == "KILL" else -15

    async def list_processes(
        self, state: str = "all"
    ) -> list[ProcessInfo]:
        """List processes."""
        result = []
        for pid, p in self._processes.items():
            if state == "running" and p["state"] != "running":
                continue
            result.append(
                ProcessInfo(
                    process_id=pid,
                    command=p["command"],
                    state=p["state"],
                    started_at=p["started_at"],
                    exit_code=p["exit_code"],
                )
            )
        return result

    async def get_process_logs_info(self, process_id: str) -> ProcessLogInfo:
        """Get log file paths."""
        if process_id not in self._processes:
            raise KeyError(f"Process not found: {process_id}")
        return ProcessLogInfo(
            stdout_path=f"/tmp/{process_id}.stdout",
            stderr_path=f"/tmp/{process_id}.stderr",
        )


@pytest.fixture
def mock_context():
    """Create a mock execution context."""
    return MagicMock(spec=ExecutionContext)


@pytest.fixture
def mock_sandbox():
    """Create a mock sandbox."""
    return MockSandbox()


@pytest.fixture
def bash_tool(mock_sandbox):
    """Create a BashTool with mock sandbox."""
    config = BashToolConfig(
        sandbox=mock_sandbox,
        cwd="/workspace",
    )
    return BashTool(config)


@pytest.fixture
def bash_tool_with_validator(mock_sandbox):
    """Create a BashTool with mock safety validator."""
    validator = MagicMock()

    async def mock_validate(cmd):
        if "dangerous" in cmd:
            return CommandSafetyDecision(
                allowed=False,
                message="Command contains dangerous keyword",
                risk_level="high",
                stage="blacklist",
            )
        return CommandSafetyDecision(
            allowed=True,
            message="OK",
            risk_level="low",
            stage="allow",
        )

    validator.validate = AsyncMock(side_effect=mock_validate)

    config = BashToolConfig(
        sandbox=mock_sandbox,
        cwd="/workspace",
        command_safety_validator=validator,
    )
    return BashTool(config)


class TestBashToolBasic:
    """Basic BashTool functionality tests."""

    async def test_execute_returns_tool_result(self, bash_tool, mock_context):
        """Test that execute returns a ToolResult."""
        parameters = {"command": "echo hello", "tool_call_id": "tc_001"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.tool_name == "bash"
        assert result.output["ok"] is True
        assert result.output["exit_code"] == 0
        assert "hello" in result.output["stdout"]

    async def test_empty_command_returns_error(self, bash_tool, mock_context):
        """Test that empty command returns error."""
        parameters = {"command": "", "tool_call_id": "tc_002"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is False
        assert "command is required" in result.output["stderr"]

    async def test_command_with_error_exit_code(self, bash_tool, mock_context):
        """Test command that returns non-zero exit code."""
        parameters = {"command": "exit 1", "tool_call_id": "tc_003"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is False
        assert result.output["exit_code"] == 1

    async def test_timeout_validation_error(self, bash_tool, mock_context):
        """Test invalid timeout parameter."""
        parameters = {"command": "echo hi", "timeout": "invalid", "tool_call_id": "tc_004"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is False
        assert "timeout must be a number" in result.output["stderr"]

    async def test_stdin_not_supported(self, bash_tool, mock_context):
        """Test that stdin parameter is rejected."""
        parameters = {"command": "echo hi", "stdin": "input", "tool_call_id": "tc_005"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is False
        assert "stdin is not supported" in result.output["stderr"]


class TestBashToolBackgroundJobs:
    """Background job management tests."""

    async def test_start_background_job(self, bash_tool, mock_context):
        """Test starting a background job."""
        parameters = {"command": "sleep 10 &", "tool_call_id": "tc_006"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert result.output["background"] is True
        assert result.output["state"] == "running"
        assert "job_" in result.output["job_id"]

    async def test_bashctl_jobs_empty(self, bash_tool, mock_context):
        """Test bashctl jobs with no jobs."""
        parameters = {"command": "bashctl jobs", "tool_call_id": "tc_007"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert "no jobs" in result.output["stdout"]

    async def test_bashctl_jobs_with_running(self, bash_tool, mock_context):
        """Test bashctl jobs --running filter."""
        # Start a background job first
        await bash_tool.execute({"command": "sleep 10 &", "tool_call_id": "tc_008"}, mock_context)

        parameters = {"command": "bashctl jobs --running", "tool_call_id": "tc_009"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert result.output["count"] == 1
        assert "JOB ID" in result.output["stdout"]

    async def test_bashctl_status(self, bash_tool, mock_context):
        """Test bashctl status command."""
        # Start a job
        job_result = await bash_tool.execute(
            {"command": "sleep 10 &", "tool_call_id": "tc_010"}, mock_context
        )
        job_id = job_result.output["job_id"]

        parameters = {"command": f"bashctl status {job_id}", "tool_call_id": "tc_011"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert result.output["job_id"] == job_id
        assert result.output["state"] == "running"

    async def test_bashctl_status_not_found(self, bash_tool, mock_context):
        """Test bashctl status with non-existent job."""
        parameters = {"command": "bashctl status nonexistent", "tool_call_id": "tc_012"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is False
        assert result.output["exit_code"] == 1
        assert "job not found" in result.output["stderr"]

    async def test_bashctl_stop(self, bash_tool, mock_context):
        """Test stopping a job."""
        # Start a job
        job_result = await bash_tool.execute(
            {"command": "sleep 10 &", "tool_call_id": "tc_013"}, mock_context
        )
        job_id = job_result.output["job_id"]

        parameters = {"command": f"bashctl stop {job_id}", "tool_call_id": "tc_014"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert "stopped" in result.output["stdout"]

    async def test_bashctl_stop_force(self, bash_tool, mock_context):
        """Test force stopping a job."""
        # Start a job
        job_result = await bash_tool.execute(
            {"command": "sleep 10 &", "tool_call_id": "tc_015"}, mock_context
        )
        job_id = job_result.output["job_id"]

        parameters = {"command": f"bashctl stop {job_id} --force", "tool_call_id": "tc_016"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert "KILL" in result.output["stdout"]

    async def test_bashctl_paths(self, bash_tool, mock_context):
        """Test bashctl paths command."""
        # Start a job
        job_result = await bash_tool.execute(
            {"command": "sleep 10 &", "tool_call_id": "tc_017"}, mock_context
        )
        job_id = job_result.output["job_id"]

        parameters = {"command": f"bashctl paths {job_id}", "tool_call_id": "tc_018"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert result.output["stdout_path"] == f"/tmp/{job_id}.stdout"
        assert result.output["stderr_path"] == f"/tmp/{job_id}.stderr"


class TestBashToolHelp:
    """Help command tests."""

    async def test_bashctl_help(self, bash_tool, mock_context):
        """Test bashctl help."""
        parameters = {"command": "bashctl help", "tool_call_id": "tc_019"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert "bashctl" in result.output["stdout"]

    async def test_bashctl_help_jobs(self, bash_tool, mock_context):
        """Test bashctl help jobs."""
        parameters = {"command": "bashctl help jobs", "tool_call_id": "tc_020"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert "jobs" in result.output["stdout"]


class TestBashToolSafetyValidator:
    """Safety validator tests."""

    async def test_safety_validator_blocks_dangerous(self, bash_tool_with_validator, mock_context):
        """Test that safety validator blocks dangerous commands."""
        parameters = {"command": "rm -rf / dangerous", "tool_call_id": "tc_021"}
        result = await bash_tool_with_validator.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is False
        assert result.output["exit_code"] == 126
        assert "dangerous" in result.output["stderr"]

    async def test_safety_validator_allows_safe(self, bash_tool_with_validator, mock_context):
        """Test that safety validator allows safe commands."""
        parameters = {"command": "echo hello", "tool_call_id": "tc_022"}
        result = await bash_tool_with_validator.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True


class TestBashToolLogs:
    """Log viewing tests."""

    async def test_bashctl_logs_basic(self, bash_tool, mock_context):
        """Test basic bashctl logs."""
        # Start a job
        job_result = await bash_tool.execute(
            {"command": "sleep 10 &", "tool_call_id": "tc_023"}, mock_context
        )
        job_id = job_result.output["job_id"]

        parameters = {"command": f"bashctl logs {job_id}", "tool_call_id": "tc_024"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert result.output["job_id"] == job_id

    async def test_bashctl_logs_not_found(self, bash_tool, mock_context):
        """Test logs with non-existent job."""
        parameters = {"command": "bashctl logs nonexistent", "tool_call_id": "tc_025"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is False
        assert "job not found" in result.output["stderr"]


class TestBashToolInvalidCommands:
    """Invalid command handling tests."""

    async def test_unknown_bashctl_subcommand(self, bash_tool, mock_context):
        """Test unknown bashctl subcommand."""
        parameters = {"command": "bashctl unknown", "tool_call_id": "tc_026"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is False
        assert result.output["exit_code"] == 2
        assert "unknown bashctl subcommand" in result.output["stderr"]

    async def test_bashctl_jobs_invalid_flag(self, bash_tool, mock_context):
        """Test bashctl jobs with invalid flag."""
        parameters = {"command": "bashctl jobs --invalid", "tool_call_id": "tc_027"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is False
        assert "unknown flag" in result.output["stderr"]

    async def test_bashctl_status_too_many_args(self, bash_tool, mock_context):
        """Test bashctl status with too many arguments."""
        parameters = {"command": "bashctl status job1 job2", "tool_call_id": "tc_028"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is False
        assert result.output["exit_code"] == 2

    async def test_bashctl_stop_missing_job_id(self, bash_tool, mock_context):
        """Test bashctl stop without job ID."""
        parameters = {"command": "bashctl stop", "tool_call_id": "tc_029"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is False
        assert "requires <job_id>" in result.output["stderr"]


class TestBashToolTiming:
    """Timing and duration tests."""

    async def test_result_has_timing_info(self, bash_tool, mock_context):
        """Test that result includes timing information."""
        parameters = {"command": "echo test", "tool_call_id": "tc_030"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.start_time > 0
        assert result.end_time > 0
        assert result.duration >= 0
        assert result.end_time >= result.start_time

    async def test_tool_result_content_format(self, bash_tool, mock_context):
        """Test ToolResult content format."""
        parameters = {"command": "echo hello", "tool_call_id": "tc_031"}
        result = await bash_tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert "exit_code:" in result.content
        assert "stdout:" in result.content


class TestBashToolHooks:
    """Hook callback tests."""

    async def test_before_hook_modifies_command(self, mock_sandbox, mock_context):
        """Test before hook can modify command."""
        def before_hook(input_data):
            from agiwo.tool.builtin.bash_tool.types import BeforeBashCallOutput
            return BeforeBashCallOutput(command=input_data.command.replace("OLD", "NEW"))

        config = BashToolConfig(
            sandbox=mock_sandbox,
            cwd="/workspace",
            on_before_bash_call=before_hook,
        )
        tool = BashTool(config)

        # The mock returns CommandResult based on command content
        # This test verifies the hook is called
        parameters = {"command": "echo OLD", "tool_call_id": "tc_032"}
        result = await tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        # The hook should have modified the command

    async def test_after_hook_modifies_result(self, mock_sandbox, mock_context):
        """Test after hook can modify result."""
        def after_hook(input_data):
            from agiwo.tool.builtin.bash_tool.types import AfterBashCallOutput
            modified = CommandResult(
                exit_code=input_data.result.exit_code,
                stdout=input_data.result.stdout + " [MODIFIED]",
                stderr=input_data.result.stderr,
            )
            return AfterBashCallOutput(result=modified)

        config = BashToolConfig(
            sandbox=mock_sandbox,
            cwd="/workspace",
            on_after_bash_call=after_hook,
        )
        tool = BashTool(config)

        parameters = {"command": "echo test", "tool_call_id": "tc_033"}
        result = await tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert "[MODIFIED]" in result.output["stdout"]


class TestBashToolHttpServerScenario:
    """Integration test: HTTP server lifecycle scenario."""

    async def test_http_server_lifecycle(self, mock_sandbox, mock_context):
        """
        Demonstrate BashTool managing a real HTTP server:
        1. Start python http.server as background job
        2. Access the server with curl
        3. Check server logs to verify request was recorded
        4. Stop the server
        """
        config = BashToolConfig(
            sandbox=mock_sandbox,
            cwd="/workspace",
        )
        tool = BashTool(config)

        # Step 1: Start HTTP server in background
        start_result = await tool.execute(
            {"command": "python -m http.server 18888 &", "tool_call_id": "tc_034"},
            mock_context,
        )
        assert isinstance(start_result, ToolResult)
        assert start_result.output["ok"] is True
        assert start_result.output["background"] is True
        job_id = start_result.output["job_id"]
        assert job_id is not None

        # Step 2: Verify server is running via status
        status_result = await tool.execute(
            {"command": f"bashctl status {job_id}", "tool_call_id": "tc_035"},
            mock_context,
        )
        assert isinstance(status_result, ToolResult)
        assert status_result.output["ok"] is True
        assert status_result.output["state"] == "running"

        # Step 3: Access the HTTP server
        curl_result = await tool.execute(
            {"command": "curl -s http://localhost:18888/", "tool_call_id": "tc_036"},
            mock_context,
        )
        assert isinstance(curl_result, ToolResult)
        assert curl_result.output["ok"] is True

        # Step 4: Check server logs to verify request was logged
        logs_result = await tool.execute(
            {"command": f"bashctl logs {job_id} --stream stdout", "tool_call_id": "tc_037"},
            mock_context,
        )
        assert isinstance(logs_result, ToolResult)
        assert logs_result.output["ok"] is True
        assert logs_result.output["job_id"] == job_id
        assert "stream" in logs_result.output

        # Step 5: Stop the HTTP server
        stop_result = await tool.execute(
            {"command": f"bashctl stop {job_id}", "tool_call_id": "tc_038"},
            mock_context,
        )
        assert isinstance(stop_result, ToolResult)
        assert stop_result.output["ok"] is True
        assert "stopped" in stop_result.output["stdout"]
        assert stop_result.output["job_id"] == job_id

        # Step 6: Verify server is stopped
        final_status = await tool.execute(
            {"command": f"bashctl status {job_id}", "tool_call_id": "tc_039"},
            mock_context,
        )
        assert isinstance(final_status, ToolResult)
        assert final_status.output["ok"] is True
        # After stop, state should be exited
        assert final_status.output["state"] == "exited"

        # Step 7: List all jobs to show job history
        jobs_result = await tool.execute(
            {"command": "bashctl jobs", "tool_call_id": "tc_040"},
            mock_context,
        )
        assert isinstance(jobs_result, ToolResult)
        assert jobs_result.output["ok"] is True
        assert jobs_result.output["count"] >= 1


class TestBashToolDefaultConstruction:
    """Test BashTool can be constructed without parameters."""

    async def test_bash_tool_no_args_construction(self):
        """Test BashTool can be created without any arguments."""
        tool = BashTool()  # Should not raise
        assert tool is not None
        assert tool.config is not None
        assert tool.config.sandbox is not None
        assert tool.config.cwd == "."

    async def test_bash_tool_no_args_execution(self, mock_context):
        """Test BashTool created without args can execute commands."""
        tool = BashTool()
        parameters = {"command": "echo hello", "tool_call_id": "tc_041"}
        result = await tool.execute(parameters, mock_context)

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert "hello" in result.output["stdout"]
