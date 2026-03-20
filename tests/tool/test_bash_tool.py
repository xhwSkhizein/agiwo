"""Focused tests for BashTool command execution behavior."""

from agiwo.tool.base import ToolResult
from agiwo.tool.context import ToolContext
from agiwo.tool.builtin.bash_tool.tool import BashTool, BashToolConfig
from agiwo.tool.builtin.bash_tool.types import (
    AfterBashCallOutput,
    BeforeBashCallOutput,
    CommandResult,
)

pytest_plugins = ("tests.tool.bash_tool_test_support",)


class TestBashToolBasic:
    async def test_gate_allows_non_destructive_command(self, bash_tool):
        decision = await bash_tool.gate(
            {"command": "sudo echo hello"},
            ToolContext(session_id="session-1"),
        )

        assert decision.action == "allow"

    async def test_execute_returns_tool_result(self, bash_tool, mock_context):
        result = await bash_tool.execute(
            {"command": "echo hello", "tool_call_id": "tc_001"},
            mock_context,
        )

        assert isinstance(result, ToolResult)
        assert result.tool_name == "bash"
        assert result.output["ok"] is True
        assert result.output["exit_code"] == 0
        assert "hello" in result.output["stdout"]

    async def test_empty_command_returns_error(self, bash_tool, mock_context):
        result = await bash_tool.execute(
            {"command": "", "tool_call_id": "tc_002"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert "command is required" in result.output["stderr"]

    async def test_command_with_error_exit_code(self, bash_tool, mock_context):
        result = await bash_tool.execute(
            {"command": "exit 1", "tool_call_id": "tc_003"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert result.output["exit_code"] == 1
        assert result.is_success is False

    async def test_direct_execute_denies_hard_block_command(
        self, bash_tool, mock_context
    ):
        result = await bash_tool.execute(
            {"command": "rm -rf /", "tool_call_id": "tc_003b"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert "hard safety rule" in result.output["stderr"]

    async def test_timeout_validation_error(self, bash_tool, mock_context):
        result = await bash_tool.execute(
            {"command": "echo hi", "timeout": "invalid", "tool_call_id": "tc_004"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert "timeout must be a number" in result.output["stderr"]

    async def test_stdin_requires_pty(self, bash_tool, mock_context):
        result = await bash_tool.execute(
            {"command": "echo hi", "stdin": "input", "tool_call_id": "tc_005"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert "stdin requires pty=true" in result.output["stderr"]

    async def test_pty_foreground_uses_default_tty_size(self, bash_tool, mock_context):
        result = await bash_tool.execute(
            {
                "command": "echo hello",
                "pty": True,
                "stdin": "hi\n",
                "tool_call_id": "tc_005b",
            },
            mock_context,
        )

        assert result.output["ok"] is True
        assert result.output["mode"] == "pty"
        call = bash_tool.config.sandbox.execute_calls[-1]
        assert call["use_pty"] is True
        assert call["pty_cols"] == 120
        assert call["pty_rows"] == 40
        assert call["stdin"] == "hi\n"

    async def test_bashctl_like_command_is_forwarded_to_shell(
        self, bash_tool, mock_context
    ):
        await bash_tool.execute(
            {"command": "bashctl jobs", "tool_call_id": "tc_005c"},
            mock_context,
        )

        assert bash_tool.config.sandbox.executed_commands[-1] == "bashctl jobs"

    async def test_description_points_to_bash_process_tool(self, bash_tool):
        assert "bash_process" in bash_tool.description
        assert "bashctl" not in bash_tool.description


class TestBashToolBackgroundJobs:
    async def test_start_background_job(self, bash_tool, mock_context):
        result = await bash_tool.execute(
            {"command": "sleep 10", "background": True, "tool_call_id": "tc_006"},
            mock_context,
        )

        assert result.output["ok"] is True
        assert result.output["background"] is True
        assert result.output["state"] == "running"
        assert "job_" in result.output["job_id"]
        assert result.output["mode"] == "pipe"
        assert (
            bash_tool.config.sandbox.started_process_calls[-1]["agent_id"] == "agent_1"
        )

    async def test_start_background_job_with_pty(self, bash_tool, mock_context):
        result = await bash_tool.execute(
            {
                "command": "codex",
                "pty": True,
                "background": True,
                "tool_call_id": "tc_006b",
            },
            mock_context,
        )

        assert result.output["ok"] is True
        assert result.output["background"] is True
        assert result.output["mode"] == "pty"
        assert bash_tool.config.sandbox.started_process_calls[-1]["mode"] == "pty"


class TestBashToolInvalidCommands:
    async def test_trailing_ampersand_is_rejected(self, bash_tool, mock_context):
        result = await bash_tool.execute(
            {"command": "sleep 10 &", "tool_call_id": "tc_029b"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert result.output["exit_code"] == 2
        assert "background=true" in result.output["stderr"]


class TestBashToolTiming:
    async def test_result_has_timing_info(self, bash_tool, mock_context):
        result = await bash_tool.execute(
            {"command": "echo test", "tool_call_id": "tc_030"},
            mock_context,
        )

        assert result.start_time > 0
        assert result.end_time > 0
        assert result.duration >= 0
        assert result.end_time >= result.start_time

    async def test_tool_result_content_format(self, bash_tool, mock_context):
        result = await bash_tool.execute(
            {"command": "echo hello", "tool_call_id": "tc_031"},
            mock_context,
        )

        assert "exit_code:" in result.content
        assert "stdout:" in result.content


class TestBashToolHooks:
    async def test_before_hook_modifies_command(self, mock_sandbox, mock_context):
        def before_hook(input_data):
            return BeforeBashCallOutput(
                command=input_data.command.replace("OLD", "NEW")
            )

        tool = BashTool(
            BashToolConfig(
                sandbox=mock_sandbox,
                cwd="/workspace",
                on_before_bash_call=before_hook,
            )
        )

        result = await tool.execute(
            {"command": "echo OLD", "tool_call_id": "tc_032"},
            mock_context,
        )

        assert result.output["stdout"].strip() == "NEW"
        assert mock_sandbox.executed_commands[-1] == "echo NEW"

    async def test_after_hook_modifies_result(self, mock_sandbox, mock_context):
        def after_hook(input_data):
            modified = CommandResult(
                exit_code=input_data.result.exit_code,
                stdout=input_data.result.stdout + " [MODIFIED]",
                stderr=input_data.result.stderr,
            )
            return AfterBashCallOutput(result=modified)

        tool = BashTool(
            BashToolConfig(
                sandbox=mock_sandbox,
                cwd="/workspace",
                on_after_bash_call=after_hook,
            )
        )

        result = await tool.execute(
            {"command": "echo test", "tool_call_id": "tc_033"},
            mock_context,
        )

        assert "[MODIFIED]" in result.output["stdout"]


class TestBashToolDefaultConstruction:
    async def test_bash_tool_no_args_construction(self):
        tool = BashTool()
        assert tool.config is not None
        assert tool.config.sandbox is not None
        assert tool.config.cwd == "."

    async def test_bash_tool_no_args_execution(self, mock_context):
        tool = BashTool()
        result = await tool.execute(
            {"command": "echo hello", "tool_call_id": "tc_041"},
            mock_context,
        )

        assert result.output["ok"] is True
        assert "hello" in result.output["stdout"]
