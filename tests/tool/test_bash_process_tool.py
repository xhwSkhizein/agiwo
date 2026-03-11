"""Tests for BashProcessTool background job management."""

import pytest

from agiwo.tool.base import ToolResult
from agiwo.tool.builtin.bash_tool.process_tool import BashProcessTool

pytest_plugins = ("tests.tool.bash_tool_test_support",)


class TestBashProcessToolBasic:
    async def test_jobs_empty(self, bash_process_tool, mock_context):
        result = await bash_process_tool.execute(
            {"action": "jobs", "tool_call_id": "tc_101"},
            mock_context,
        )

        assert isinstance(result, ToolResult)
        assert result.output["ok"] is True
        assert "no jobs" in result.output["stdout"]

    async def test_jobs_with_running_filter(self, bash_tool, bash_process_tool, mock_context):
        await bash_tool.execute(
            {"command": "sleep 10", "background": True, "tool_call_id": "tc_102"},
            mock_context,
        )

        result = await bash_process_tool.execute(
            {"action": "jobs", "running_only": True, "tool_call_id": "tc_103"},
            mock_context,
        )

        assert result.output["ok"] is True
        assert result.output["count"] == 1
        assert "JOB ID" in result.output["stdout"]

    async def test_status(self, bash_tool, bash_process_tool, mock_context):
        job_result = await bash_tool.execute(
            {"command": "sleep 10", "background": True, "tool_call_id": "tc_104"},
            mock_context,
        )
        job_id = job_result.output["job_id"]

        result = await bash_process_tool.execute(
            {"action": "status", "job_id": job_id, "tool_call_id": "tc_105"},
            mock_context,
        )

        assert result.output["ok"] is True
        assert result.output["job_id"] == job_id
        assert result.output["state"] == "running"
        assert result.output["mode"] == "pipe"

    async def test_status_not_found(self, bash_process_tool, mock_context):
        result = await bash_process_tool.execute(
            {"action": "status", "job_id": "missing", "tool_call_id": "tc_106"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert result.output["exit_code"] == 1
        assert "job not found" in result.output["stderr"]

    async def test_paths(self, bash_tool, bash_process_tool, mock_context):
        job_result = await bash_tool.execute(
            {"command": "sleep 10", "background": True, "tool_call_id": "tc_107"},
            mock_context,
        )
        job_id = job_result.output["job_id"]

        result = await bash_process_tool.execute(
            {"action": "paths", "job_id": job_id, "tool_call_id": "tc_108"},
            mock_context,
        )

        assert result.output["ok"] is True
        assert result.output["stdout_path"] == f"/tmp/{job_id}.stdout"
        assert result.output["stderr_path"] == f"/tmp/{job_id}.stderr"
        assert result.output["mode"] == "pipe"

    async def test_stop(self, bash_tool, bash_process_tool, mock_context):
        job_result = await bash_tool.execute(
            {"command": "sleep 10", "background": True, "tool_call_id": "tc_109"},
            mock_context,
        )
        job_id = job_result.output["job_id"]

        result = await bash_process_tool.execute(
            {"action": "stop", "job_id": job_id, "tool_call_id": "tc_110"},
            mock_context,
        )

        assert result.output["ok"] is True
        assert "stopped" in result.output["stdout"]

    async def test_stop_force(self, bash_tool, bash_process_tool, mock_context):
        job_result = await bash_tool.execute(
            {"command": "sleep 10", "background": True, "tool_call_id": "tc_111"},
            mock_context,
        )
        job_id = job_result.output["job_id"]

        result = await bash_process_tool.execute(
            {
                "action": "stop",
                "job_id": job_id,
                "force": True,
                "tool_call_id": "tc_112",
            },
            mock_context,
        )

        assert result.output["ok"] is True
        assert result.output["signal"] == "KILL"

    async def test_input(self, bash_tool, bash_process_tool, mock_context):
        job_result = await bash_tool.execute(
            {"command": "codex", "pty": True, "background": True, "tool_call_id": "tc_113"},
            mock_context,
        )
        job_id = job_result.output["job_id"]

        result = await bash_process_tool.execute(
            {
                "action": "input",
                "job_id": job_id,
                "text": "hello world",
                "tool_call_id": "tc_114",
            },
            mock_context,
        )

        assert result.output["ok"] is True
        write = bash_process_tool.config.sandbox.stdin_writes[-1]
        assert write["process_id"] == job_id
        assert write["data"] == "hello world\n"


class TestBashProcessToolLogs:
    async def test_logs_basic(self, bash_tool, bash_process_tool, mock_context):
        job_result = await bash_tool.execute(
            {"command": "sleep 10", "background": True, "tool_call_id": "tc_115"},
            mock_context,
        )
        job_id = job_result.output["job_id"]

        result = await bash_process_tool.execute(
            {"action": "logs", "job_id": job_id, "tool_call_id": "tc_116"},
            mock_context,
        )

        assert result.output["ok"] is True
        assert result.output["job_id"] == job_id

    async def test_logs_not_found(self, bash_process_tool, mock_context):
        result = await bash_process_tool.execute(
            {"action": "logs", "job_id": "missing", "tool_call_id": "tc_117"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert "job not found" in result.output["stderr"]

    async def test_logs_invalid_stream(self, bash_tool, bash_process_tool, mock_context):
        await bash_tool.execute(
            {"command": "sleep 10", "background": True, "tool_call_id": "tc_118"},
            mock_context,
        )

        result = await bash_process_tool.execute(
            {
                "action": "logs",
                "job_id": "job_1",
                "stream": "combined",
                "tool_call_id": "tc_119",
            },
            mock_context,
        )

        assert result.output["ok"] is False
        assert result.output["exit_code"] == 2
        assert "must be one of" in result.output["stderr"]

    async def test_logs_grep_flags(self, bash_tool, bash_process_tool, mock_context):
        await bash_tool.execute(
            {"command": "sleep 10", "background": True, "tool_call_id": "tc_120"},
            mock_context,
        )

        result = await bash_process_tool.execute(
            {
                "action": "logs",
                "job_id": "job_1",
                "grep": "line",
                "context": 2,
                "ignore_case": True,
                "tool_call_id": "tc_121",
            },
            mock_context,
        )

        assert result.output["ok"] is True
        assert "grep -n -i -C 2 -- line" in result.output["logs_command"]


class TestBashProcessToolInvalidRequests:
    async def test_unknown_action(self, bash_process_tool, mock_context):
        result = await bash_process_tool.execute(
            {"action": "unknown", "tool_call_id": "tc_122"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert result.output["exit_code"] == 2

    async def test_jobs_invalid_flag_type(self, bash_process_tool, mock_context):
        result = await bash_process_tool.execute(
            {"action": "jobs", "running_only": "yes please", "tool_call_id": "tc_123"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert "running_only must be a boolean" in result.output["stderr"]

    async def test_status_missing_job_id(self, bash_process_tool, mock_context):
        result = await bash_process_tool.execute(
            {"action": "status", "tool_call_id": "tc_124"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert result.output["exit_code"] == 2

    async def test_stop_missing_job_id(self, bash_process_tool, mock_context):
        result = await bash_process_tool.execute(
            {"action": "stop", "tool_call_id": "tc_125"},
            mock_context,
        )

        assert result.output["ok"] is False
        assert "requires job_id" in result.output["stderr"]

    async def test_input_requires_text(self, bash_tool, bash_process_tool, mock_context):
        job_result = await bash_tool.execute(
            {"command": "codex", "pty": True, "background": True, "tool_call_id": "tc_126"},
            mock_context,
        )

        result = await bash_process_tool.execute(
            {
                "action": "input",
                "job_id": job_result.output["job_id"],
                "tool_call_id": "tc_127",
            },
            mock_context,
        )

        assert result.output["ok"] is False
        assert result.output["exit_code"] == 2

    async def test_list_agent_processes_probe(self, bash_tool, bash_process_tool, mock_context):
        await bash_tool.execute(
            {"command": "sleep 10", "background": True, "tool_call_id": "tc_128"},
            mock_context,
        )

        result = await bash_process_tool.list_agent_processes("agent_1")

        assert len(result) == 1
        assert result[0]["process_id"] == "job_1"


class TestBashProcessToolHttpServerScenario:
    @pytest.mark.asyncio
    async def test_http_server_lifecycle(self, bash_tool, bash_process_tool, mock_context):
        start_result = await bash_tool.execute(
            {
                "command": "python -m http.server 18888",
                "background": True,
                "tool_call_id": "tc_129",
            },
            mock_context,
        )
        assert start_result.output["ok"] is True
        job_id = start_result.output["job_id"]

        status_result = await bash_process_tool.execute(
            {"action": "status", "job_id": job_id, "tool_call_id": "tc_130"},
            mock_context,
        )
        assert status_result.output["ok"] is True
        assert status_result.output["state"] == "running"

        curl_result = await bash_tool.execute(
            {"command": "curl -s http://localhost:18888/", "tool_call_id": "tc_131"},
            mock_context,
        )
        assert curl_result.output["ok"] is True

        logs_result = await bash_process_tool.execute(
            {
                "action": "logs",
                "job_id": job_id,
                "stream": "stdout",
                "tool_call_id": "tc_132",
            },
            mock_context,
        )
        assert logs_result.output["ok"] is True
        assert logs_result.output["job_id"] == job_id

        stop_result = await bash_process_tool.execute(
            {"action": "stop", "job_id": job_id, "tool_call_id": "tc_133"},
            mock_context,
        )
        assert stop_result.output["ok"] is True

        final_status = await bash_process_tool.execute(
            {"action": "status", "job_id": job_id, "tool_call_id": "tc_134"},
            mock_context,
        )
        assert final_status.output["ok"] is True
        assert final_status.output["state"] == "exited"

        jobs_result = await bash_process_tool.execute(
            {"action": "jobs", "tool_call_id": "tc_135"},
            mock_context,
        )
        assert jobs_result.output["ok"] is True
        assert jobs_result.output["count"] >= 1


class TestBashProcessToolDefaultConstruction:
    async def test_no_args_construction(self):
        tool = BashProcessTool()
        assert tool.get_name() == "bash_process"
        assert "background jobs" in tool.get_description()
