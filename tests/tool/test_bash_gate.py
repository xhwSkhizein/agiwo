"""Focused tests for bash gate decisions."""

import pytest

from agiwo.tool.builtin.bash_tool.tool import BashTool, BashToolConfig
from agiwo.tool.context import ToolContext
from tests.tool.bash_tool_test_support import MockSandbox


@pytest.mark.asyncio
async def test_bash_tool_gate_allows_safe_command() -> None:
    tool = BashTool(BashToolConfig(sandbox=MockSandbox(), cwd="/workspace"))

    decision = await tool.gate(
        {"command": "echo hello"},
        ToolContext(session_id="session-1"),
    )

    assert decision.action == "allow"


@pytest.mark.asyncio
async def test_bash_tool_gate_allows_non_destructive_command() -> None:
    tool = BashTool(BashToolConfig(sandbox=MockSandbox(), cwd="/workspace"))

    decision = await tool.gate(
        {"command": "sudo rm -rf /tmp/test"},
        ToolContext(session_id="session-1"),
    )

    assert decision.action == "allow"
    assert "passed hard safety checks" in decision.reason


@pytest.mark.asyncio
async def test_bash_tool_gate_denies_hard_block_command() -> None:
    tool = BashTool(BashToolConfig(sandbox=MockSandbox(), cwd="/workspace"))

    decision = await tool.gate(
        {"command": "rm -rf /"},
        ToolContext(session_id="session-1"),
    )

    assert decision.action == "deny"
    assert "hard safety rule" in decision.reason


@pytest.mark.asyncio
async def test_bash_tool_direct_execute_allows_non_destructive_command_without_gate_flag() -> (
    None
):
    sandbox = MockSandbox()
    tool = BashTool(BashToolConfig(sandbox=sandbox, cwd="/workspace"))

    result = await tool.execute(
        {"command": "sudo echo hello", "tool_call_id": "call-1"},
        ToolContext(session_id="session-1"),
    )

    assert result.is_success is True
    assert sandbox.executed_commands == ["sudo echo hello"]
