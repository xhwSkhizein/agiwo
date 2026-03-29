from types import SimpleNamespace

import pytest

from agiwo.agent.nested.agent_tool import AgentTool
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.tool_executor import execute_tool_batch
from agiwo.agent.storage.base import InMemoryRunStepStorage
from agiwo.tool.base import BaseTool, ToolGateDecision, ToolResult
from agiwo.tool.context import ToolContext


class EchoTool(BaseTool):
    name = "echo"
    description = "echo"

    def get_parameters(self) -> dict[str, object]:
        return {"type": "object"}

    async def execute(
        self,
        parameters: dict[str, object],
        context: ToolContext,
        abort_signal=None,
    ) -> ToolResult:
        del context, abort_signal
        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=str(parameters.get("tool_call_id", "")),
            input_args=parameters,
            content="ok",
            output={"ok": True},
        )


class DenyTool(EchoTool):
    async def gate(
        self,
        parameters: dict[str, object],
        context: ToolContext,
    ) -> ToolGateDecision:
        del parameters, context
        return ToolGateDecision.deny("Blocked")


def build_context():
    return RunContext(
        session_runtime=SessionRuntime(
            session_id="session-1",
            run_step_storage=InMemoryRunStepStorage(),
        ),
        run_id="run-1",
        agent_id="agent-1",
        agent_name="agent-1",
    )


@pytest.mark.asyncio
async def test_base_tool_defaults_to_allow_gate() -> None:
    tool = EchoTool()

    decision = await tool.gate({}, ToolContext(session_id="session-1"))

    assert decision.action == "allow"


@pytest.mark.asyncio
async def test_agent_tool_defaults_to_allow_gate() -> None:
    tool = AgentTool(SimpleNamespace(id="child", name="child", description="child"))

    decision = await tool.gate({}, ToolContext(session_id="session-1"))

    assert decision.action == "allow"


@pytest.mark.asyncio
async def test_tool_runtime_allows_default_gate() -> None:
    tool = EchoTool()

    results = await execute_tool_batch(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "echo", "arguments": "{}"},
            }
        ],
        tools_map={tool.name: tool},
        context=build_context(),
    )
    result = results[0]

    assert result.is_success is True
    assert result.content == "ok"
    assert result.termination_reason is None


@pytest.mark.asyncio
async def test_tool_runtime_denies_immediately_when_gate_denies() -> None:
    tool = DenyTool()

    results = await execute_tool_batch(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "echo", "arguments": "{}"},
            }
        ],
        tools_map={tool.name: tool},
        context=build_context(),
    )
    result = results[0]

    assert result.is_success is False
    assert "Blocked" in (result.error or "")
    assert result.termination_reason is None
