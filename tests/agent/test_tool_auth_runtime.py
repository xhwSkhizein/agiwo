from types import SimpleNamespace

import pytest

from agiwo.agent.engine.context import AgentRunContext
from agiwo.agent.lifecycle.session import AgentSessionRuntime
from agiwo.agent.engine.tool_runtime import ResolvedToolCall, ToolRuntime
from agiwo.agent.runtime_tools.agent_tool import AgentTool
from agiwo.agent.storage.base import InMemoryRunStepStorage
from agiwo.agent.storage.session import InMemorySessionStorage
from agiwo.tool.base import BaseTool, ToolGateDecision, ToolResult
from agiwo.tool.context import ToolContext


class EchoTool(BaseTool):
    def get_name(self) -> str:
        return "echo"

    def get_description(self) -> str:
        return "echo"

    def get_parameters(self) -> dict[str, object]:
        return {"type": "object"}

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict[str, object],
        context: ToolContext,
        abort_signal=None,
    ) -> ToolResult:
        del context, abort_signal
        return ToolResult.success(
            tool_name=self.get_name(),
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


def build_context() -> AgentRunContext:
    session_runtime = AgentSessionRuntime(
        session_id="session-1",
        run_step_storage=InMemoryRunStepStorage(),
        session_storage=InMemorySessionStorage(),
    )
    return AgentRunContext(
        session_runtime=session_runtime,
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

    decision = await tool.gate_for_agent({}, build_context())

    assert decision.action == "allow"


@pytest.mark.asyncio
async def test_tool_runtime_allows_default_gate() -> None:
    tool = EchoTool()
    runtime = ToolRuntime([tool])

    outcome = await runtime.execute_resolved(
        ResolvedToolCall(
            raw_call={},
            call_id="call-1",
            tool_name="echo",
            tool=runtime.tools_map["echo"],
            args={"tool_call_id": "call-1"},
        ),
        context=build_context(),
    )

    assert outcome.result.is_success is True
    assert outcome.result.content == "ok"


@pytest.mark.asyncio
async def test_tool_runtime_denies_immediately_when_gate_denies() -> None:
    runtime = ToolRuntime([DenyTool()])

    outcome = await runtime.execute_resolved(
        ResolvedToolCall(
            raw_call={},
            call_id="call-1",
            tool_name="echo",
            tool=runtime.tools_map["echo"],
            args={"tool_call_id": "call-1"},
        ),
        context=build_context(),
    )

    assert outcome.result.is_success is False
    assert "Blocked" in (outcome.result.error or "")
