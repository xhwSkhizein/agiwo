from dataclasses import dataclass

import pytest

from agiwo.agent import AgentTool
from agiwo.agent.execution_context import ExecutionContext, SessionSequenceCounter
from agiwo.agent.runtime import RunOutput
from agiwo.agent.stream_channel import StreamChannel


def _make_context() -> ExecutionContext:
    return ExecutionContext(
        session_id="sess-1",
        run_id="run-1",
        agent_id="caller-id",
        agent_name="caller",
        channel=StreamChannel(),
        sequence_counter=SessionSequenceCounter(0),
        metadata={},
    )


@dataclass
class FakeAgent:
    id: str
    name: str
    description: str = ""
    last_context: ExecutionContext | None = None

    async def run(self, user_input: str, *, context: ExecutionContext, abort_signal=None) -> RunOutput:
        del user_input, abort_signal
        self.last_context = context
        return RunOutput(response=f"ran:{self.id}")


@pytest.mark.asyncio
async def test_agent_tool_allows_same_name_when_agent_ids_differ():
    parent_agent = FakeAgent(id="agent-parent", name="shared-name")
    child_agent = FakeAgent(id="agent-child", name="shared-name")

    parent_tool = AgentTool(parent_agent)
    child_tool = AgentTool(child_agent)

    context = _make_context()
    parent_result = await parent_tool.execute({"task": "first"}, context)

    assert parent_result.is_success is True
    assert parent_agent.last_context is not None

    child_result = await child_tool.execute(
        {"task": "second"},
        parent_agent.last_context,
    )

    assert child_result.is_success is True
    assert child_result.content == "ran:agent-child"


@pytest.mark.asyncio
async def test_agent_tool_detects_circular_reference_by_agent_id():
    agent = FakeAgent(id="agent-loop", name="shared-name")
    tool = AgentTool(agent)

    context = _make_context()
    context.metadata["_call_stack"] = ["agent-loop"]

    result = await tool.execute({"task": "loop"}, context)

    assert result.is_success is False
    assert "agent-loop" in result.error
