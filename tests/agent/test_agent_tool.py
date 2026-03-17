from dataclasses import dataclass

import pytest

from agiwo.agent import AgentTool
from agiwo.agent.execution import ChildAgentSpec
from agiwo.agent.inner.context import AgentRunContext
from agiwo.agent.runtime import RunOutput
from tests.utils.agent_context import build_agent_context


def _make_context(*, metadata: dict | None = None) -> AgentRunContext:
    return build_agent_context(
        session_id="sess-1",
        run_id="run-1",
        agent_id="caller-id",
        agent_name="caller",
        metadata=metadata,
    )


@dataclass
class FakeAgent:
    id: str
    name: str
    description: str = ""
    version: str = "v1"
    last_metadata_updates: dict | None = None
    last_spec: object | None = None

    def derive_child_spec(self, *, child_id: str) -> ChildAgentSpec:
        return ChildAgentSpec(
            agent_id=child_id,
            agent_name=self.name,
            description=f"{self.description}:{self.version}",
        )

    async def run_child(
        self,
        user_input: str,
        *,
        spec,
        parent_context: AgentRunContext,
        metadata_updates: dict | None = None,
        abort_signal=None,
    ) -> RunOutput:
        del user_input, parent_context, abort_signal
        self.last_spec = spec
        self.last_metadata_updates = metadata_updates
        return RunOutput(response=f"ran:{self.id}:{self.version}")


@pytest.mark.asyncio
async def test_agent_tool_allows_same_name_when_agent_ids_differ():
    parent_agent = FakeAgent(id="agent-parent", name="shared-name")
    child_agent = FakeAgent(id="agent-child", name="shared-name")

    parent_tool = AgentTool(parent_agent)
    child_tool = AgentTool(child_agent)

    context = _make_context()
    parent_result = await parent_tool.execute({"task": "first"}, context)

    assert parent_result.is_success is True
    assert parent_agent.last_metadata_updates == {"_call_stack": ["agent-parent"]}

    child_result = await child_tool.execute(
        {"task": "second"},
        _make_context(metadata={"_call_stack": ["agent-parent"]}),
    )

    assert child_result.is_success is True
    assert child_result.content == "ran:agent-child:v1"


@pytest.mark.asyncio
async def test_agent_tool_resolves_child_spec_at_execution_time():
    agent = FakeAgent(id="agent-dynamic", name="shared-name", description="child")
    tool = AgentTool(agent)
    agent.version = "v2"

    result = await tool.execute({"task": "dynamic"}, _make_context())

    assert result.is_success is True
    assert result.content == "ran:agent-dynamic:v2"
    assert agent.last_spec.description == "child:v2"


@pytest.mark.asyncio
async def test_agent_tool_detects_circular_reference_by_agent_id():
    agent = FakeAgent(id="agent-loop", name="shared-name")
    tool = AgentTool(agent)

    context = _make_context()
    context.metadata["_call_stack"] = ["agent-loop"]

    result = await tool.execute({"task": "loop"}, context)

    assert result.is_success is False
    assert "agent-loop" in result.error
