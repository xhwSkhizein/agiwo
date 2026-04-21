from dataclasses import dataclass

import pytest

from agiwo.agent import RunOutput
from agiwo.agent.nested.agent_tool import AgentTool
from agiwo.agent.nested.context import AgentToolContext
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunStepStorage


def _make_context(*, metadata: dict | None = None):
    session_runtime = SessionRuntime(
        session_id="sess-1",
        run_step_storage=InMemoryRunStepStorage(),
    )
    return RunContext(
        session_runtime=session_runtime,
        run_id="run-1",
        agent_id="caller-id",
        agent_name="caller",
        metadata=dict(metadata or {}),
    )


def _make_runtime_tool_context(*, metadata: dict | None = None) -> AgentToolContext:
    run_context = _make_context(metadata=metadata)
    return AgentToolContext.from_run_context(
        run_context,
        timeout_at=run_context.timeout_at,
    )


@dataclass
class FakeAgent:
    id: str
    name: str
    description: str = ""
    version: str = "v1"
    last_metadata_updates: dict | None = None

    async def run_child(
        self,
        user_input: str,
        *,
        session_runtime,
        parent_run_id: str,
        parent_depth: int,
        parent_user_id: str | None,
        parent_timeout_at: float | None,
        parent_metadata: dict[str, object],
        instruction: str | None = None,
        system_prompt_override: str | None = None,
        child_allowed_tools: list[str] | None = None,
        metadata_overrides: dict | None = None,
        metadata_updates: dict | None = None,
        abort_signal=None,
    ) -> RunOutput:
        del (
            user_input,
            session_runtime,
            parent_run_id,
            parent_depth,
            parent_user_id,
            parent_timeout_at,
            parent_metadata,
            abort_signal,
            instruction,
            system_prompt_override,
            child_allowed_tools,
            metadata_overrides,
        )
        self.last_metadata_updates = metadata_updates
        return RunOutput(response=f"ran:{self.id}:{self.version}")


@pytest.mark.asyncio
async def test_agent_tool_allows_same_name_when_agent_ids_differ():
    parent_agent = FakeAgent(id="agent-parent", name="shared-name")
    child_agent = FakeAgent(id="agent-child", name="shared-name")

    parent_tool = AgentTool(parent_agent)
    child_tool = AgentTool(child_agent)

    context = _make_runtime_tool_context()
    parent_result = await parent_tool.execute({"task": "first"}, context)

    assert parent_result.is_success is True
    assert parent_agent.last_metadata_updates == {"_call_stack": ["agent-parent"]}

    child_result = await child_tool.execute(
        {"task": "second"},
        _make_runtime_tool_context(metadata={"_call_stack": ["agent-parent"]}),
    )

    assert child_result.is_success is True
    assert child_result.content == "ran:agent-child:v1"


@pytest.mark.asyncio
async def test_agent_tool_resolves_child_spec_at_execution_time():
    agent = FakeAgent(id="agent-dynamic", name="shared-name", description="child")
    tool = AgentTool(agent)
    agent.version = "v2"

    result = await tool.execute({"task": "dynamic"}, _make_runtime_tool_context())

    assert result.is_success is True
    assert result.content == "ran:agent-dynamic:v2"


@pytest.mark.asyncio
async def test_agent_tool_detects_circular_reference_by_agent_id():
    agent = FakeAgent(id="agent-loop", name="shared-name")
    tool = AgentTool(agent)

    context = _make_runtime_tool_context()
    context.metadata["_call_stack"] = ["agent-loop"]

    result = await tool.execute({"task": "loop"}, context)

    assert result.is_success is False
    assert "agent-loop" in result.error


def test_agent_tool_context_can_be_built_from_run_context() -> None:
    runtime_context = AgentToolContext.from_run_context(_make_context(), timeout_at=1.0)

    assert runtime_context.session_id == "sess-1"
    assert runtime_context.parent_run_id == "run-1"
    assert runtime_context.depth == 0
