import pytest

from agiwo.agent import AgentConfig, AgentHooks, ChildAgentSpec
from agiwo.agent.lifecycle.definition import AgentDefinitionRuntime
from agiwo.agent.options import AgentOptions
from agiwo.llm.base import Model
from agiwo.tool.base import BaseTool, ToolResult


class MockModel(Model):
    async def arun_stream(self, messages, tools=None):
        if False:
            yield None


class DummyTool(BaseTool):
    def get_name(self) -> str:
        return "dummy_tool"

    def get_description(self) -> str:
        return "Dummy tool for definition contracts"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(self, parameters, context, abort_signal=None) -> ToolResult:
        del parameters, context, abort_signal
        return ToolResult.success(
            tool_name=self.get_name(),
            content="ok",
            start_time=0.0,
        )


def _build_runtime() -> AgentDefinitionRuntime:
    return AgentDefinitionRuntime(
        config=AgentConfig(
            name="definition-agent",
            description="definition contract test",
            system_prompt="Base prompt",
            options=AgentOptions(enable_skill=False),
        ),
        agent_id="definition-agent",
        provided_tools=[DummyTool()],
        hooks=AgentHooks(),
    )


@pytest.mark.asyncio
async def test_child_definition_materialization_applies_overrides() -> None:
    runtime = _build_runtime()
    definition = await runtime.snapshot_child_definition(
        model=MockModel(id="mock", name="mock"),
        spec=ChildAgentSpec(
            agent_id="child-agent",
            agent_name="child-agent",
            description="child",
            instruction="Handle only child work",
            exclude_tool_names=frozenset({"dummy_tool"}),
        ),
    )

    tool_names = {tool.get_name() for tool in definition.tools}
    assert definition.agent_id == "child-agent"
    assert "Handle only child work" in definition.system_prompt
    assert "dummy_tool" not in tool_names
    assert definition.options.enable_termination_summary is True


@pytest.mark.asyncio
async def test_scheduler_child_clone_matches_child_materialization() -> None:
    runtime = _build_runtime()
    model = MockModel(id="mock", name="mock")
    spec = ChildAgentSpec(
        agent_id="child-agent",
        agent_name="definition-agent",
        description="child",
        instruction="Same instruction",
        exclude_tool_names=frozenset({"dummy_tool"}),
    )

    definition = await runtime.snapshot_child_definition(model=model, spec=spec)
    clone = runtime.build_scheduler_child_clone(
        child_id="child-agent",
        instruction="Same instruction",
        exclude_tool_names={"dummy_tool"},
    )

    assert clone.agent_id == definition.agent_id
    assert clone.config.system_prompt.startswith("Base prompt")
    assert "Same instruction" in clone.config.system_prompt
    assert {tool.get_name() for tool in clone.tools} == {
        tool.get_name() for tool in definition.tools
    }
    assert clone.hooks is not runtime.hooks
