"""Direct Agent, Scheduler root, and child Run share the same update_plan schema."""

from agiwo.agent import Agent, AgentConfig, AgentOptions
from agiwo.agent.plan import UpdatePlanTool
from agiwo.llm.base import Model, StreamChunk
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.context import ToolContext


class _NeverCalledModel(Model):
    async def arun_stream(self, messages, tools=None):
        del messages, tools
        raise AssertionError("model should not be called")
        yield StreamChunk(content="")  # pragma: no cover


class _SchedulingStubTool(BaseTool):
    """Minimal stand-in for a Scheduler-injected system tool."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"stub {self._name}"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, parameters: dict, context: ToolContext) -> ToolResult:
        del parameters
        return ToolResult.success(
            tool_name=self.name,
            output={},
            tool_call_id=context.tool_call_id,
        )


def _update_plan_schema(tools: list[BaseTool] | tuple[BaseTool, ...]) -> dict:
    for tool in tools:
        if tool.name == "update_plan":
            return tool.to_openai_schema()
    raise AssertionError("update_plan tool not found")


def test_update_plan_schema_matches_across_direct_scheduler_root_and_child() -> None:
    model = _NeverCalledModel(id="mock", name="mock", provider="openai")
    canonical = UpdatePlanTool().to_openai_schema()

    direct = Agent(
        AgentConfig(
            name="direct",
            options=AgentOptions(enable_trajectory_review=False),
        ),
        model=model,
        id="direct-agent",
    )
    assert _update_plan_schema(direct.tools) == canonical

    root = Agent(
        AgentConfig(
            name="root",
            options=AgentOptions(enable_trajectory_review=True),
        ),
        model=model,
        id="root-agent",
    )
    # Scheduler injects scheduling tools; agent-owned update_plan must stay identical.
    root._inject_system_tools(
        [
            _SchedulingStubTool("spawn_child_agent"),
            _SchedulingStubTool("sleep_and_wait"),
        ]
    )
    assert _update_plan_schema(root.tools) == canonical
    assert "update_plan" in {tool.name for tool in root.system_tools}
    assert "spawn_child_agent" in {tool.name for tool in root.system_tools}

    # Child inherits filtered system tools; update_plan schema must still match.
    child_system_tools = [
        tool
        for tool in root.system_tools
        if tool.name not in {"spawn_child_agent", "fork_child_agent"}
    ]
    child = Agent(
        AgentConfig(
            name="child",
            options=AgentOptions(enable_trajectory_review=True),
        ),
        model=model,
        id="child-agent",
    )
    child._inject_system_tools(list(child_system_tools))
    assert _update_plan_schema(child.tools) == canonical
    assert "update_plan" in {tool.name for tool in child.system_tools}
    assert "spawn_child_agent" not in {tool.name for tool in child.system_tools}


def test_update_plan_schema_requires_description_or_status_shape() -> None:
    """Schema encodes create (id+description) vs update (id+status) shapes."""
    params = UpdatePlanTool().get_parameters()
    items = params["properties"]["changes"]["items"]
    assert items["required"] == ["id"]
    any_of = items["anyOf"]
    assert {"id", "description"} == set(any_of[0]["required"])
    assert {"id", "status"} == set(any_of[1]["required"])
    assert (
        "REQUIRED when this id is new"
        in items["properties"]["description"]["description"]
    )
    assert (
        "New milestone ids MUST include"
        in params["properties"]["changes"]["description"]
    )
