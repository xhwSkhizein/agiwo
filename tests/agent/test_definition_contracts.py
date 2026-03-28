import inspect
import pytest
from pathlib import Path
import agiwo.agent.agent as agent_module
from agiwo.agent import Agent
from agiwo.agent import AgentConfig, AgentOptions
from agiwo.agent.definition import resolve_child_definition
from agiwo.agent import AgentHooks
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


def _build_agent() -> Agent:
    return Agent(
        config=AgentConfig(
            name="definition-agent",
            description="definition contract test",
            system_prompt="Base prompt",
            options=AgentOptions(enable_skill=False),
        ),
        id="definition-agent",
        model=MockModel(id="mock", name="mock", provider="openai"),
        tools=[DummyTool()],
        hooks=AgentHooks(),
    )


@pytest.mark.asyncio
async def test_create_child_agent_applies_overrides() -> None:
    agent = _build_agent()

    clone = await agent.create_child_agent(
        child_id="child-agent",
        instruction="Handle only child work",
        exclude_tool_names={"dummy_tool"},
    )

    tool_names = {tool.get_name() for tool in clone.tools}
    assert clone.id == "child-agent"
    assert "Handle only child work" in clone.config.system_prompt
    assert "dummy_tool" not in tool_names
    assert clone.config.options.enable_termination_summary is True


@pytest.mark.asyncio
async def test_create_child_agent_clones_runtime_configuration() -> None:
    agent = _build_agent()

    clone = await agent.create_child_agent(
        child_id="child-agent",
        instruction="Same instruction",
        exclude_tool_names={"dummy_tool"},
    )

    assert clone.id == "child-agent"
    assert clone.config.system_prompt.startswith("Base prompt")
    assert "Same instruction" in clone.config.system_prompt
    assert {tool.get_name() for tool in clone.tools} == {
        tool.get_name() for tool in agent.tools if tool.get_name() != "dummy_tool"
    }


@pytest.mark.asyncio
async def test_create_child_agent_and_child_resolution_stay_in_sync() -> None:
    agent = _build_agent()

    resolved = resolve_child_definition(
        agent,
        instruction="Handle only child work",
        system_prompt_override=None,
        exclude_tool_names={"dummy_tool"},
    )
    clone = await agent.create_child_agent(
        child_id="child-agent",
        instruction="Handle only child work",
        exclude_tool_names={"dummy_tool"},
    )

    assert clone.config.system_prompt == resolved.config.system_prompt
    assert {tool.get_name() for tool in clone.tools} == {
        tool.get_name() for tool in resolved.tools
    }
    assert clone.config.options.enable_termination_summary is True


def test_agent_resolves_definition_once_during_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_calls = 0
    real_resolve = agent_module.resolve_agent_definition

    def counted_resolve(*args, **kwargs):
        nonlocal resolve_calls
        resolve_calls += 1
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(agent_module, "resolve_agent_definition", counted_resolve)

    Agent(
        config=AgentConfig(
            name="definition-agent",
            description="definition contract test",
            system_prompt="Base prompt",
            options=AgentOptions(enable_skill=True),
        ),
        id="definition-agent",
        model=MockModel(id="mock", name="mock", provider="openai"),
        tools=[DummyTool()],
        hooks=AgentHooks(),
    )

    assert resolve_calls == 1


def test_agent_constructor_does_not_expose_disabled_sdk_tool_names() -> None:
    assert "disabled_sdk_tool_names" not in inspect.signature(Agent.__init__).parameters


def test_agent_config_owns_disabled_sdk_tool_names_for_default_sdk_tools() -> None:
    agent = Agent(
        config=AgentConfig(
            name="definition-agent",
            description="definition contract test",
            system_prompt="Base prompt",
            options=AgentOptions(enable_skill=False),
            disabled_sdk_tool_names={"bash"},
        ),
        id="definition-agent",
        model=MockModel(id="mock", name="mock", provider="openai"),
        hooks=AgentHooks(),
    )

    tool_names = {tool.get_name() for tool in agent.tools}

    assert "bash" not in tool_names
    assert "bash_process" not in tool_names


def test_guardrails_treat_runtime_package_as_internal_execution_module() -> None:
    repo_guard = Path("/Users/hongv/workspace/agiwo/scripts/repo_guard.py").read_text(
        encoding="utf-8"
    )
    import_linter = Path(
        "/Users/hongv/workspace/agiwo/lint/importlinter_agiwo.ini"
    ).read_text(encoding="utf-8")
    agents_doc = Path("/Users/hongv/workspace/agiwo/AGENTS.md").read_text(
        encoding="utf-8"
    )

    assert "agiwo.agent.runtime" in import_linter
    assert '"agiwo.agent.runtime"' in repo_guard
    assert "agiwo.agent.runtime" in agents_doc


def test_import_linter_guardrails_allow_public_agent_facade_transitively() -> None:
    import_linter = Path(
        "/Users/hongv/workspace/agiwo/lint/importlinter_agiwo.ini"
    ).read_text(encoding="utf-8")

    scheduler_contract = """[importlinter:contract:8]
name = scheduler-runtime-must-not-depend-on-agent-internals
type = forbidden
source_modules =
    agiwo.scheduler.engine
    agiwo.scheduler.runner
    agiwo.scheduler.runtime_state
    agiwo.scheduler.tool_control
forbidden_modules =
    agiwo.agent.run_loop
    agiwo.agent.tool_executor
    agiwo.agent.compaction
    agiwo.agent.llm_caller
    agiwo.agent.prompt
    agiwo.agent.runtime
allow_indirect_imports = True"""
    storage_contract = """[importlinter:contract:9]
name = agent-storage-factory-must-not-depend-on-observability
type = forbidden
source_modules =
    agiwo.agent.storage.factory
forbidden_modules =
    agiwo.observability
allow_indirect_imports = True"""

    assert scheduler_contract in import_linter
    assert storage_contract in import_linter
