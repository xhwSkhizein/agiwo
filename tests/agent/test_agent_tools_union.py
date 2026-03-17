"""Tests for Agent tools union logic with DEFAULT_TOOLS."""

import pytest
from agiwo.agent import Agent, AgentConfig
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.agent.inner.context import AgentRunContext
from agiwo.utils.abort_signal import AbortSignal
from agiwo.tool.builtin.registry import DEFAULT_TOOLS
from agiwo.llm.base import StreamChunk


class MockTool(BaseTool):
    """A mock tool for testing."""

    def __init__(self, name: str = "mock_tool"):
        self._name = name

    def get_name(self) -> str:
        return self._name

    def get_description(self) -> str:
        return f"Mock tool: {self._name}"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict,
        context: AgentRunContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        return ToolResult(
            tool_name=self._name,
            tool_call_id="test",
            input_args=parameters,
            content="mock result",
            output={"ok": True},
        )


class MockModel:
    """A mock model for testing Agent."""

    async def arun_stream(self, messages, tools=None):
        """Mock stream that yields a simple response."""
        del messages, tools
        # Yield a simple text chunk
        yield StreamChunk(text="Hello from mock model")


@pytest.fixture
def mock_model():
    """Create a mock model."""
    return MockModel()


class TestAgentToolsUnion:
    """Test Agent tools union logic with DEFAULT_TOOLS."""

    @staticmethod
    def _make_agent(mock_model, tools=None) -> Agent:
        return Agent(
            AgentConfig(name="test-agent", description="Test agent"),
            model=mock_model,
            tools=tools,
        )

    def test_agent_loads_default_tools_when_no_tools_passed(self, mock_model):
        """Test that Agent loads DEFAULT_TOOLS when tools=None."""
        agent = self._make_agent(mock_model, tools=None)

        # Should have all DEFAULT_TOOLS
        assert len(agent.tools) == len(DEFAULT_TOOLS)
        tool_names = {t.get_name() for t in agent.tools}
        assert tool_names == set(DEFAULT_TOOLS.keys())

    def test_agent_loads_default_tools_when_empty_list_passed(self, mock_model):
        """Test that Agent loads DEFAULT_TOOLS when tools=[]."""
        agent = self._make_agent(mock_model, tools=[])

        # Should have all DEFAULT_TOOLS
        assert len(agent.tools) == len(DEFAULT_TOOLS)
        tool_names = {t.get_name() for t in agent.tools}
        assert tool_names == set(DEFAULT_TOOLS.keys())

    def test_agent_merges_user_tools_with_default_tools(self, mock_model):
        """Test that Agent merges user tools with DEFAULT_TOOLS (union)."""
        user_tool = MockTool(name="user_custom_tool")
        agent = self._make_agent(mock_model, tools=[user_tool])

        # Should have user tool + all DEFAULT_TOOLS
        tool_names = {t.get_name() for t in agent.tools}
        assert "user_custom_tool" in tool_names
        assert all(name in tool_names for name in DEFAULT_TOOLS.keys())

    def test_user_tool_takes_precedence_over_default(self, mock_model):
        """Test that user tool takes precedence when name conflicts."""
        # Create a custom tool with same name as a default tool
        default_tool_name = list(DEFAULT_TOOLS.keys())[0]
        custom_tool = MockTool(name=default_tool_name)
        agent = self._make_agent(mock_model, tools=[custom_tool])

        # Should only have one tool with that name (user's version)
        tool_names = [t.get_name() for t in agent.tools]
        assert tool_names.count(default_tool_name) == 1

        # Verify the custom tool instance is used
        matching_tools = [t for t in agent.tools if t.get_name() == default_tool_name]
        assert len(matching_tools) == 1
        assert matching_tools[0] is custom_tool

    def test_agent_tools_count_with_merge(self, mock_model):
        """Test correct tool count when merging user tools with defaults."""
        user_tools = [MockTool(name="tool1"), MockTool(name="tool2")]
        agent = self._make_agent(mock_model, tools=user_tools)

        # Count should be: user tools + default tools (no duplicates)
        expected_count = len(user_tools) + len(DEFAULT_TOOLS)
        assert len(agent.tools) == expected_count

    def test_bash_tool_auto_loaded_when_no_tools(self, mock_model):
        """Test that BashTool is auto-loaded via @default_enable when no tools passed."""
        agent = self._make_agent(mock_model, tools=None)

        # Should have bash tool loaded
        tool_names = {t.get_name() for t in agent.tools}
        assert "bash" in tool_names
        assert "bash_process" in tool_names

    def test_bash_tool_auto_loaded_with_other_tools(self, mock_model):
        """Test that BashTool is auto-loaded even when other tools are passed."""
        user_tool = MockTool(name="my_tool")
        agent = self._make_agent(mock_model, tools=[user_tool])

        # Should have both user tool and bash tool
        tool_names = {t.get_name() for t in agent.tools}
        assert "my_tool" in tool_names
        assert "bash" in tool_names
        assert "bash_process" in tool_names
