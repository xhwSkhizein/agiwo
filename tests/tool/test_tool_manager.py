"""Tests for ToolManager.

Tests cover:
- Tool name resolution
- Builtin tool building with caching
- Skill tool building
- Tool merging and finalization
- State management and caching behavior
"""

import pytest
from unittest.mock import Mock, patch

from agiwo.tool.base import BaseTool
from agiwo.tool.manager import ToolManager
import agiwo.tool.manager as manager_module
from agiwo.tool.builtin.registry import BUILTIN_TOOLS
from agiwo.tool.storage.citation import CitationStoreConfig


class MockTool(BaseTool):
    """Mock tool for testing."""

    is_stateless = True
    name = "mock_tool"
    description = "A mock tool for testing"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, context):
        return Mock()


class MockStatefulTool(BaseTool):
    """Mock stateful tool for testing."""

    is_stateless = False
    name = "mock_stateful_tool"
    description = "A mock stateful tool"

    def get_parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, context):
        return Mock()


class TestToolManagerBasics:
    """Test basic ToolManager functionality."""

    def test_list_available_tool_names(self, tool_manager):
        """Test that list_available_tool_names returns builtin tool names."""
        names = tool_manager.list_available_tool_names()
        assert isinstance(names, list)
        assert "bash" in names  # bash is a standard builtin tool

    def test_list_default_tool_names(self, tool_manager):
        """Test that list_default_tool_names returns default tool names."""
        names = tool_manager.list_default_tool_names()
        assert isinstance(names, list)
        assert "bash" in names

    def test_is_tool_available(self, tool_manager):
        """Test tool availability check."""
        assert tool_manager.is_tool_available("bash") is True
        assert tool_manager.is_tool_available("nonexistent_tool") is False

    def test_get_tool_class(self, tool_manager):
        """Test getting tool class by name."""
        tool_cls = tool_manager.get_tool_class("bash")
        assert tool_cls is not None
        assert issubclass(tool_cls, BaseTool)

        assert tool_manager.get_tool_class("nonexistent") is None


class TestResolveToolNames:
    """Test _resolve_tool_names method."""

    def test_resolve_none_returns_defaults(self, tool_manager):
        """When allowed_tools is None, should return DEFAULT_TOOLS keys."""
        result = tool_manager._resolve_tool_names(None)
        assert result == set(tool_manager.list_default_tool_names())

    def test_resolve_explicit_list(self, tool_manager):
        """When allowed_tools is provided, should return those names."""
        result = tool_manager._resolve_tool_names(["bash", "web_search"])
        assert result == {"bash", "web_search"}

    def test_resolve_empty_list(self, tool_manager):
        """Empty list should return empty set."""
        result = tool_manager._resolve_tool_names([])
        assert result == set()


class TestBuildBuiltinTools:
    """Test _build_builtin_tools method."""

    def test_build_skips_unknown_tools(self, tool_manager):
        """Unknown tool names should be skipped gracefully."""
        result = tool_manager._build_builtin_tools({"nonexistent_tool", "bash"})
        # Should only contain bash
        assert len(result) == 1
        assert result[0].name == "bash"

    def test_stateless_tools_are_cached(self, tool_manager):
        """Stateless tools should be cached and reused."""
        # Clear cache first
        tool_manager._tool_cache.clear()

        # Use web_search which is stateless
        result1 = tool_manager._build_builtin_tools({"web_search"})
        assert len(result1) == 1

        # Second call should return same instance from cache
        result2 = tool_manager._build_builtin_tools({"web_search"})
        assert result1[0] is result2[0]  # Same instance

    def test_stateful_tools_not_cached(self, tool_manager):
        """Stateful tools should not be cached - test using a non-existent tool
        that gets patched into BUILTIN_TOOLS temporarily.
        """

        # Create a mock stateful tool class
        class MockStatefulBuiltin(BaseTool):
            is_stateless = False
            name = "mock_stateful_builtin"
            description = "A mock stateful builtin tool"

            def get_parameters(self) -> dict:
                return {"type": "object", "properties": {}}

            async def execute(self, context):
                return Mock()

        # Register it temporarily
        BUILTIN_TOOLS["mock_stateful_builtin"] = MockStatefulBuiltin

        try:
            # Clear any cached instance
            tool_manager._tool_cache.pop("mock_stateful_builtin", None)

            with patch.object(tool_manager, "_create_tool_instance") as mock_create:
                mock_tool = Mock()
                mock_tool.name = "mock_stateful_builtin"
                mock_create.return_value = mock_tool

                # Build the tool twice
                tool_manager._build_builtin_tools({"mock_stateful_builtin"})
                tool_manager._build_builtin_tools({"mock_stateful_builtin"})

                # Should be called twice (not cached)
                assert mock_create.call_count == 2
        finally:
            # Clean up
            BUILTIN_TOOLS.pop("mock_stateful_builtin", None)


class TestGetOrCreateTool:
    """Test _get_or_create_tool method."""

    def test_get_or_create_uses_cache_for_stateless(self, tool_manager):
        """Stateless tools should use cache."""
        tool_manager._tool_cache.clear()

        mock_cls = Mock()
        mock_cls.is_stateless = True
        mock_instance = Mock()
        mock_cls.return_value = mock_instance

        # First call - creates and caches
        result1 = tool_manager._get_or_create_tool("test_tool", mock_cls)
        assert mock_cls.call_count == 1

        # Second call - uses cache
        result2 = tool_manager._get_or_create_tool("test_tool", mock_cls)
        assert mock_cls.call_count == 1  # Not called again
        assert result1 is result2

    def test_get_or_create_fresh_for_stateful(self, tool_manager):
        """Stateful tools should create fresh instances."""
        mock_cls = Mock()
        mock_cls.is_stateless = False
        mock_instance1 = Mock()
        mock_instance2 = Mock()
        mock_cls.side_effect = [mock_instance1, mock_instance2]

        result1 = tool_manager._get_or_create_tool("test_tool", mock_cls)
        result2 = tool_manager._get_or_create_tool("test_tool", mock_cls)

        assert mock_cls.call_count == 2
        assert result1 is mock_instance1
        assert result2 is mock_instance2
        assert result1 is not result2


class TestBuildSkillTool:
    """Test _build_skill_tool method."""

    def test_build_skill_tool_returns_none_for_empty(self, tool_manager):
        """When allowed_skills is empty list (disabled), should return None."""
        result = tool_manager._build_skill_tool([])
        assert result is None

    @patch("agiwo.tool.manager.get_global_skill_manager")
    def test_build_skill_tool_creates_tool_for_none(self, mock_get_sm, tool_manager):
        """When allowed_skills is None (all skills), should create skill tool."""
        mock_skill_manager = Mock()
        mock_skill_tool = Mock()
        mock_skill_manager.create_skill_tool.return_value = mock_skill_tool
        mock_get_sm.return_value = mock_skill_manager

        result = tool_manager._build_skill_tool(None)

        assert result is mock_skill_tool
        mock_skill_manager.create_skill_tool.assert_called_once_with(None)

    @patch("agiwo.tool.manager.get_global_skill_manager")
    def test_build_skill_tool_creates_tool_for_explicit_list(
        self, mock_get_sm, tool_manager
    ):
        """When allowed_skills has items, should create skill tool."""
        mock_skill_manager = Mock()
        mock_skill_tool = Mock()
        mock_skill_manager.create_skill_tool.return_value = mock_skill_tool
        mock_get_sm.return_value = mock_skill_manager

        result = tool_manager._build_skill_tool(["skill1", "skill2"])

        assert result is mock_skill_tool
        mock_skill_manager.create_skill_tool.assert_called_once_with(
            ["skill1", "skill2"]
        )


class TestMergeTools:
    """Test _merge_tools method."""

    def test_merge_builtin_only(self, tool_manager):
        """Merge with only builtin tools."""
        builtin = [Mock(name="tool1"), Mock(name="tool2")]
        result = tool_manager._merge_tools(builtin, [], None, None)
        assert len(result) == 2
        assert result == builtin

    def test_merge_with_extra_tools(self, tool_manager):
        """Merge builtin with extra tools."""
        builtin = [Mock(name="builtin1")]
        extra = [Mock(name="extra1"), Mock(name="extra2")]
        result = tool_manager._merge_tools(builtin, extra, None, None)
        assert len(result) == 3
        assert result[0] == builtin[0]
        assert result[1:] == extra

    def test_merge_with_skill_tool(self, tool_manager):
        """Merge with skill tool."""
        builtin = [Mock(name="builtin1")]
        skill = Mock(name="skill")
        result = tool_manager._merge_tools(builtin, [], skill, None)
        assert len(result) == 2
        assert result[0] == builtin[0]
        assert result[1] == skill

    def test_merge_with_system_tools(self, tool_manager):
        """Merge with system tools."""
        builtin = [Mock(name="builtin1")]
        sys_tools = [Mock(name="sys1"), Mock(name="sys2")]
        result = tool_manager._merge_tools(builtin, [], None, sys_tools)
        assert len(result) == 3
        assert result[0] == builtin[0]
        assert result[1:] == sys_tools

    def test_merge_all_types(self, tool_manager):
        """Merge builtin, extra, skill, and system tools."""
        builtin = [Mock(name="builtin1")]
        extra = [Mock(name="extra1")]
        skill = Mock(name="skill")
        sys_tools = [Mock(name="sys1")]
        result = tool_manager._merge_tools(builtin, extra, skill, sys_tools)
        assert len(result) == 4
        assert result[0] == builtin[0]
        assert result[1] == extra[0]
        assert result[2] == skill
        assert result[3] == sys_tools[0]


class TestFinalizeTools:
    """Test _finalize_tools method."""

    @patch("agiwo.tool.manager.ensure_bash_tool_pair")
    def test_finalize_calls_bash_pair(self, mock_ensure, tool_manager):
        """Should call ensure_bash_tool_pair for consistency."""
        tools = [Mock(name="tool1")]
        mock_ensure.return_value = tools

        result = tool_manager._finalize_tools(tools)

        mock_ensure.assert_called_once_with(tools)
        assert result == tuple(tools)


class TestNormalizeAllowedTools:
    """Test normalize_allowed_tools method."""

    def test_normalize_none_returns_none(self, tool_manager):
        """None should return None."""
        result = tool_manager.normalize_allowed_tools(None)
        assert result is None

    def test_normalize_valid_tools(self, tool_manager):
        """Valid tool names should be returned as list."""
        # Use actual builtin tool names
        builtin_names = tool_manager.list_available_tool_names()[:2]
        result = tool_manager.normalize_allowed_tools(builtin_names)
        assert result == builtin_names

    def test_normalize_keeps_unknown_tools(self, tool_manager):
        """Unknown tool names are kept (may refer to user-supplied custom tools)."""
        result = tool_manager.normalize_allowed_tools(["nonexistent_tool_xyz"])
        assert result == ["nonexistent_tool_xyz"]


class TestGetToolsIntegration:
    """Integration tests for get_tools method."""

    def test_get_tools_default(self, tool_manager):
        """Get tools with default settings."""
        tools = tool_manager.get_tools()
        assert isinstance(tools, tuple)
        # Should have default tools
        default_names = set(tool_manager.list_default_tool_names())
        actual_names = {t.name for t in tools}
        assert default_names <= actual_names

    def test_get_tools_explicit_allowlist(self, tool_manager):
        """Get tools with explicit allowlist."""
        tools = tool_manager.get_tools(allowed_tools=["bash"])
        assert len(tools) >= 1  # At least bash (bash_process may be auto-added)
        tool_names = {t.name for t in tools}
        assert "bash" in tool_names

    def test_get_tools_empty_allowlist(self, tool_manager):
        """Empty allowlist should return no builtin tools."""
        tools = tool_manager.get_tools(allowed_tools=[])
        # Should only have tools added by other means (extra_tools, skills)
        builtin_names = set(tool_manager.list_available_tool_names())
        actual_builtin = {t.name for t in tools} & builtin_names
        assert actual_builtin == set()

    def test_get_tools_with_extra_tools(self, tool_manager):
        """Extra tools included when their name is in allowed_tools."""
        extra = [MockTool()]
        tools = tool_manager.get_tools(
            allowed_tools=["bash", "mock_tool"], extra_tools=extra
        )
        tool_names = {t.name for t in tools}
        assert "mock_tool" in tool_names
        assert "bash" in tool_names

    def test_get_tools_filters_extra_tools_by_allowlist(self, tool_manager):
        """Extra tools not in allowed_tools are filtered out."""
        extra = [MockTool()]
        tools = tool_manager.get_tools(allowed_tools=["bash"], extra_tools=extra)
        tool_names = {t.name for t in tools}
        assert "mock_tool" not in tool_names
        assert "bash" in tool_names

    def test_get_tools_extra_tools_pass_when_allowed_none(self, tool_manager):
        """allowed_tools=None means no filtering on extras."""
        extra = [MockTool()]
        tools = tool_manager.get_tools(allowed_tools=None, extra_tools=extra)
        tool_names = {t.name for t in tools}
        assert "mock_tool" in tool_names

    def test_get_tools_system_tools_bypass_allowlist(self, tool_manager):
        """System tools are never filtered by allowed_tools."""
        tools = tool_manager.get_tools(
            allowed_tools=[], system_tools=[MockStatefulTool()]
        )
        tool_names = {t.name for t in tools}
        assert "mock_stateful_tool" in tool_names


# Fixtures


@pytest.fixture
def tool_manager():
    """Create a fresh ToolManager for testing."""
    manager = ToolManager()
    return manager


@pytest.fixture(autouse=True)
def clear_global_manager():
    """Clear global tool manager before each test."""

    original = manager_module._GLOBAL_TOOL_MANAGER
    manager_module._GLOBAL_TOOL_MANAGER = None
    yield
    manager_module._GLOBAL_TOOL_MANAGER = original


class TestGetGlobalToolManager:
    """Tests for get_global_tool_manager singleton behavior."""

    def test_first_call_creates_manager(self):
        """First call should create the global manager."""
        assert manager_module._GLOBAL_TOOL_MANAGER is None
        manager = manager_module.get_global_tool_manager()
        assert manager is not None
        assert manager_module._GLOBAL_TOOL_MANAGER is manager
        assert manager._citation_store_config is None

    def test_subsequent_calls_return_same_manager(self):
        """Subsequent calls should return the same manager instance."""
        manager1 = manager_module.get_global_tool_manager()
        manager2 = manager_module.get_global_tool_manager()
        assert manager1 is manager2

    def test_upgrade_citation_config(self):
        """If initial manager has no citation config, subsequent call with config should upgrade it."""
        # First call without config
        manager = manager_module.get_global_tool_manager(citation_store_config=None)
        assert manager._citation_store_config is None

        # Pre-populate cache for web_search to simulate real usage
        manager._tool_cache["web_search"] = object()

        # Second call with config should upgrade the existing manager
        # CitationStoreConfig 默认就是 memory 类型
        config = CitationStoreConfig()
        manager2 = manager_module.get_global_tool_manager(citation_store_config=config)

        # Should be the same instance but with updated config
        assert manager2 is manager
        assert manager._citation_store_config is config

        # Web search cache should be cleared so it gets recreated with citation
        assert "web_search" not in manager._tool_cache

    def test_no_upgrade_if_already_has_config(self):
        """If manager already has citation config, new config should be ignored."""
        # First call with config (默认 memory)
        config1 = CitationStoreConfig()
        manager = manager_module.get_global_tool_manager(citation_store_config=config1)
        assert manager._citation_store_config is config1

        # Second call with different config should not change existing config
        # 创建一个不同 db_path 的 config
        config2 = CitationStoreConfig()
        config2.sqlite_db_path = "/tmp/test.db"
        manager2 = manager_module.get_global_tool_manager(citation_store_config=config2)

        assert manager2 is manager
        # Config should remain unchanged
        assert manager._citation_store_config is config1
