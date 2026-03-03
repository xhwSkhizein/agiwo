"""Tests for default memory hooks."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agiwo.agent import Agent, AgentHooks
from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.memory_hooks import DefaultMemoryHook, create_default_memory_hooks
from agiwo.agent.schema import MemoryRecord, UserInput
from agiwo.llm.base import Model
from agiwo.tool.builtin.config import MemoryConfig


class TestDefaultMemoryHook:
    """Tests for DefaultMemoryHook."""

    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            memory_dir = workspace / "MEMORY"
            memory_dir.mkdir()
            yield workspace

    def test_init_default_config(self):
        hook = DefaultMemoryHook()
        assert hook._config is not None

    def test_init_custom_config(self):
        config = MemoryConfig(top_k=10)
        hook = DefaultMemoryHook(config)
        assert hook._config.top_k == 10

    def test_resolve_workspace(self, temp_workspace):
        hook = DefaultMemoryHook()
        context = MagicMock(spec=ExecutionContext)
        context.agent_name = "test-agent"

        with patch("os.path.expanduser", return_value=str(temp_workspace.parent)):
            result = hook._resolve_workspace(context)
            assert result is not None
            assert "test-agent" in str(result)

    def test_resolve_workspace_no_agent_name(self):
        hook = DefaultMemoryHook()
        context = MagicMock(spec=ExecutionContext)
        context.agent_name = None

        result = hook._resolve_workspace(context)
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieve_memories_empty_query(self):
        hook = DefaultMemoryHook()
        context = MagicMock(spec=ExecutionContext)

        result = await hook.retrieve_memories("", context)
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_memories_short_query(self):
        hook = DefaultMemoryHook()
        context = MagicMock(spec=ExecutionContext)

        result = await hook.retrieve_memories("hi", context)
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_memories_no_workspace(self):
        hook = DefaultMemoryHook()
        context = MagicMock(spec=ExecutionContext)
        context.agent_name = None

        result = await hook.retrieve_memories("test query", context)
        assert result == []


class TestCreateDefaultMemoryHooks:
    """Tests for create_default_memory_hooks helper."""

    def test_create_default_hooks(self):
        hooks = create_default_memory_hooks()
        assert hooks.on_memory_retrieve is not None

    def test_create_with_custom_config(self):
        config = MemoryConfig(top_k=10)
        hooks = create_default_memory_hooks(config)
        assert hooks.on_memory_retrieve is not None


class TestAgentAutoInjectMemoryHook:
    """Tests for Agent auto-injecting default memory hook."""

    def test_agent_auto_injects_memory_hook(self):
        """Test that Agent auto-injects default memory hook when hooks not provided."""
        mock_model = MagicMock(spec=Model)
        mock_model.id = "test-model"

        agent = Agent(
            name="test-agent",
            description="Test agent",
            model=mock_model,
        )

        assert agent.hooks.on_memory_retrieve is not None
        assert agent.id.startswith("test-agent-")

    def test_agent_respects_custom_memory_hook(self):
        """Test that Agent respects user-provided memory hook."""
        mock_model = MagicMock(spec=Model)
        custom_hook = AsyncMock(return_value=[])

        hooks = AgentHooks(on_memory_retrieve=custom_hook)
        agent = Agent(
            name="test-agent",
            description="Test agent",
            model=mock_model,
            hooks=hooks,
        )

        assert agent.hooks.on_memory_retrieve is custom_hook

    def test_agent_preserves_other_hooks_when_injecting(self):
        """Test that auto-inject preserves other user hooks."""
        mock_model = MagicMock(spec=Model)
        custom_step_hook = AsyncMock()

        hooks = AgentHooks(on_step=custom_step_hook)
        agent = Agent(
            name="test-agent",
            description="Test agent",
            model=mock_model,
            hooks=hooks,
        )

        assert agent.hooks.on_step is custom_step_hook
        assert agent.hooks.on_memory_retrieve is not None

    def test_agent_empty_hooks_gets_memory_hook(self):
        """Test that empty hooks object gets memory hook injected."""
        mock_model = MagicMock(spec=Model)

        hooks = AgentHooks()
        agent = Agent(
            name="test-agent",
            description="Test agent",
            model=mock_model,
            hooks=hooks,
        )

        assert agent.hooks.on_memory_retrieve is not None


class TestMemoryHookIntegration:
    """Integration tests for memory hook with actual search."""

    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            memory_dir = workspace / "MEMORY"
            memory_dir.mkdir()
            yield workspace

    @pytest.mark.asyncio
    async def test_memory_hook_retrieves_from_files(self, temp_workspace):
        """Test that memory hook retrieves content from MEMORY files."""
        memory_dir = temp_workspace / "MEMORY"

        (memory_dir / "notes.md").write_text(
            "This is important information about Python programming."
        )

        config = MemoryConfig(
            embedding_provider="disabled",
            top_k=3,
        )
        hook = DefaultMemoryHook(config)

        # Mock context with agent_name that maps to our temp workspace
        context = MagicMock(spec=ExecutionContext)
        context.agent_name = str(temp_workspace.name)

        # Patch the workspace resolution to return our temp directory
        with patch.object(
            hook, "_resolve_workspace", return_value=temp_workspace
        ):
            # Clear any cached stores
            hook._stores.clear()

            result = await hook.retrieve_memories("Python programming", context)

            assert len(result) >= 1
            assert any("Python" in r.content for r in result)

    @pytest.mark.asyncio
    async def test_memory_record_format(self, temp_workspace):
        """Test that MemoryRecord has correct format."""
        memory_dir = temp_workspace / "MEMORY"
        (memory_dir / "test.md").write_text("Important note about machine learning.")

        config = MemoryConfig(embedding_provider="disabled", top_k=3)
        hook = DefaultMemoryHook(config)

        context = MagicMock(spec=ExecutionContext)
        context.agent_name = str(temp_workspace.name)

        with patch.object(hook, "_resolve_workspace", return_value=temp_workspace):
            hook._stores.clear()

            result = await hook.retrieve_memories("machine learning", context)

            if result:
                record = result[0]
                assert isinstance(record, MemoryRecord)
                assert record.content is not None
                assert record.relevance_score is not None
                assert record.source is not None
                assert "chunk_id" in record.metadata
                assert "start_line" in record.metadata
