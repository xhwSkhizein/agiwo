"""Tests for default memory hooks."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agiwo.agent import Agent
from agiwo.agent import AgentConfig
from agiwo.agent.hooks import (
    DefaultMemoryHook,
    HookPhase,
    HookRegistry,
    filter_relevant_memories,
    observe,
    transform,
)
from agiwo.agent.models.run import MemoryRecord
from agiwo.llm.base import Model


class _FakeEncoding:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


class TestDefaultMemoryHook:
    """Tests for DefaultMemoryHook."""

    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            memory_dir = workspace / "MEMORY"
            memory_dir.mkdir()
            yield workspace

    def test_init_default(self):
        hook = DefaultMemoryHook()
        assert hook._top_k is not None

    def test_init_custom_top_k(self):
        hook = DefaultMemoryHook(top_k=10)
        assert hook._top_k == 10

    def test_resolve_workspace(self, temp_workspace):
        hook = DefaultMemoryHook()
        context = SimpleNamespace(agent_name="test-agent")

        with patch("os.path.expanduser", return_value=str(temp_workspace.parent)):
            result = hook._resolve_workspace(context)
            assert result is not None
            assert "test-agent" in str(result)

    def test_resolve_workspace_no_agent_name(self):
        hook = DefaultMemoryHook()
        context = SimpleNamespace(agent_name=None, agent_id=None)

        result = hook._resolve_workspace(context)
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieve_memories_empty_query(self):
        hook = DefaultMemoryHook()
        context = SimpleNamespace()

        result = await hook.retrieve_memories("", context)
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_memories_short_query(self):
        hook = DefaultMemoryHook()
        context = SimpleNamespace()

        result = await hook.retrieve_memories("hi", context)
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_memories_no_workspace(self):
        hook = DefaultMemoryHook()
        context = SimpleNamespace(agent_name=None, agent_id=None)

        result = await hook.retrieve_memories("test query", context)
        assert result == []


class TestAgentAutoInjectMemoryHook:
    """Tests for Agent auto-injecting default memory hook."""

    def test_agent_auto_injects_memory_hook(self):
        """Test that Agent auto-injects default memory hook when hooks not provided."""
        mock_model = MagicMock(spec=Model)
        mock_model.id = "test-model"

        agent = Agent(
            AgentConfig(name="test-agent", description="Test agent"),
            model=mock_model,
        )

        assert agent.hooks.has_phase(HookPhase.ASSEMBLE_CONTEXT)
        assert agent.id.startswith("test-agent-")

    def test_agent_respects_custom_memory_hook(self):
        """Test that Agent respects user-provided memory hook."""
        mock_model = MagicMock(spec=Model)
        custom_hook = AsyncMock(return_value={"memories": []})

        hooks = HookRegistry(
            [
                transform(
                    HookPhase.ASSEMBLE_CONTEXT,
                    "custom_memory_retrieve",
                    custom_hook,
                )
            ]
        )
        agent = Agent(
            AgentConfig(name="test-agent", description="Test agent"),
            model=mock_model,
            hooks=hooks,
        )

        assert agent.hooks.has_handler("custom_memory_retrieve")

    def test_agent_preserves_other_hooks_when_injecting(self):
        """Test that auto-inject preserves other user hooks."""
        mock_model = MagicMock(spec=Model)
        custom_step_hook = AsyncMock()

        hooks = HookRegistry(
            [
                observe(
                    HookPhase.AFTER_STEP_COMMIT,
                    "custom_step_hook",
                    custom_step_hook,
                )
            ]
        )
        agent = Agent(
            AgentConfig(name="test-agent", description="Test agent"),
            model=mock_model,
            hooks=hooks,
        )

        assert agent.hooks.has_handler("custom_step_hook")
        assert agent.hooks.has_phase(HookPhase.ASSEMBLE_CONTEXT)

    def test_agent_empty_hooks_gets_memory_hook(self):
        """Test that empty hooks object gets memory hook injected."""
        mock_model = MagicMock(spec=Model)

        hooks = HookRegistry()
        agent = Agent(
            AgentConfig(name="test-agent", description="Test agent"),
            model=mock_model,
            hooks=hooks,
        )

        assert agent.hooks.has_phase(HookPhase.ASSEMBLE_CONTEXT)


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

        hook = DefaultMemoryHook(
            embedding_provider="disabled",
            top_k=3,
            root_path=temp_workspace.parent,
        )

        # Mock context with agent_name that maps to our temp workspace
        context = SimpleNamespace(
            agent_name=str(temp_workspace.name),
            agent_id=str(temp_workspace.name),
        )

        with patch(
            "agiwo.memory.chunker._resolve_encoding",
            return_value=_FakeEncoding(),
        ):
            # Clear any cached stores
            hook._memory_service._stores.clear()

            result = await hook.retrieve_memories("Python programming", context)

            assert len(result) >= 1
            assert any("Python" in r.content for r in result)

    @pytest.mark.asyncio
    async def test_memory_record_format(self, temp_workspace):
        """Test that MemoryRecord has correct format."""
        memory_dir = temp_workspace / "MEMORY"
        (memory_dir / "test.md").write_text("Important note about machine learning.")

        hook = DefaultMemoryHook(
            embedding_provider="disabled",
            top_k=3,
            root_path=temp_workspace.parent,
        )

        context = SimpleNamespace(
            agent_name=str(temp_workspace.name),
            agent_id=str(temp_workspace.name),
        )

        with patch(
            "agiwo.memory.chunker._resolve_encoding",
            return_value=_FakeEncoding(),
        ):
            hook._memory_service._stores.clear()

            result = await hook.retrieve_memories("machine learning", context)

            if result:
                record = result[0]
                assert isinstance(record, MemoryRecord)
                assert record.content is not None
                assert record.relevance_score is not None
                assert record.source is not None
                assert "chunk_id" in record.metadata
                assert "start_line" in record.metadata


def test_filter_relevant_memories_drops_low_score_duplicates_and_history_overlap():
    messages = [
        {
            "role": "assistant",
            "content": "You already know Alpha architecture decision",
        },
        {"role": "user", "content": "What changed?"},
    ]
    memories = [
        MemoryRecord(content="Alpha architecture decision", relevance_score=0.95),
        MemoryRecord(content="Alpha architecture decision", relevance_score=0.90),
        MemoryRecord(content="Beta follow-up task", relevance_score=0.75),
        MemoryRecord(content="Low score item", relevance_score=0.40),
    ]

    filtered = filter_relevant_memories(messages, memories)

    assert [memory.content for memory in filtered] == ["Beta follow-up task"]
