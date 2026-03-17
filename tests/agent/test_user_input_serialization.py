"""
Test cases for UserInput serialization in storage layers.

Tests that complex UserInput types (UserMessage, list[ContentPart]) are properly
serialized and deserialized in SQLite and MongoDB storage.
"""

import os
import sqlite3
import pytest
import tempfile

from agiwo.agent import (
    ContentPart,
    ContentType,
    Run,
    RunStatus,
    RunMetrics,
    StepRecord,
    UserMessage,
    ChannelContext,
    serialize_user_input,
    deserialize_user_input,
)
from agiwo.agent.storage.sqlite import SQLiteRunStepStorage
from tests.utils.agent_context import build_agent_context


class TestUserInputSerialization:
    """Test UserInput serialization utilities."""

    def test_serialize_deserialize_string(self):
        """String input should remain unchanged."""
        original = "test message"
        serialized = serialize_user_input(original)
        assert serialized == original
        deserialized = deserialize_user_input(serialized)
        assert deserialized == original

    def test_serialize_deserialize_content_parts(self):
        """list[ContentPart] should be serialized to JSON and back."""
        original = [
            ContentPart(type=ContentType.TEXT, text="Hello"),
            ContentPart(type=ContentType.IMAGE, url="http://example.com/img.jpg"),
        ]
        serialized = serialize_user_input(original)
        assert isinstance(serialized, str)
        assert serialized.startswith("{")
        deserialized = deserialize_user_input(serialized)
        assert isinstance(deserialized, list)
        assert len(deserialized) == 2
        assert deserialized[0].type == ContentType.TEXT
        assert deserialized[0].text == "Hello"
        assert deserialized[1].type == ContentType.IMAGE
        assert deserialized[1].url == "http://example.com/img.jpg"

    def test_serialize_deserialize_user_message(self):
        """UserMessage should be serialized to JSON and back."""
        context = ChannelContext(source="test", metadata={"key": "value"})
        original = UserMessage(
            content=[ContentPart(type=ContentType.TEXT, text="Hello World")],
            context=context,
        )
        serialized = serialize_user_input(original)
        assert isinstance(serialized, str)
        assert serialized.startswith("{")
        deserialized = deserialize_user_input(serialized)
        assert isinstance(deserialized, UserMessage)
        assert len(deserialized.content) == 1
        assert deserialized.content[0].text == "Hello World"
        assert deserialized.context.source == "test"
        assert deserialized.context.metadata == {"key": "value"}

    def test_deserialize_plain_string(self):
        """Plain string without JSON markers should remain unchanged."""
        original = "just a plain string"
        deserialized = deserialize_user_input(original)
        assert deserialized == original

    def test_deserialize_invalid_json(self):
        """Invalid JSON should remain unchanged."""
        original = "{not valid json"
        deserialized = deserialize_user_input(original)
        assert deserialized == original


class TestSQLiteUserInputStorage:
    """Test UserInput serialization in SQLite storage."""

    @staticmethod
    def _make_context():
        return build_agent_context(
            session_id="test-session",
            run_id="test-run",
            agent_id="test-agent",
            agent_name="test-agent",
        )

    @pytest.mark.asyncio
    async def test_save_run_with_string_user_input(self):
        """Test saving run with string user_input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteRunStepStorage(db_path=db_path)

            run = Run(
                id="test-run-1",
                agent_id="test-agent",
                session_id="test-session",
                user_input="simple string input",
                status=RunStatus.COMPLETED,
            )
            run.metrics = RunMetrics()
            await storage.save_run(run)

            retrieved = await storage.get_run("test-run-1")
            assert retrieved is not None
            assert retrieved.user_input == "simple string input"

    @pytest.mark.asyncio
    async def test_save_run_with_content_parts(self):
        """Test saving run with list[ContentPart] user_input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteRunStepStorage(db_path=db_path)

            user_input = [
                ContentPart(type=ContentType.TEXT, text="Hello"),
                ContentPart(type=ContentType.IMAGE, url="http://example.com/img.jpg"),
            ]
            run = Run(
                id="test-run-2",
                agent_id="test-agent",
                session_id="test-session",
                user_input=user_input,
                status=RunStatus.COMPLETED,
            )
            run.metrics = RunMetrics()
            await storage.save_run(run)

            retrieved = await storage.get_run("test-run-2")
            assert retrieved is not None
            assert isinstance(retrieved.user_input, list)
            assert len(retrieved.user_input) == 2
            assert retrieved.user_input[0].type == ContentType.TEXT
            assert retrieved.user_input[0].text == "Hello"
            assert retrieved.user_input[1].type == ContentType.IMAGE
            assert retrieved.user_input[1].url == "http://example.com/img.jpg"

    @pytest.mark.asyncio
    async def test_save_run_with_user_message(self):
        """Test saving run with UserMessage user_input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteRunStepStorage(db_path=db_path)

            context = ChannelContext(source="api", metadata={"channel": "test"})
            user_input = UserMessage(
                content=[ContentPart(type=ContentType.TEXT, text="Test message")],
                context=context,
            )
            run = Run(
                id="test-run-3",
                agent_id="test-agent",
                session_id="test-session",
                user_input=user_input,
                status=RunStatus.COMPLETED,
            )
            run.metrics = RunMetrics()
            await storage.save_run(run)

            retrieved = await storage.get_run("test-run-3")
            assert retrieved is not None
            assert isinstance(retrieved.user_input, UserMessage)
            assert len(retrieved.user_input.content) == 1
            assert retrieved.user_input.content[0].text == "Test message"
            assert retrieved.user_input.context is not None
            assert retrieved.user_input.context.source == "api"
            assert retrieved.user_input.context.metadata == {"channel": "test"}

    @pytest.mark.asyncio
    async def test_save_step_with_user_input_creates_required_columns(self):
        """Fresh SQLite databases should persist step.user_input successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteRunStepStorage(db_path=db_path)

            step = StepRecord.user(
                self._make_context(),
                sequence=1,
                user_input=UserMessage(
                    content=[ContentPart(type=ContentType.TEXT, text="hello")],
                    context=ChannelContext(source="api", metadata={"channel": "test"}),
                ),
            )

            await storage.save_step(step)

            retrieved = await storage.get_last_step("test-session")
            assert retrieved is not None
            assert isinstance(retrieved.user_input, UserMessage)
            assert retrieved.user_input.context is not None
            assert retrieved.user_input.context.source == "api"

            with sqlite3.connect(db_path) as conn:
                columns = {
                    row[1] for row in conn.execute("PRAGMA table_info(steps)").fetchall()
                }
            assert "user_input" in columns
            assert "content_for_user" in columns

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
