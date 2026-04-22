"""
Test cases for UserInput serialization in storage layers.

Tests that complex UserInput types (UserMessage, list[ContentPart]) are properly
serialized and deserialized in SQLite and MongoDB storage.
"""

import os
import tempfile

import pytest

from agiwo.agent import ChannelContext, ContentPart, ContentType, UserMessage
from agiwo.agent.models.log import RunStarted, UserStepCommitted
from agiwo.agent.models.run import RunIdentity
from agiwo.agent.runtime.context import RunContext
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.agent.storage.sqlite import SQLiteRunLogStorage
from agiwo.agent import MessageRole, StepView


class TestUserInputSerialization:
    """Test UserInput serialization utilities."""

    def test_serialize_deserialize_string(self):
        """String input should remain unchanged."""
        original = "test message"
        serialized = UserMessage.serialize(original)
        assert serialized == original
        deserialized = UserMessage.deserialize(serialized)
        assert deserialized == original

    def test_serialize_deserialize_content_parts(self):
        """list[ContentPart] should be serialized to JSON and back."""
        original = [
            ContentPart(type=ContentType.TEXT, text="Hello"),
            ContentPart(type=ContentType.IMAGE, url="http://example.com/img.jpg"),
        ]
        serialized = UserMessage.serialize(original)
        assert isinstance(serialized, str)
        assert serialized.startswith("{")
        deserialized = UserMessage.deserialize(serialized)
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
        serialized = UserMessage.serialize(original)
        assert isinstance(serialized, str)
        assert serialized.startswith("{")
        deserialized = UserMessage.deserialize(serialized)
        assert isinstance(deserialized, UserMessage)
        assert len(deserialized.content) == 1
        assert deserialized.content[0].text == "Hello World"
        assert deserialized.context.source == "test"
        assert deserialized.context.metadata == {"key": "value"}

    def test_deserialize_plain_string(self):
        """Plain string without JSON markers should remain unchanged."""
        original = "just a plain string"
        deserialized = UserMessage.deserialize(original)
        assert deserialized == original

    def test_deserialize_invalid_json(self):
        """Invalid JSON should remain unchanged."""
        original = "{not valid json"
        deserialized = UserMessage.deserialize(original)
        assert deserialized == original

    def test_to_transport_payload_normalizes_serialized_structured_input(self):
        original = UserMessage(
            content=[
                ContentPart(type=ContentType.TEXT, text="Hello"),
                ContentPart(type=ContentType.IMAGE, url="https://example.com/a.png"),
            ],
            context=ChannelContext(source="api", metadata={"channel": "test"}),
        )

        payload = UserMessage.to_transport_payload(UserMessage.serialize(original))

        assert payload == {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "url": "https://example.com/a.png"},
            ],
            "context": {
                "source": "api",
                "metadata": {"channel": "test"},
            },
        }

    def test_to_storage_value_round_trips_structured_input(self):
        original = [
            ContentPart(type=ContentType.TEXT, text="Hello"),
            ContentPart(type=ContentType.IMAGE, url="https://example.com/a.png"),
        ]

        stored = UserMessage.to_storage_value(original)

        assert isinstance(stored, str)
        assert UserMessage.from_storage_value(stored) == original


class TestSQLiteUserInputStorage:
    """Test UserInput serialization in SQLite storage."""

    @staticmethod
    def _make_context():
        return RunContext(
            identity=RunIdentity(
                run_id="test-run",
                agent_id="test-agent",
                agent_name="test-agent",
            ),
            session_runtime=SessionRuntime(
                session_id="test-session",
                run_log_storage=InMemoryRunLogStorage(),
            ),
        )

    @pytest.mark.asyncio
    async def test_run_log_entry_with_string_user_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteRunLogStorage(db_path=db_path)

            await storage.append_entries(
                [
                    RunStarted(
                        sequence=1,
                        session_id="test-session",
                        run_id="test-run-1",
                        agent_id="test-agent",
                        user_input="simple string input",
                    )
                ]
            )

            loaded = await storage.list_entries(session_id="test-session")
            assert len(loaded) == 1
            assert isinstance(loaded[0], RunStarted)
            assert loaded[0].user_input == "simple string input"

    @pytest.mark.asyncio
    async def test_run_log_entry_with_content_parts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteRunLogStorage(db_path=db_path)

            user_input = [
                ContentPart(type=ContentType.TEXT, text="Hello"),
                ContentPart(type=ContentType.IMAGE, url="http://example.com/img.jpg"),
            ]
            await storage.append_entries(
                [
                    RunStarted(
                        sequence=1,
                        session_id="test-session",
                        run_id="test-run-2",
                        agent_id="test-agent",
                        user_input=user_input,
                    )
                ]
            )

            loaded = await storage.list_entries(session_id="test-session")
            assert len(loaded) == 1
            assert isinstance(loaded[0], RunStarted)
            assert isinstance(loaded[0].user_input, list)
            assert len(loaded[0].user_input) == 2
            assert loaded[0].user_input[0].type == ContentType.TEXT
            assert loaded[0].user_input[0].text == "Hello"
            assert loaded[0].user_input[1].type == ContentType.IMAGE
            assert loaded[0].user_input[1].url == "http://example.com/img.jpg"

    @pytest.mark.asyncio
    async def test_run_log_entry_with_user_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteRunLogStorage(db_path=db_path)

            context = ChannelContext(source="api", metadata={"channel": "test"})
            user_input = UserMessage(
                content=[ContentPart(type=ContentType.TEXT, text="Test message")],
                context=context,
            )
            await storage.append_entries(
                [
                    RunStarted(
                        sequence=1,
                        session_id="test-session",
                        run_id="test-run-3",
                        agent_id="test-agent",
                        user_input=user_input,
                    )
                ]
            )

            loaded = await storage.list_entries(session_id="test-session")
            assert len(loaded) == 1
            assert isinstance(loaded[0], RunStarted)
            assert isinstance(loaded[0].user_input, UserMessage)
            assert len(loaded[0].user_input.content) == 1
            assert loaded[0].user_input.content[0].text == "Test message"
            assert loaded[0].user_input.context is not None
            assert loaded[0].user_input.context.source == "api"
            assert loaded[0].user_input.context.metadata == {"channel": "test"}

    @pytest.mark.asyncio
    async def test_committed_step_entry_with_user_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteRunLogStorage(db_path=db_path)

            step = StepView.user(
                self._make_context(),
                sequence=1,
                user_input=UserMessage(
                    content=[ContentPart(type=ContentType.TEXT, text="hello")],
                    context=ChannelContext(source="api", metadata={"channel": "test"}),
                ),
            )

            await storage.append_entries(
                [
                    UserStepCommitted(
                        sequence=step.sequence,
                        session_id=step.session_id,
                        run_id=step.run_id,
                        agent_id=step.agent_id or "",
                        step_id=step.id,
                        role=MessageRole.USER,
                        user_input=step.user_input,
                        content=step.content,
                        created_at=step.created_at,
                    )
                ]
            )

            retrieved = await storage.list_step_views(session_id="test-session")
            assert len(retrieved) == 1
            assert isinstance(retrieved[0].user_input, UserMessage)
            assert retrieved[0].user_input.context is not None
            assert retrieved[0].user_input.context.source == "api"

    @pytest.mark.asyncio
    async def test_save_run_log_entry_with_user_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            storage = SQLiteRunLogStorage(db_path=db_path)

            entry = RunStarted(
                sequence=1,
                session_id="test-session",
                run_id="test-run",
                agent_id="test-agent",
                user_input=UserMessage(
                    content=[ContentPart(type=ContentType.TEXT, text="hello")],
                    context=ChannelContext(source="api", metadata={"channel": "test"}),
                ),
            )

            await storage.append_entries([entry])
            loaded = await storage.list_entries(session_id="test-session")

            assert len(loaded) == 1
            assert isinstance(loaded[0], RunStarted)
            assert isinstance(loaded[0].user_input, UserMessage)
            assert loaded[0].user_input.context is not None
            assert loaded[0].user_input.context.source == "api"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
