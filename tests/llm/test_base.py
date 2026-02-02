import pytest
from agiwo.llm.base import Model, StreamChunk


def test_stream_chunk():
    chunk = StreamChunk(content="Hello", finish_reason="stop")
    assert chunk.content == "Hello"
    assert chunk.finish_reason == "stop"
    assert chunk.reasoning_content is None
    assert chunk.tool_calls is None
    assert chunk.usage is None


def test_stream_chunk_all_fields():
    chunk = StreamChunk(
        content="Hello",
        reasoning_content="Thinking...",
        tool_calls=[{"id": "1", "function": {"name": "test"}}],
        usage={"input_tokens": 10, "output_tokens": 5},
        finish_reason="stop",
    )
    assert chunk.content == "Hello"
    assert chunk.reasoning_content == "Thinking..."
    assert len(chunk.tool_calls) == 1
    assert chunk.usage["input_tokens"] == 10
    assert chunk.finish_reason == "stop"


def test_model_validation_temperature():
    class TestModel(Model):
        async def arun_stream(self, messages, tools=None):
            yield StreamChunk()

    with pytest.raises(ValueError, match="temperature must be between"):
        TestModel(
            id="test",
            name="test",
            temperature=3.0,
            top_p=1.0,
            max_tokens=100,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )


def test_model_validation_max_tokens():
    class TestModel(Model):
        async def arun_stream(self, messages, tools=None):
            yield StreamChunk()

    with pytest.raises(ValueError, match="max_tokens must be at least"):
        TestModel(
            id="test",
            name="test",
            temperature=0.7,
            top_p=1.0,
            max_tokens=0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
