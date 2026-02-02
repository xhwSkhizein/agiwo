import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agiwo.llm.nvidia import NvidiaModel
from agiwo.llm.base import StreamChunk


@pytest.fixture
def mock_openai_client():
    client = AsyncMock()
    return client


@pytest.mark.asyncio
@patch("agiwo.llm.nvidia.settings")
async def test_nvidia_model_arun_stream_basic(mock_settings, mock_openai_client):
    mock_settings.nvidia_api_key = None
    model = NvidiaModel(
        id="meta/llama-3.1-8b-instruct",
        name="meta/llama-3.1-8b-instruct",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_openai_client

    mock_chunk = MagicMock()
    mock_chunk.usage = None
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", tool_calls=None),
            finish_reason=None,
        )
    ]

    async def async_iter(self):
        yield mock_chunk
    
    mock_stream = AsyncMock()
    mock_stream.__aiter__ = async_iter
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    messages = [{"role": "user", "content": "Hello"}]
    chunks = []
    async for chunk in model.arun_stream(messages):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"
    mock_openai_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
@patch("agiwo.llm.nvidia.settings")
async def test_nvidia_model_arun_stream_with_reasoning(mock_settings, mock_openai_client):
    mock_settings.nvidia_api_key = None
    model = NvidiaModel(
        id="meta/llama-3.1-8b-instruct",
        name="meta/llama-3.1-8b-instruct",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_openai_client

    mock_chunk = MagicMock()
    mock_chunk.usage = None
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="Response",
                reasoning_content="Reasoning content",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]

    async def async_iter(self):
        yield mock_chunk
    
    mock_stream = AsyncMock()
    mock_stream.__aiter__ = async_iter
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    messages = [{"role": "user", "content": "Think about this"}]
    chunks = []
    async for chunk in model.arun_stream(messages):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].content == "Response"
    assert chunks[0].reasoning_content == "Reasoning content"
    assert chunks[0].finish_reason == "stop"
