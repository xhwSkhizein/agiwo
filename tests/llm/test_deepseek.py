import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agiwo.llm.deepseek import DeepseekModel


@pytest.fixture
def mock_openai_client():
    client = AsyncMock()
    return client


@pytest.mark.asyncio
@patch("agiwo.llm.deepseek.settings")
async def test_deepseek_model_arun_stream_basic(mock_settings, mock_openai_client):
    mock_settings.deepseek_api_key = None
    model = DeepseekModel(
        id="deepseek-chat",
        name="deepseek-chat",
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
@patch("agiwo.llm.deepseek.settings")
async def test_deepseek_model_preprocess_messages_new_turn(mock_settings, mock_openai_client):
    mock_settings.deepseek_api_key = None
    model = DeepseekModel(
        id="deepseek-reasoner",
        name="deepseek-reasoner",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_openai_client

    messages = [
        {
            "role": "assistant",
            "content": "Previous response",
            "reasoning_content": "Previous reasoning",
        },
        {"role": "user", "content": "New question"},
    ]

    processed = model._preprocess_messages_for_thinking_mode(messages)

    assert len(processed) == 2
    assert processed[0]["role"] == "assistant"
    assert "reasoning_content" not in processed[0]
    assert processed[1]["role"] == "user"


@pytest.mark.asyncio
@patch("agiwo.llm.deepseek.settings")
async def test_deepseek_model_preprocess_messages_thinking_mode(mock_settings, mock_openai_client):
    mock_settings.deepseek_api_key = None
    model = DeepseekModel(
        id="deepseek-reasoner",
        name="deepseek-reasoner",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_openai_client

    messages = [
        {"role": "assistant", "content": "Response without reasoning"},
        {"role": "assistant", "content": "Another response"},
    ]

    processed = model._preprocess_messages_for_thinking_mode(messages)

    assert len(processed) == 2
    for msg in processed:
        if msg["role"] == "assistant":
            assert "reasoning_content" in msg
            assert msg["reasoning_content"] is None


@pytest.mark.asyncio
@patch("agiwo.llm.deepseek.settings")
async def test_deepseek_model_with_reasoning_content(mock_settings, mock_openai_client):
    mock_settings.deepseek_api_key = None
    model = DeepseekModel(
        id="deepseek-reasoner",
        name="deepseek-reasoner",
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
                reasoning_content="Thinking process",
                tool_calls=None,
            ),
            finish_reason=None,
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
    assert chunks[0].reasoning_content == "Thinking process"
