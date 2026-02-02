import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agiwo.llm.anthropic import AnthropicModel
from agiwo.llm.base import StreamChunk


@pytest.fixture
def mock_anthropic_client():
    client = AsyncMock()
    return client


@pytest.mark.asyncio
@patch("agiwo.llm.anthropic.settings")
async def test_anthropic_model_arun_stream_basic(mock_settings, mock_anthropic_client):
    mock_settings.anthropic_api_key = None
    model = AnthropicModel(
        id="claude-3-5-sonnet",
        name="claude-3-5-sonnet",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_anthropic_client
    model.model_name = "claude-3-5-sonnet"
    model.max_tokens_to_sample = 100

    mock_event = MagicMock()
    mock_event.type = "content_block_delta"
    mock_event.delta = MagicMock(type="text_delta", text="Hello")
    mock_event.index = 0

    async def async_iter(self):
        yield mock_event
    
    mock_stream = AsyncMock()
    mock_stream.__aiter__ = async_iter
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_stream)

    messages = [{"role": "user", "content": "Hello"}]
    chunks = []
    async for chunk in model.arun_stream(messages):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].content == "Hello"
    mock_anthropic_client.messages.create.assert_called_once()


@pytest.mark.asyncio
@patch("agiwo.llm.anthropic.settings")
async def test_anthropic_model_arun_stream_with_system_prompt(mock_settings, mock_anthropic_client):
    mock_settings.anthropic_api_key = None
    model = AnthropicModel(
        id="claude-3-5-sonnet",
        name="claude-3-5-sonnet",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_anthropic_client
    model.model_name = "claude-3-5-sonnet"
    model.max_tokens_to_sample = 100

    mock_event = MagicMock()
    mock_event.type = "content_block_delta"
    mock_event.delta = MagicMock(type="text_delta", text="Response")
    mock_event.index = 0

    async def async_iter(self):
        yield mock_event
    
    mock_stream = AsyncMock()
    mock_stream.__aiter__ = async_iter
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_stream)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
    chunks = []
    async for chunk in model.arun_stream(messages):
        chunks.append(chunk)

    assert len(chunks) == 1
    call = mock_anthropic_client.messages.create
    call.assert_called_once()
    assert "system" in call.call_args.kwargs


@pytest.mark.asyncio
@patch("agiwo.llm.anthropic.settings")
async def test_anthropic_model_arun_stream_with_usage(mock_settings, mock_anthropic_client):
    mock_settings.anthropic_api_key = None
    model = AnthropicModel(
        id="claude-3-5-sonnet",
        name="claude-3-5-sonnet",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_anthropic_client
    model.model_name = "claude-3-5-sonnet"
    model.max_tokens_to_sample = 100

    from types import SimpleNamespace
    mock_usage = SimpleNamespace()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5
    mock_usage.cache_read_input_tokens = 0
    mock_usage.cache_creation_input_tokens = 0

    mock_message = MagicMock()
    mock_message.usage = mock_usage

    mock_start_event = MagicMock()
    mock_start_event.type = "message_start"
    mock_start_event.message = mock_message

    mock_delta_event = MagicMock()
    mock_delta_event.type = "content_block_delta"
    mock_delta_event.delta = MagicMock(type="text_delta", text="Hello")
    mock_delta_event.index = 0

    async def async_iter(self):
        yield mock_start_event
        yield mock_delta_event
    
    mock_stream = AsyncMock()
    mock_stream.__aiter__ = async_iter
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_stream)

    messages = [{"role": "user", "content": "Hello"}]
    chunks = []
    async for chunk in model.arun_stream(messages):
        chunks.append(chunk)

    assert len(chunks) >= 1
    usage_chunk = next((c for c in chunks if c.usage is not None), None)
    assert usage_chunk is not None
    assert usage_chunk.usage["input_tokens"] == 10
    assert usage_chunk.usage["output_tokens"] == 5


@pytest.mark.asyncio
@patch("agiwo.llm.anthropic.settings")
async def test_anthropic_model_arun_stream_with_tool_use(mock_settings, mock_anthropic_client):
    mock_settings.anthropic_api_key = None
    model = AnthropicModel(
        id="claude-3-5-sonnet",
        name="claude-3-5-sonnet",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_anthropic_client
    model.model_name = "claude-3-5-sonnet"
    model.max_tokens_to_sample = 100

    mock_start_event = MagicMock()
    mock_start_event.type = "content_block_start"
    mock_content_block = MagicMock()
    mock_content_block.type = "tool_use"
    mock_content_block.id = "tool_123"
    mock_content_block.name = "test_function"
    mock_start_event.content_block = mock_content_block
    mock_start_event.index = 0

    mock_delta_event = MagicMock()
    mock_delta_event.type = "content_block_delta"
    mock_delta_event.delta = MagicMock(
        type="input_json_delta", partial_json='{"x": 1}'
    )
    mock_delta_event.index = 0

    mock_stop_event = MagicMock()
    mock_stop_event.type = "content_block_stop"
    mock_stop_event.index = 0

    async def async_iter(self):
        yield mock_start_event
        yield mock_delta_event
        yield mock_stop_event
    
    mock_stream = AsyncMock()
    mock_stream.__aiter__ = async_iter
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_stream)

    messages = [{"role": "user", "content": "Call a function"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "A test function",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    chunks = []
    async for chunk in model.arun_stream(messages, tools=tools):
        chunks.append(chunk)

    tool_call_chunk = next((c for c in chunks if c.tool_calls is not None), None)
    assert tool_call_chunk is not None
    assert len(tool_call_chunk.tool_calls) == 1
    assert tool_call_chunk.tool_calls[0]["function"]["name"] == "test_function"


@pytest.mark.asyncio
@patch("agiwo.llm.anthropic.settings")
async def test_anthropic_model_convert_messages(mock_settings, mock_anthropic_client):
    mock_settings.anthropic_api_key = None
    model = AnthropicModel(
        id="claude-3-5-sonnet",
        name="claude-3-5-sonnet",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_anthropic_client

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User message"},
        {"role": "assistant", "content": "Assistant message"},
    ]

    system_prompt, anthropic_messages = model._convert_messages(messages)

    assert system_prompt == "System prompt"
    assert len(anthropic_messages) == 2
    assert anthropic_messages[0]["role"] == "user"
    assert anthropic_messages[1]["role"] == "assistant"
