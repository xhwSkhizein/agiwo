from types import SimpleNamespace
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agiwo.llm.openai import OpenAIModel


@pytest.fixture
def mock_openai_client():
    client = AsyncMock()
    return client


@patch("agiwo.llm.openai.get_settings")
def test_openai_model_name_only_defaults_id(mock_get_settings):
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None

    model = OpenAIModel(name="gpt-5.4", api_key="test-key")

    assert model.id == "gpt-5.4"
    assert model.name == "gpt-5.4"


def test_openai_model_requires_name_or_id():
    with pytest.raises(
        TypeError,
        match="OpenAIModel requires at least one of id or name",
    ):
        OpenAIModel()


@pytest.mark.asyncio
@patch("agiwo.llm.openai.get_settings")
async def test_openai_model_arun_stream_basic(mock_get_settings, mock_openai_client):
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None
    model = OpenAIModel(
        id="gpt-4",
        name="gpt-4",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_output_tokens=100,
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
@patch("agiwo.llm.openai.get_settings")
async def test_openai_model_arun_stream_with_usage(
    mock_get_settings, mock_openai_client
):
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None
    model = OpenAIModel(
        id="gpt-4",
        name="gpt-4",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_output_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_openai_client

    mock_usage = SimpleNamespace()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    mock_chunk = MagicMock()
    mock_chunk.usage = mock_usage
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="World", tool_calls=None),
            finish_reason="stop",
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
    assert chunks[0].content == "World"
    assert chunks[0].finish_reason == "stop"
    assert chunks[0].usage is not None
    assert chunks[0].usage["input_tokens"] == 10
    assert chunks[0].usage["output_tokens"] == 5


@pytest.mark.asyncio
@patch("agiwo.llm.openai.get_settings")
async def test_openai_model_arun_stream_with_tools(
    mock_get_settings, mock_openai_client
):
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None
    model = OpenAIModel(
        id="gpt-4",
        name="gpt-4",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_output_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_openai_client

    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.type = "function"
    mock_tool_call.function = MagicMock(name="test_function", arguments='{"x": 1}')

    mock_chunk = MagicMock()
    mock_chunk.usage = None
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None,
                tool_calls=[mock_tool_call],
            ),
            finish_reason=None,
        )
    ]

    async def async_iter(self):
        yield mock_chunk

    mock_stream = AsyncMock()
    mock_stream.__aiter__ = async_iter
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

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

    assert len(chunks) == 1
    assert chunks[0].tool_calls is not None
    assert len(chunks[0].tool_calls) == 1
    call = mock_openai_client.chat.completions.create
    call.assert_called_once()
    assert "tools" in call.call_args.kwargs


@pytest.mark.asyncio
@patch("agiwo.llm.openai.get_settings")
async def test_openai_model_arun_stream_multiple_chunks(
    mock_get_settings, mock_openai_client
):
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None
    model = OpenAIModel(
        id="gpt-4",
        name="gpt-4",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_output_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_openai_client

    chunks_data = [
        ("Hello", None),
        (" ", None),
        ("World", "stop"),
    ]

    mock_chunks = []
    for content, finish_reason in chunks_data:
        mock_chunk = MagicMock()
        mock_chunk.usage = None
        mock_chunk.choices = [
            MagicMock(
                delta=MagicMock(content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ]
        mock_chunks.append(mock_chunk)

    async def async_iter(self):
        for chunk in mock_chunks:
            yield chunk

    mock_stream = AsyncMock()
    mock_stream.__aiter__ = async_iter
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    messages = [{"role": "user", "content": "Say hello world"}]
    result_chunks = []
    async for chunk in model.arun_stream(messages):
        result_chunks.append(chunk)

    assert len(result_chunks) == 3
    assert result_chunks[0].content == "Hello"
    assert result_chunks[1].content == " "
    assert result_chunks[2].content == "World"
    assert result_chunks[2].finish_reason == "stop"


@pytest.mark.asyncio
@patch("agiwo.llm.openai.get_settings")
async def test_openai_model_arun_stream_rejects_empty_stream(
    mock_get_settings, mock_openai_client
):
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None
    model = OpenAIModel(
        id="qwen3.6-plus-preview",
        name="qwen3.6-plus-preview",
        api_key="test-key",
        base_url="https://embed.o-ai.tech",
    )
    model.client = mock_openai_client

    async def async_iter(self):
        if False:
            yield None

    mock_stream = AsyncMock()
    mock_stream.__aiter__ = async_iter
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    messages = [{"role": "user", "content": "Hello"}]
    with pytest.raises(RuntimeError, match="returned no chunks"):
        async for _ in model.arun_stream(messages):
            pass
