import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agiwo.llm.deepseek import DeepseekModel, parse_dsml_function_calls


def test_parse_dsml_function_calls_single_tool():
    """Test parsing DSML function_calls format with single tool."""
    content = """<｜DSML｜function_calls>
<｜DSML｜invoke name="bash">
<｜DSML｜parameter name="command" string="true">ls -la /Users/hongv/workspace/</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

    result = parse_dsml_function_calls(content)

    assert result is not None
    assert len(result) == 1
    assert result[0]["index"] == 0
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "bash"
    parsed_args = json.loads(result[0]["function"]["arguments"])
    assert parsed_args["command"] == "ls -la /Users/hongv/workspace/"
    assert result[0]["id"].startswith("call_")


def test_parse_dsml_function_calls_multiple_tools():
    """Test parsing DSML function_calls format with multiple tools."""
    content = """<｜DSML｜function_calls>
<｜DSML｜invoke name="bash">
<｜DSML｜parameter name="command" string="true">ls -la</｜DSML｜parameter>
</｜DSML｜invoke>
<｜DSML｜invoke name="current_time">
<｜DSML｜parameter name="timezone" string="true">UTC</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

    result = parse_dsml_function_calls(content)

    assert result is not None
    assert len(result) == 2
    assert result[0]["function"]["name"] == "bash"
    args_0 = json.loads(result[0]["function"]["arguments"])
    assert args_0["command"] == "ls -la"
    assert result[1]["function"]["name"] == "current_time"
    args_1 = json.loads(result[1]["function"]["arguments"])
    assert args_1["timezone"] == "UTC"


def test_parse_dsml_function_calls_no_dsml():
    """Test parsing content without DSML format returns None."""
    content = "This is just a regular response without any function calls."

    result = parse_dsml_function_calls(content)

    assert result is None


def test_parse_dsml_function_calls_empty():
    """Test parsing empty content returns None."""
    result = parse_dsml_function_calls("")

    assert result is None


@pytest.mark.asyncio
@patch("agiwo.llm.deepseek.settings")
async def test_deepseek_model_arun_stream_with_dsml_function_calls(
    mock_settings, mock_openai_client
):
    """Test that DSML function_calls in content are parsed into tool_calls."""
    mock_settings.deepseek_api_key = None
    model = DeepseekModel(
        id="deepseek-reasoner",
        name="deepseek-reasoner",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_output_tokens=100,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    model.client = mock_openai_client

    dsml_content = """<｜DSML｜function_calls>
<｜DSML｜invoke name="bash">
<｜DSML｜parameter name="command" string="true">ls -la</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

    mock_chunk = MagicMock()
    mock_chunk.usage = None
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=dsml_content, tool_calls=None),
            finish_reason="stop",
        )
    ]

    async def async_iter(self):
        yield mock_chunk

    mock_stream = AsyncMock()
    mock_stream.__aiter__ = async_iter
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    tools = [
        {
            "type": "function",
            "function": {"name": "bash", "description": "Execute bash command"},
        }
    ]
    messages = [{"role": "user", "content": "List files"}]
    chunks = []
    async for chunk in model.arun_stream(messages, tools=tools):
        chunks.append(chunk)

    # Should have 2 chunks: original content + parsed tool_calls
    assert len(chunks) == 2
    assert chunks[0].content == dsml_content
    assert chunks[1].tool_calls is not None
    assert len(chunks[1].tool_calls) == 1
    assert chunks[1].tool_calls[0]["function"]["name"] == "bash"


@pytest.mark.asyncio
@patch("agiwo.llm.deepseek.settings")
async def test_deepseek_model_arun_stream_with_normal_tool_calls(
    mock_settings, mock_openai_client
):
    """Test that normal tool_calls are not affected by DSML parsing."""
    mock_settings.deepseek_api_key = None
    model = DeepseekModel(
        id="deepseek-chat",
        name="deepseek-chat",
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
            delta=MagicMock(
                content=None,
                tool_calls=[
                    MagicMock(
                        index=0,
                        id="call_123",
                        type="function",
                        function=MagicMock(name="bash", arguments='{"command":"ls"}'),
                    )
                ],
            ),
            finish_reason="tool_calls",
        )
    ]

    async def async_iter(self):
        yield mock_chunk

    mock_stream = AsyncMock()
    mock_stream.__aiter__ = async_iter
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_stream)

    tools = [
        {
            "type": "function",
            "function": {"name": "bash", "description": "Execute bash command"},
        }
    ]
    messages = [{"role": "user", "content": "List files"}]
    chunks = []
    async for chunk in model.arun_stream(messages, tools=tools):
        chunks.append(chunk)

    # Should only have 1 chunk (no DSML parsing needed)
    assert len(chunks) == 1
    assert chunks[0].tool_calls is not None


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
@patch("agiwo.llm.deepseek.settings")
async def test_deepseek_model_preprocess_messages_new_turn(
    mock_settings, mock_openai_client
):
    mock_settings.deepseek_api_key = None
    model = DeepseekModel(
        id="deepseek-reasoner",
        name="deepseek-reasoner",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_output_tokens=100,
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
async def test_deepseek_model_preprocess_messages_thinking_mode(
    mock_settings, mock_openai_client
):
    mock_settings.deepseek_api_key = None
    model = DeepseekModel(
        id="deepseek-reasoner",
        name="deepseek-reasoner",
        api_key="test-key",
        temperature=0.7,
        top_p=1.0,
        max_output_tokens=100,
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
        max_output_tokens=100,
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
