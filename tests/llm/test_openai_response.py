from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agiwo.llm.openai_response import OpenAIResponsesModel
from agiwo.llm.openai_response_converter import (
    convert_messages_to_responses_input,
    convert_tools_to_responses_tools,
    split_system_instructions,
)


def _event(event_type: str, **kwargs):
    return SimpleNamespace(type=event_type, **kwargs)


class _MockResponseStream:
    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self._events:
            yield item


def test_split_system_instructions_collects_system_messages() -> None:
    instructions, remaining = split_system_instructions(
        [
            {"role": "system", "content": "You are careful."},
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "Prefer tools."},
        ]
    )

    assert instructions == "You are careful.\n\nPrefer tools."
    assert remaining == [{"role": "user", "content": "hello"}]


def test_convert_messages_to_responses_input_maps_user_assistant_tool_flow() -> None:
    items = convert_messages_to_responses_input(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "weather_lookup",
                            "arguments": '{"city":"Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": '{"temp_c":21}'},
            {"role": "assistant", "content": "It is 21C."},
        ]
    )

    assert items[0] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "weather?"}],
    }
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_123"
    assert items[2]["type"] == "function_call_output"
    assert items[2]["call_id"] == "call_123"
    assert items[3]["type"] == "message"
    assert items[3]["role"] == "assistant"
    assert items[3]["content"] == [
        {"type": "output_text", "text": "It is 21C.", "annotations": []}
    ]


def test_convert_tools_to_responses_tools_flattens_function_schema() -> None:
    tools = convert_tools_to_responses_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "weather_lookup",
                    "description": "Look up weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
    )

    assert tools == [
        {
            "type": "function",
            "name": "weather_lookup",
            "description": "Look up weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]


def test_convert_messages_to_responses_input_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="Unsupported message role"):
        convert_messages_to_responses_input([{"role": "developer", "content": "x"}])


def test_convert_tools_to_responses_tools_rejects_non_function_tools() -> None:
    with pytest.raises(ValueError, match="Unsupported tool type"):
        convert_tools_to_responses_tools([{"type": "web_search"}])


@pytest.mark.asyncio
@patch("agiwo.llm.openai_response.get_settings")
async def test_openai_response_model_streams_text_reasoning_and_usage(
    mock_get_settings,
) -> None:
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None

    model = OpenAIResponsesModel(
        id="gpt-4.1-mini",
        name="gpt-4.1-mini",
        api_key="test-key",
    )
    model.client = AsyncMock()

    stream_events = [
        _event("response.output_text.delta", delta="Hel"),
        _event("response.reasoning_summary_text.delta", delta="think"),
        _event("response.output_text.delta", delta="lo"),
        _event(
            "response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=7,
                    output_tokens=3,
                    total_tokens=10,
                    input_tokens_details=SimpleNamespace(cached_tokens=2),
                ),
                output=[SimpleNamespace(type="message")],
                incomplete_details=None,
            ),
        ),
    ]

    mock_stream = _MockResponseStream(stream_events)
    model.client.responses.create = AsyncMock(return_value=mock_stream)

    chunks = []
    async for chunk in model.arun_stream([{"role": "user", "content": "hello"}]):
        chunks.append(chunk)

    assert chunks[0].content == "Hel"
    assert chunks[1].reasoning_content == "think"
    assert chunks[2].content == "lo"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage["input_tokens"] == 7
    assert chunks[-1].usage["cache_read_tokens"] == 2
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
@patch("agiwo.llm.openai_response.get_settings")
async def test_openai_response_model_streams_function_call_deltas(
    mock_get_settings,
) -> None:
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None

    model = OpenAIResponsesModel(
        id="gpt-4.1-mini",
        name="gpt-4.1-mini",
        api_key="test-key",
    )
    model.client = AsyncMock()

    stream_events = [
        _event(
            "response.output_item.added",
            output_index=0,
            item=SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_123",
                name="weather_lookup",
                arguments="",
            ),
        ),
        _event(
            "response.function_call_arguments.delta",
            output_index=0,
            item_id="fc_1",
            delta='{"city":"Par',
        ),
        _event(
            "response.function_call_arguments.delta",
            output_index=0,
            item_id="fc_1",
            delta='is"}',
        ),
        _event(
            "response.completed",
            response=SimpleNamespace(
                usage=None,
                output=[SimpleNamespace(type="function_call")],
                incomplete_details=None,
            ),
        ),
    ]

    mock_stream = _MockResponseStream(stream_events)
    model.client.responses.create = AsyncMock(return_value=mock_stream)

    chunks = []
    async for chunk in model.arun_stream([{"role": "user", "content": "weather"}]):
        chunks.append(chunk)

    assert chunks[0].tool_calls == [
        {
            "index": 0,
            "id": "call_123",
            "type": "function",
            "function": {"name": "weather_lookup", "arguments": '{"city":"Par'},
        }
    ]
    assert chunks[1].tool_calls == [
        {
            "index": 0,
            "id": "call_123",
            "type": "function",
            "function": {"arguments": 'is"}'},
        }
    ]
    assert chunks[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
@patch("agiwo.llm.openai_response.get_settings")
async def test_openai_response_model_rejects_empty_stream(
    mock_get_settings,
) -> None:
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None

    model = OpenAIResponsesModel(
        id="gpt-4.1-mini",
        name="gpt-4.1-mini",
        api_key="test-key",
    )
    model.client = AsyncMock()

    mock_stream = _MockResponseStream([])
    model.client.responses.create = AsyncMock(return_value=mock_stream)

    with pytest.raises(RuntimeError, match="returned no chunks"):
        async for _ in model.arun_stream([{"role": "user", "content": "hello"}]):
            pass


@pytest.mark.asyncio
@patch("agiwo.llm.openai_response.get_settings")
async def test_openai_response_model_raises_failed_event_message(
    mock_get_settings,
) -> None:
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None

    model = OpenAIResponsesModel(
        id="gpt-4.1-mini",
        name="gpt-4.1-mini",
        api_key="test-key",
    )
    model.client = AsyncMock()

    mock_stream = _MockResponseStream(
        [
            _event(
                "response.failed",
                response=SimpleNamespace(
                    error=SimpleNamespace(message="provider failed"),
                ),
            )
        ]
    )
    model.client.responses.create = AsyncMock(return_value=mock_stream)

    with pytest.raises(RuntimeError, match="provider failed"):
        async for _ in model.arun_stream([{"role": "user", "content": "hello"}]):
            pass
