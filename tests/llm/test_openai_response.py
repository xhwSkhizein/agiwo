import json
from types import SimpleNamespace

import httpx
import pytest
from tenacity import wait_none

from agiwo.llm.openai_response import OpenAIResponsesModel
from agiwo.llm.openai_response_converter import (
    convert_messages_to_responses_input,
    convert_tools_to_responses_tools,
    split_system_instructions,
)
from agiwo.tool.builtin.web_reader.web_reader_tool import WebReaderTool
from agiwo.tool.storage.citation import CitationStoreConfig


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


class _MockHTTPXResponse:
    """Mock httpx.Response for SSE streaming tests."""

    def __init__(self, events):
        self.status_code = 200
        self._events = events

    def _serialize_to_dict(self, obj):
        """Recursively convert SimpleNamespace to dict."""
        if isinstance(obj, SimpleNamespace):
            return {k: self._serialize_to_dict(v) for k, v in vars(obj).items()}
        elif isinstance(obj, list):
            return [self._serialize_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_to_dict(v) for k, v in obj.items()}
        else:
            return obj

    def _generate_sse_lines(self, events):
        """Convert event objects to SSE line format."""
        for event in events:
            yield f"event: {event.type}"
            data_dict = {k: v for k, v in vars(event).items() if k != "type"}
            serialized_data = self._serialize_to_dict(data_dict)
            yield f"data: {json.dumps(serialized_data)}"
            yield ""

    async def aread(self):
        return b""

    async def aiter_lines(self):
        for line in self._generate_sse_lines(self._events):
            yield line


def _collect_keys(payload: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        keys.update(payload.keys())
        for value in payload.values():
            keys.update(_collect_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            keys.update(_collect_keys(item))
    return keys


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
    assert "id" not in items[1]
    assert "status" not in items[1]
    assert items[2]["type"] == "function_call_output"
    assert items[2]["call_id"] == "call_123"
    assert "id" not in items[2]
    assert "status" not in items[2]
    assert items[3]["type"] == "message"
    assert items[3]["role"] == "assistant"
    assert "id" not in items[3]
    assert "status" not in items[3]
    assert items[3]["content"] == [{"type": "output_text", "text": "It is 21C."}]


def test_convert_messages_to_responses_input_multi_turn_history_stays_compatible() -> (
    None
):
    items = convert_messages_to_responses_input(
        [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "ack",
                "id": "resp_1",
                "status": "completed",
                "annotations": [{"type": "citation"}],
            },
            {"role": "user", "content": "use tool"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "weather_lookup",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": {"temp_c": 21}},
            {
                "role": "assistant",
                "content": "done",
                "id": "resp_2",
                "status": "completed",
                "annotations": [],
            },
        ]
    )

    assert [item["type"] for item in items] == [
        "message",
        "message",
        "message",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert items[1] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "ack"}],
    }
    assert items[3] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "weather_lookup",
        "arguments": '{"city": "Paris"}',
    }
    assert items[4] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"temp_c": 21}',
    }
    assert items[5] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "done"}],
    }

    all_keys = _collect_keys(items)
    assert "id" not in all_keys
    assert "status" not in all_keys
    assert "annotations" not in all_keys


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


def test_build_params_multi_turn_payload_matches_provider_compatible_shape() -> None:
    model = OpenAIResponsesModel(
        id="gpt-5.4",
        name="gpt-5.4",
        api_key="test-key",
        base_url="https://example.test/v1",
    )
    web_reader_tool = WebReaderTool(
        citation_store_config=CitationStoreConfig(storage_type="memory")
    )

    params = model._build_params(
        messages=[
            {"role": "system", "content": "system instructions"},
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "ack",
                "id": "resp_1",
                "status": "completed",
                "annotations": [{"type": "citation"}],
            },
            {"role": "user", "content": "use tool"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "web_reader",
                            "arguments": '{"url":"https://example.com"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "body",
            },
        ],
        tools=[web_reader_tool.to_openai_schema()],
    )

    assert params["instructions"] == "system instructions"
    assert [item["type"] for item in params["input"]] == [
        "message",
        "message",
        "message",
        "function_call",
        "function_call_output",
    ]
    input_keys = _collect_keys(params["input"])
    assert "id" not in input_keys
    assert "status" not in input_keys
    assert "annotations" not in input_keys

    assert params["tools"] == [
        {
            "type": "function",
            "name": "web_reader",
            "description": web_reader_tool.description,
            "parameters": web_reader_tool.get_parameters(),
        }
    ]
    tool_keys = _collect_keys(params["tools"])
    assert "oneOf" not in tool_keys


def test_convert_messages_to_responses_input_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="Unsupported message role"):
        convert_messages_to_responses_input([{"role": "developer", "content": "x"}])


def test_convert_tools_to_responses_tools_rejects_non_function_tools() -> None:
    with pytest.raises(ValueError, match="Unsupported tool type"):
        convert_tools_to_responses_tools([{"type": "web_search"}])


@pytest.mark.asyncio
async def test_parse_sse_stream_handles_text_and_reasoning_deltas() -> None:
    """Test _parse_sse_stream correctly parses SSE events."""
    model = OpenAIResponsesModel(
        id="gpt-4.1-mini",
        name="gpt-4.1-mini",
        api_key="test-key",
    )

    # Create a mock httpx response with SSE lines
    class MockResponse:
        status_code = 200

        async def aread(self):
            return b""

        async def aiter_lines(self):
            events = [
                ("response.output_text.delta", {"delta": "Hel"}),
                ("response.reasoning_summary_text.delta", {"delta": "think"}),
                ("response.output_text.delta", {"delta": "lo"}),
                (
                    "response.completed",
                    {
                        "response": {
                            "usage": {
                                "input_tokens": 7,
                                "output_tokens": 3,
                                "total_tokens": 10,
                                "input_tokens_details": {"cached_tokens": 2},
                            },
                            "output": [{"type": "message"}],
                            "incomplete_details": None,
                        }
                    },
                ),
            ]
            for event_type, data in events:
                yield f"event: {event_type}"
                yield f"data: {json.dumps(data)}"
                yield ""

    mock_response = MockResponse()
    tool_calls_state = {}

    chunks = []
    async for event in model._parse_sse_stream(mock_response):
        chunk = model._event_to_chunk(event, tool_calls_state)
        if chunk is not None:
            chunks.append(chunk)

    assert chunks[0].content == "Hel"
    assert chunks[1].reasoning_content == "think"
    assert chunks[2].content == "lo"
    assert chunks[-1].usage is not None
    assert chunks[-1].usage["input_tokens"] == 7
    assert chunks[-1].usage["cache_read_tokens"] == 2
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_parse_sse_stream_handles_function_call_deltas() -> None:
    """Test _parse_sse_stream correctly parses function call events."""
    model = OpenAIResponsesModel(
        id="gpt-4.1-mini",
        name="gpt-4.1-mini",
        api_key="test-key",
    )

    class MockResponse:
        status_code = 200

        async def aread(self):
            return b""

        async def aiter_lines(self):
            events = [
                (
                    "response.output_item.added",
                    {
                        "output_index": 0,
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_123",
                            "name": "weather_lookup",
                            "arguments": "",
                        },
                    },
                ),
                (
                    "response.function_call_arguments.delta",
                    {
                        "output_index": 0,
                        "item_id": "fc_1",
                        "delta": '{"city":"Par',
                    },
                ),
                (
                    "response.function_call_arguments.delta",
                    {
                        "output_index": 0,
                        "item_id": "fc_1",
                        "delta": 'is"}',
                    },
                ),
                (
                    "response.completed",
                    {
                        "response": {
                            "usage": None,
                            "output": [{"type": "function_call"}],
                            "incomplete_details": None,
                        }
                    },
                ),
            ]
            for event_type, data in events:
                yield f"event: {event_type}"
                yield f"data: {json.dumps(data)}"
                yield ""

    mock_response = MockResponse()
    tool_calls_state = {}

    chunks = []
    async for event in model._parse_sse_stream(mock_response):
        chunk = model._event_to_chunk(event, tool_calls_state)
        if chunk is not None:
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
async def test_parse_sse_stream_handles_empty_stream() -> None:
    """Test _parse_sse_stream correctly handles empty stream."""
    model = OpenAIResponsesModel(
        id="gpt-4.1-mini",
        name="gpt-4.1-mini",
        api_key="test-key",
    )

    class MockResponse:
        status_code = 200

        async def aread(self):
            return b""

        async def aiter_lines(self):
            # Empty stream
            return
            yield  # Never reached

    mock_response = MockResponse()
    tool_calls_state = {}

    chunks = []
    async for event in model._parse_sse_stream(mock_response):
        chunk = model._event_to_chunk(event, tool_calls_state)
        if chunk is not None:
            chunks.append(chunk)

    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_parse_sse_stream_handles_failed_event() -> None:
    """Test _parse_sse_stream correctly handles failed event."""
    model = OpenAIResponsesModel(
        id="gpt-4.1-mini",
        name="gpt-4.1-mini",
        api_key="test-key",
    )

    class MockResponse:
        status_code = 200

        async def aread(self):
            return b""

        async def aiter_lines(self):
            events = [
                (
                    "response.failed",
                    {
                        "response": {
                            "error": {"message": "provider failed"},
                        }
                    },
                ),
            ]
            for event_type, data in events:
                yield f"event: {event_type}"
                yield f"data: {json.dumps(data)}"
                yield ""

    mock_response = MockResponse()
    tool_calls_state = {}

    with pytest.raises(RuntimeError, match="provider failed"):
        async for event in model._parse_sse_stream(mock_response):
            chunk = model._event_to_chunk(event, tool_calls_state)
            if chunk is not None:
                pass


@pytest.mark.asyncio
async def test_openai_responses_model_retries_stream_opening(monkeypatch) -> None:
    model = OpenAIResponsesModel(
        id="gpt-4.1-mini",
        name="gpt-4.1-mini",
        api_key="test-key",
        base_url="https://example.test/v1",
    )
    model._open_stream.retry.wait = wait_none()

    attempts = {"stream_calls": 0}

    class _StreamContext:
        def __init__(self, enter_fn):
            self._enter_fn = enter_fn

        async def __aenter__(self):
            return self._enter_fn()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class _MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method, url, json, headers):
            del method, url, json, headers
            attempts["stream_calls"] += 1
            if attempts["stream_calls"] == 1:

                def _raise_connect_error():
                    raise httpx.ConnectError("temporary")

                return _StreamContext(_raise_connect_error)

            return _StreamContext(
                lambda: _MockHTTPXResponse(
                    [
                        _event("response.output_text.delta", delta="Recovered"),
                        _event(
                            "response.completed",
                            response={
                                "usage": None,
                                "output": [{"type": "message"}],
                                "incomplete_details": None,
                            },
                        ),
                    ]
                )
            )

    monkeypatch.setattr("agiwo.llm.openai_response.httpx.AsyncClient", _MockAsyncClient)

    messages = [{"role": "user", "content": "Hello"}]
    chunks = []
    async for chunk in model.arun_stream(messages):
        chunks.append(chunk)

    assert chunks[0].content == "Recovered"
    assert chunks[-1].finish_reason == "stop"
    assert attempts["stream_calls"] == 2
