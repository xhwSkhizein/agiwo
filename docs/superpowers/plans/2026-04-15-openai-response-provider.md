# OpenAI Responses Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `openai-response` provider that uses OpenAI Responses API while keeping the SDK public `Model.arun_stream(messages, tools) -> AsyncIterator[StreamChunk]` contract and current agent/tool path unchanged.

**Architecture:** Implement a dedicated `OpenAIResponsesModel` plus a private converter that maps canonical SDK messages/tools into Responses request items and maps Responses stream events back into normalized `StreamChunk` values. Keep all OpenAI Responses semantics inside the provider boundary and leave `openai` / `openai-compatible` on `chat.completions`.

**Tech Stack:** Python 3.11+, `openai` async SDK, pytest, pytest-asyncio, existing `Model` / `StreamChunk` abstractions

---

## File Map

- Create: `agiwo/llm/openai_response.py`
  - Own the `OpenAIResponsesModel` provider, client setup, request assembly, stream normalization, retry behavior.
- Create: `agiwo/llm/openai_response_converter.py`
  - Own provider-private conversion from canonical SDK `messages` / chat-style tools to Responses `input` / function tools.
- Modify: `agiwo/config/settings.py`
  - Add `openai-response` to `ModelProvider` and provider registries.
- Modify: `agiwo/llm/factory.py`
  - Register the new provider in `PROVIDER_SPECS`.
- Modify: `agiwo/llm/__init__.py`
  - Export `OpenAIResponsesModel`.
- Modify: `tests/llm/test_factory.py`
  - Add factory construction coverage for `openai-response`.
- Create: `tests/llm/test_openai_response.py`
  - Add provider-level conversion and streaming tests.
- Modify: `tests/agent/test_run_contracts.py`
  - Add one contract test proving upper layers consume `openai-response`-style tool-call chunks unchanged.
- Modify: `docs/api/model.md`
  - Document the new provider and its scope.
- Modify: `docs/concepts/model.md`
  - Document stateless multi-turn behavior and function-calling support.

## Task 1: Register the Provider Surface

**Files:**
- Modify: `agiwo/config/settings.py`
- Modify: `agiwo/llm/factory.py`
- Modify: `agiwo/llm/__init__.py`
- Modify: `tests/llm/test_factory.py`

- [ ] **Step 1: Write the failing factory tests**

Add these tests to `tests/llm/test_factory.py`:

```python
from agiwo.llm.openai_response import OpenAIResponsesModel


def test_create_model_builds_openai_response_instance() -> None:
    model = create_model(
        ModelSpec(
            provider="openai-response",
            model_name="gpt-4.1-mini",
            temperature=0.2,
            max_output_tokens=256,
        )
    )

    assert isinstance(model, OpenAIResponsesModel)
    assert model.provider == "openai-response"
    assert model.temperature == 0.2
    assert model.max_output_tokens == 256


def test_create_model_from_dict_builds_openai_response_instance(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = create_model_from_dict(
        provider="openai-response",
        model_name="gpt-4.1-mini",
        params={"temperature": 0.15, "max_output_tokens": 111},
    )

    assert isinstance(model, OpenAIResponsesModel)
    assert model.provider == "openai-response"
    assert model.temperature == 0.15
    assert model.max_output_tokens == 111
```

- [ ] **Step 2: Run the factory tests to verify they fail**

Run:

```bash
uv run pytest tests/llm/test_factory.py -v
```

Expected:

```text
FAIL tests/llm/test_factory.py::test_create_model_builds_openai_response_instance
E   pydantic_core._pydantic_core.ValidationError: provider
```

- [ ] **Step 3: Add the provider enum and factory registration**

Update `agiwo/config/settings.py`:

```python
ModelProvider = Literal[
    "openai",
    "openai-response",
    "openai-compatible",
    "deepseek",
    "anthropic",
    "anthropic-compatible",
    "nvidia",
    "bedrock-anthropic",
]
```

Create a minimal provider shell in `agiwo/llm/openai_response.py`:

```python
from agiwo.llm.base import LLMConfig, Model, StreamChunk


class OpenAIResponsesModel(Model):
    def __init__(
        self,
        id: str,
        name: str,
        api_key: str | None = None,
        base_url: str | None = "https://api.openai.com/v1",
        allow_env_fallback: bool = True,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_output_tokens: int = 4096,
        max_context_window: int = 200000,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        cache_hit_price: float = 0.0,
        input_price: float = 0.0,
        output_price: float = 0.0,
        provider: str = "openai-response",
    ) -> None:
        super().__init__(
            LLMConfig(
                id=id,
                name=name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                max_context_window=max_context_window,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                provider=provider,
                cache_hit_price=cache_hit_price,
                input_price=input_price,
                output_price=output_price,
            )
        )
        self.allow_env_fallback = allow_env_fallback

    async def arun_stream(self, messages, tools=None):
        del messages, tools
        if False:
            yield StreamChunk()
```

Update `agiwo/llm/factory.py` imports and provider specs:

```python
from agiwo.llm.openai_response import OpenAIResponsesModel
```

```python
    "openai-response": ProviderSpec(model_class=OpenAIResponsesModel),
```

Update `agiwo/llm/__init__.py`:

```python
from agiwo.llm.openai_response import OpenAIResponsesModel
```

and include:

```python
    "OpenAIResponsesModel",
```

- [ ] **Step 4: Run the factory tests to verify they pass**

Run:

```bash
uv run pytest tests/llm/test_factory.py -v
```

Expected:

```text
PASSED tests/llm/test_factory.py::test_create_model_builds_openai_response_instance
PASSED tests/llm/test_factory.py::test_create_model_from_dict_builds_openai_response_instance
```

- [ ] **Step 5: Commit the provider surface wiring**

Run:

```bash
git add agiwo/config/settings.py agiwo/llm/factory.py agiwo/llm/__init__.py agiwo/llm/openai_response.py tests/llm/test_factory.py
git commit -m "feat: register openai responses model provider"
```

## Task 2: Build the Responses Converter

**Files:**
- Create: `agiwo/llm/openai_response_converter.py`
- Create: `tests/llm/test_openai_response.py`

- [ ] **Step 1: Write failing converter tests**

Create `tests/llm/test_openai_response.py` with these converter tests:

```python
import pytest

from agiwo.llm.openai_response_converter import (
    convert_messages_to_responses_input,
    convert_tools_to_responses_tools,
    split_system_instructions,
)


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
                        "function": {"name": "weather_lookup", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": '{"temp_c":21}'},
            {"role": "assistant", "content": "It is 21C."},
        ]
    )

    assert items[0]["type"] == "message"
    assert items[0]["role"] == "user"
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_123"
    assert items[2]["type"] == "function_call_output"
    assert items[2]["call_id"] == "call_123"
    assert items[3]["type"] == "message"


def test_convert_tools_to_responses_tools_flattens_function_schema() -> None:
    tools = convert_tools_to_responses_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "weather_lookup",
                    "description": "Look up weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]
    )

    assert tools == [
        {
            "type": "function",
            "name": "weather_lookup",
            "description": "Look up weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]


def test_convert_messages_to_responses_input_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="Unsupported message role"):
        convert_messages_to_responses_input([{"role": "developer", "content": "x"}])


def test_convert_tools_to_responses_tools_rejects_non_function_tools() -> None:
    with pytest.raises(ValueError, match="Unsupported tool type"):
        convert_tools_to_responses_tools([{"type": "web_search"}])
```

- [ ] **Step 2: Run the converter tests to verify they fail**

Run:

```bash
uv run pytest tests/llm/test_openai_response.py -v
```

Expected:

```text
ERROR tests/llm/test_openai_response.py
E   ModuleNotFoundError: No module named 'agiwo.llm.openai_response_converter'
```

- [ ] **Step 3: Implement the converter**

Create `agiwo/llm/openai_response_converter.py`:

```python
import json
from typing import Any


def split_system_instructions(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    instruction_parts: list[str] = []
    remaining: list[dict[str, Any]] = []

    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str) and content:
                instruction_parts.append(content)
            continue
        remaining.append(message)

    instructions = "\n\n".join(instruction_parts) or None
    return instructions, remaining


def _serialize_tool_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def _convert_user_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Unsupported user content for openai-response provider")
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": content}],
    }


def _convert_assistant_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        if tool_call.get("type") != "function":
            raise ValueError("Unsupported assistant tool call type for openai-response provider")
        function = tool_call.get("function") or {}
        items.append(
            {
                "type": "function_call",
                "call_id": tool_call["id"],
                "name": function["name"],
                "arguments": function.get("arguments", ""),
            }
        )

    content = message.get("content")
    if content is not None:
        if not isinstance(content, str):
            raise ValueError("Unsupported assistant content for openai-response provider")
        items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        )

    return items


def _convert_tool_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": message["tool_call_id"],
        "output": _serialize_tool_output(message.get("content")),
    }


def convert_messages_to_responses_input(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        if role == "user":
            items.append(_convert_user_message(message))
            continue
        if role == "assistant":
            items.extend(_convert_assistant_message(message))
            continue
        if role == "tool":
            items.append(_convert_tool_message(message))
            continue
        raise ValueError(f"Unsupported message role for openai-response provider: {role}")

    return items


def convert_tools_to_responses_tools(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if not tools:
        return None

    converted: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            raise ValueError("Unsupported tool type for openai-response provider")
        function = tool["function"]
        converted.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
        )

    return converted
```

- [ ] **Step 4: Run the converter tests to verify they pass**

Run:

```bash
uv run pytest tests/llm/test_openai_response.py -v
```

Expected:

```text
PASSED tests/llm/test_openai_response.py::test_split_system_instructions_collects_system_messages
PASSED tests/llm/test_openai_response.py::test_convert_messages_to_responses_input_maps_user_assistant_tool_flow
PASSED tests/llm/test_openai_response.py::test_convert_tools_to_responses_tools_flattens_function_schema
```

- [ ] **Step 5: Commit the converter**

Run:

```bash
git add agiwo/llm/openai_response_converter.py tests/llm/test_openai_response.py
git commit -m "feat: add openai responses request converter"
```

## Task 3: Implement Responses Streaming and Usage Normalization

**Files:**
- Modify: `agiwo/llm/openai_response.py`
- Modify: `tests/llm/test_openai_response.py`

- [ ] **Step 1: Extend provider tests with streaming scenarios**

Append these tests to `tests/llm/test_openai_response.py`:

```python
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agiwo.llm.openai_response import OpenAIResponsesModel


def _event(event_type: str, **kwargs):
    return SimpleNamespace(type=event_type, **kwargs)


@pytest.mark.asyncio
@patch("agiwo.llm.openai_response.get_settings")
async def test_openai_response_model_streams_text_and_usage(
    mock_get_settings,
) -> None:
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None

    model = OpenAIResponsesModel(id="gpt-4.1-mini", name="gpt-4.1-mini", api_key="test-key")
    model.client = AsyncMock()

    stream_events = [
        _event("response.output_text.delta", delta="Hel"),
        _event("response.output_text.delta", delta="lo"),
        _event(
            "response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=7, output_tokens=3, total_tokens=10),
                output=[SimpleNamespace(type="message")],
                incomplete_details=None,
            ),
        ),
    ]

    async def async_iter():
        for item in stream_events:
            yield item

    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = async_iter()
    model.client.responses.create = AsyncMock(return_value=mock_stream)

    chunks = []
    async for chunk in model.arun_stream([{"role": "user", "content": "hello"}]):
        chunks.append(chunk)

    assert [chunk.content for chunk in chunks[:-1]] == ["Hel", "lo"]
    assert chunks[-1].usage["input_tokens"] == 7
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
@patch("agiwo.llm.openai_response.get_settings")
async def test_openai_response_model_streams_function_call_deltas(
    mock_get_settings,
) -> None:
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None

    model = OpenAIResponsesModel(id="gpt-4.1-mini", name="gpt-4.1-mini", api_key="test-key")
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

    async def async_iter():
        for item in stream_events:
            yield item

    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = async_iter()
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
            "function": {"name": "weather_lookup", "arguments": 'is"}'},
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

    model = OpenAIResponsesModel(id="gpt-4.1-mini", name="gpt-4.1-mini", api_key="test-key")
    model.client = AsyncMock()

    async def async_iter():
        if False:
            yield None

    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = async_iter()
    model.client.responses.create = AsyncMock(return_value=mock_stream)

    with pytest.raises(RuntimeError, match="returned no chunks"):
        async for _ in model.arun_stream([{"role": "user", "content": "hello"}]):
            pass
```

- [ ] **Step 2: Run the provider tests to verify they fail**

Run:

```bash
uv run pytest tests/llm/test_openai_response.py -v
```

Expected:

```text
FAIL tests/llm/test_openai_response.py::test_openai_response_model_streams_text_and_usage
E   AssertionError
```

- [ ] **Step 3: Implement request assembly and stream normalization**

Replace the shell implementation in `agiwo/llm/openai_response.py` with:

```python
from typing import Any, AsyncIterator

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from agiwo.config.settings import get_settings
from agiwo.llm.base import LLMConfig, Model, StreamChunk
from agiwo.llm.event_normalizer import normalize_usage_metrics
from agiwo.llm.openai_response_converter import (
    convert_messages_to_responses_input,
    convert_tools_to_responses_tools,
    split_system_instructions,
)
from agiwo.utils.logging import get_logger
from agiwo.utils.retry import retry_async

logger = get_logger(__name__)

OPENAI_RETRYABLE = (
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    APITimeoutError,
)


class OpenAIResponsesModel(Model):
    def __init__(
        self,
        id: str,
        name: str,
        api_key: str | None = None,
        base_url: str | None = "https://api.openai.com/v1",
        allow_env_fallback: bool = True,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_output_tokens: int = 4096,
        max_context_window: int = 200000,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        cache_hit_price: float = 0.0,
        input_price: float = 0.0,
        output_price: float = 0.0,
        provider: str = "openai-response",
    ) -> None:
        super().__init__(
            LLMConfig(
                id=id,
                name=name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                max_context_window=max_context_window,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                provider=provider,
                cache_hit_price=cache_hit_price,
                input_price=input_price,
                output_price=output_price,
            )
        )
        self.allow_env_fallback = allow_env_fallback
        self.client = self._create_client()

    def _resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.allow_env_fallback:
            settings = get_settings()
            if settings.openai_api_key:
                return settings.openai_api_key.get_secret_value()
        return None

    def _resolve_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        if self.allow_env_fallback:
            return get_settings().openai_base_url
        return None

    def _create_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self._resolve_api_key(),
            base_url=self._resolve_base_url(),
        )

    def _build_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        instructions, remaining = split_system_instructions(messages)
        params: dict[str, Any] = {
            "model": self.id or self.name,
            "input": convert_messages_to_responses_input(remaining),
            "stream": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if instructions is not None:
            params["instructions"] = instructions
        if self.max_output_tokens:
            params["max_output_tokens"] = self.max_output_tokens
        if tools:
            params["tools"] = convert_tools_to_responses_tools(tools)
        return params

    @retry_async(exceptions=OPENAI_RETRYABLE)
    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        stream = await self.client.responses.create(**self._build_params(messages, tools))

        chunk_count = 0
        tool_calls: dict[int, dict[str, Any]] = {}

        async for event in stream:
            stream_chunk = self._event_to_chunk(event, tool_calls)
            if stream_chunk is None:
                continue
            chunk_count += 1
            yield stream_chunk

        if chunk_count == 0:
            raise RuntimeError("OpenAI Responses stream returned no chunks")

    def _event_to_chunk(
        self,
        event: Any,
        tool_calls: dict[int, dict[str, Any]],
    ) -> StreamChunk | None:
        if event.type == "response.output_text.delta":
            return StreamChunk(content=event.delta)

        if event.type == "response.output_item.added" and event.item.type == "function_call":
            tool_calls[event.output_index] = {
                "id": event.item.call_id,
                "name": event.item.name,
            }
            return None

        if event.type == "response.function_call_arguments.delta":
            state = tool_calls.get(event.output_index)
            if state is None:
                return None
            return StreamChunk(
                tool_calls=[
                    {
                        "index": event.output_index,
                        "id": state["id"],
                        "type": "function",
                        "function": {
                            "name": state["name"],
                            "arguments": event.delta,
                        },
                    }
                ]
            )

        if event.type == "response.completed":
            finish_reason = "stop"
            output_items = getattr(event.response, "output", None) or []
            if any(getattr(item, "type", None) == "function_call" for item in output_items):
                finish_reason = "tool_calls"

            usage = None
            if getattr(event.response, "usage", None):
                usage = normalize_usage_metrics(event.response.usage)

            return StreamChunk(usage=usage, finish_reason=finish_reason)

        return None
```

Also make sure `__init__()` assigns:

```python
self.client = self._create_client()
```

- [ ] **Step 4: Run the provider tests to verify they pass**

Run:

```bash
uv run pytest tests/llm/test_openai_response.py -v
```

Expected:

```text
PASSED tests/llm/test_openai_response.py::test_openai_response_model_streams_text_and_usage
PASSED tests/llm/test_openai_response.py::test_openai_response_model_streams_function_call_deltas
PASSED tests/llm/test_openai_response.py::test_openai_response_model_rejects_empty_stream
```

- [ ] **Step 5: Commit the provider implementation**

Run:

```bash
git add agiwo/llm/openai_response.py tests/llm/test_openai_response.py
git commit -m "feat: implement openai responses streaming provider"
```

## Task 4: Verify Upper-Layer Compatibility

**Files:**
- Modify: `tests/agent/test_run_contracts.py`

- [ ] **Step 1: Add a contract test for chat-compatible tool-call deltas**

Append this test to `tests/agent/test_run_contracts.py`:

```python
class ToolCallDeltaModel(Model):
    def __init__(self) -> None:
        super().__init__(id="tool-call-delta-model", name="tool-call-delta-model")

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "weather_lookup", "arguments": '{"city":"Par'},
                }
            ]
        )
        yield StreamChunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "weather_lookup", "arguments": 'is"}'},
                }
            ]
        )
        yield StreamChunk(finish_reason="tool_calls")


@pytest.mark.asyncio
async def test_stream_assistant_step_accumulates_chat_compatible_tool_call_deltas() -> None:
    session_runtime = SessionRuntime(
        session_id="tool-call-contract-session",
        run_step_storage=InMemoryRunStepStorage(),
    )
    agent = Agent(
        AgentConfig(name="tool-call-contract", description="tool call contract test"),
        model=ToolCallDeltaModel(),
    )

    handle = agent.start("weather?", session_id="tool-call-contract-session")
    result = await handle.wait()
    steps = await session_runtime.run_step_storage.get_steps(
        session_id="tool-call-contract-session",
        agent_id=agent.id,
    )

    assert result.response is None
    assert steps[-1].tool_calls == [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "weather_lookup", "arguments": '{"city":"Paris"}'},
        }
    ]
```

- [ ] **Step 2: Run the contract test to verify it fails**

Run:

```bash
uv run pytest tests/agent/test_run_contracts.py::test_stream_assistant_step_accumulates_chat_compatible_tool_call_deltas -v
```

Expected:

```text
FAIL tests/agent/test_run_contracts.py::test_stream_assistant_step_accumulates_chat_compatible_tool_call_deltas
E   AssertionError
```

- [ ] **Step 3: Fix the test to inspect persisted assistant steps through the agent session**

Replace the test body with this exact version so it reads the persisted assistant step from the same `SessionRuntime` used by the run:

```python
@pytest.mark.asyncio
async def test_stream_assistant_step_accumulates_chat_compatible_tool_call_deltas() -> None:
    session_runtime = SessionRuntime(
        session_id="tool-call-contract-session",
        run_step_storage=InMemoryRunStepStorage(),
    )
    agent = Agent(
        AgentConfig(name="tool-call-contract", description="tool call contract test"),
        model=ToolCallDeltaModel(),
    )

    handle = agent.start(
        "weather?",
        session_id="tool-call-contract-session",
        session_runtime=session_runtime,
    )
    result = await handle.wait()
    steps = await session_runtime.run_step_storage.get_steps(
        session_id="tool-call-contract-session",
        agent_id=agent.id,
    )

    assert result.response is None
    assert steps[-1].tool_calls == [
        {
            "id": "call_123",
            "type": "function",
            "function": {"name": "weather_lookup", "arguments": '{"city":"Paris"}'},
        }
    ]
```

- [ ] **Step 4: Run the contract test to verify it passes**

Run:

```bash
uv run pytest tests/agent/test_run_contracts.py::test_stream_assistant_step_accumulates_chat_compatible_tool_call_deltas -v
```

Expected:

```text
PASSED tests/agent/test_run_contracts.py::test_stream_assistant_step_accumulates_chat_compatible_tool_call_deltas
```

- [ ] **Step 5: Commit the contract coverage**

Run:

```bash
git add tests/agent/test_run_contracts.py
git commit -m "test: cover tool-call delta contract for responses provider"
```

## Task 5: Document the Provider and Run the Repo Checks

**Files:**
- Modify: `docs/api/model.md`
- Modify: `docs/concepts/model.md`

- [ ] **Step 1: Update the docs**

In `docs/api/model.md`, add `OpenAIResponsesModel` to the provider list:

~~~markdown
### `OpenAIResponsesModel`

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="openai-response",
    model_name="gpt-4.1-mini",
)
```

Uses OpenAI Responses API internally while preserving the SDK `StreamChunk` contract.
First-version support covers streamed text and function calling. Multi-turn replay is stateless and does not use `previous_response_id`.
~~~

In `docs/concepts/model.md`, add this provider note under built-in providers:

~~~markdown
### OpenAI Responses

For OpenAI models that should use the Responses API instead of Chat Completions:

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="openai-response",
    model_name="gpt-4.1-mini",
)
```

This provider still exposes `Model.arun_stream()` and normalized `StreamChunk` output. It currently supports text streaming and function calling, and reconstructs each request statelessly from the SDK message ledger.
~~~

- [ ] **Step 2: Run focused tests**

Run:

```bash
uv run pytest tests/llm/test_factory.py tests/llm/test_openai_response.py tests/agent/test_run_contracts.py::test_stream_assistant_step_accumulates_chat_compatible_tool_call_deltas -v
```

Expected:

```text
PASSED tests/llm/test_factory.py::test_create_model_builds_openai_response_instance
PASSED tests/llm/test_openai_response.py::test_openai_response_model_streams_text_and_usage
PASSED tests/agent/test_run_contracts.py::test_stream_assistant_step_accumulates_chat_compatible_tool_call_deltas
```

- [ ] **Step 3: Run changed-file lint**

Run:

```bash
uv run python scripts/lint.py changed
```

Expected:

```text
All checks passed
```

- [ ] **Step 4: Commit the docs and verification updates**

Run:

```bash
git add docs/api/model.md docs/concepts/model.md
git commit -m "docs: add openai responses provider documentation"
```

- [ ] **Step 5: Run the pre-submit lint sequence**

Run:

```bash
uv run ruff check --ignore C901 --ignore PLR0911 --ignore PLR0912 agiwo/ console/server/ tests/ console/tests/ scripts/
uv run ruff format --check agiwo/ console/server/ tests/ console/tests/ scripts/
uv run python scripts/lint.py imports
uv run python scripts/repo_guard.py
```

Expected:

```text
All commands exit with code 0.
```

## Self-Review

### Spec coverage

- New provider enum and factory registration: covered by Task 1.
- Dedicated `OpenAIResponsesModel`: covered by Task 3.
- Private converter for canonical messages/tools: covered by Task 2.
- Stateless replay without `previous_response_id`: covered by Task 2 and Task 3 request assembly.
- Streamed text and function calling: covered by Task 3.
- Upper-layer `StreamChunk` compatibility: covered by Task 4.
- Docs updates: covered by Task 5.

### Placeholder scan

Checked for `TBD`, `TODO`, vague "handle edge cases" wording, and references to undefined future tasks. None remain.

### Type consistency

- Provider class name is consistently `OpenAIResponsesModel`.
- Provider string is consistently `openai-response`.
- Converter helper names are consistently `split_system_instructions`, `convert_messages_to_responses_input`, and `convert_tools_to_responses_tools`.
- Normalized tool-call delta shape matches `agiwo.agent.llm_caller._accumulate_tool_calls()`.
