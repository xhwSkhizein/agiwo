# Model

Models are the LLM backends that power agent reasoning. Agiwo uses a streaming-first abstraction that all providers implement.

## Model Interface

All models extend the `Model` base class and implement `arun_stream()`:

```python
from agiwo.llm.base import Model, StreamChunk
from typing import AsyncIterator


class MyProvider(Model):
    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        # Provider-specific streaming implementation
        yield StreamChunk(content="Hello", finish_reason="stop")
```

## StreamChunk

Every provider yields standardized `StreamChunk` objects:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str \| None` | Text content delta |
| `reasoning_content` | `str \| None` | Reasoning/thinking content (for models that support it) |
| `tool_calls` | `list[dict] \| None` | Tool call deltas |
| `usage` | `dict \| None` | Token usage (prompt, completion, total) |
| `finish_reason` | `str \| None` | `"stop"`, `"tool_calls"`, `"length"`, etc. |

## Model Configuration

All models share these fields:

| Field | Default | Description |
|-------|---------|-------------|
| `id` | (required) | Provider model ID (e.g., `"gpt-4o"`) |
| `name` | (required) | Human-readable name |
| `temperature` | `0.7` | Sampling temperature (0.0–2.0) |
| `top_p` | `1.0` | Nucleus sampling |
| `max_output_tokens` | `4096` | Max tokens in response |
| `max_context_window` | `200000` | Max context length |
| `frequency_penalty` | `0.0` | Frequency penalty |
| `presence_penalty` | `0.0` | Presence penalty |

Pricing fields (for cost tracking):

| Field | Default | Description |
|-------|---------|-------------|
| `input_price` | `0.0` | Cost per input token |
| `output_price` | `0.0` | Cost per output token |
| `cache_hit_price` | `0.0` | Cost per cached token |

## Built-in Providers

### OpenAI

```python
from agiwo.llm import OpenAIModel

model = OpenAIModel(id="gpt-4o", name="gpt-4o")
# Reads OPENAI_API_KEY from environment
```

### OpenAI Responses

For OpenAI models that should use the Responses API instead of Chat
Completions:

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="openai-response",
    model_name="gpt-4.1-mini",
)
```

This provider still exposes `Model.arun_stream()` and normalized
`StreamChunk` output. It currently supports text streaming and function
calling, and reconstructs each request statelessly from the SDK message
ledger.

### Anthropic

```python
from agiwo.llm import AnthropicModel

model = AnthropicModel(id="claude-sonnet-4-20250514", name="claude-sonnet-4")
# Reads ANTHROPIC_API_KEY from environment
```

### DeepSeek

```python
from agiwo.llm import DeepseekModel

model = DeepseekModel(id="deepseek-chat", name="deepseek-chat")
# Reads DEEPSEEK_API_KEY from environment
```

### NVIDIA

```python
from agiwo.llm import NvidiaModel

model = NvidiaModel(id="meta/llama-3.1-70b-instruct", name="llama-3.1-70b")
```

### Bedrock (Anthropic)

```python
from agiwo.llm import BedrockAnthropicModel

model = BedrockAnthropicModel(id="anthropic.claude-3-sonnet", name="claude-3-sonnet")
# Uses AWS credentials (boto3)
```

### OpenAI-Compatible

For any OpenAI-compatible API endpoint:

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="openai-compatible",
    model_name="my-custom-model",
    params={
        "base_url": "https://api.example.com/v1",
        "api_key_env_name": "MY_API_KEY",
    },
)
```

### Anthropic-Compatible

For Anthropic-compatible endpoints:

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="anthropic-compatible",
    model_name="my-anthropic-model",
    params={
        "base_url": "https://api.example.com/v1",
        "api_key_env_name": "MY_API_KEY",
    },
)
```

## Model Factory

Models are typically constructed through the factory for consistency:

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="openai",
    model_name="gpt-4o",
    params={"temperature": 0.5},
)
```

The factory reads provider credentials from environment variables automatically.

## Adding a New Provider

1. Implement the provider class in `agiwo/llm/`
2. Register the provider enum in `agiwo/config/settings.py`
3. Add a `ProviderSpec` in `agiwo/llm/factory.py`
4. Export from `agiwo/llm/__init__.py` if public
5. Add tests in `tests/llm/`

See the existing providers in `agiwo/llm/` for reference implementations.
