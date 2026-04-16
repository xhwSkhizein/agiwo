# Model API Reference

## `Model`

Base class for all LLM providers.

```python
@dataclass
class Model(ABC):
    id: str
    name: str
    temperature: float = 0.7
    top_p: float = 1.0
    max_output_tokens: int = 4096
    max_context_window: int = 200000
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_key: str | None = None
    base_url: str | None = None
    provider: str = ""
    cache_hit_price: float = 0.0
    input_price: float = 0.0
    output_price: float = 0.0

    @abstractmethod
    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    async def close(self) -> None: ...
```

### Validation

The `__post_init__` method validates:
- `temperature` must be in `[0.0, 2.0]`
- `max_output_tokens` must be ≥ 1
- `max_context_window` must be ≥ 1
- Token prices must be non-negative

---

## `StreamChunk`

```python
@dataclass
class StreamChunk:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
```

---

## Provider Classes

### `OpenAIModel`

```python
from agiwo.llm import OpenAIModel

model = OpenAIModel(name="gpt-5.4")
# Reads OPENAI_API_KEY from environment
```

### `OpenAIResponsesModel`

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="openai-response",
    model_name="gpt-4.1-mini",
)
```

Uses OpenAI Responses API internally while preserving the SDK `StreamChunk`
contract. First-version support covers streamed text and function calling.
Multi-turn replay is stateless and does not use `previous_response_id`.

### `AnthropicModel`

```python
from agiwo.llm import AnthropicModel

model = AnthropicModel(id="claude-sonnet-4-20250514", name="claude-sonnet-4")
# Reads ANTHROPIC_API_KEY from environment
```

### `DeepseekModel`

```python
from agiwo.llm import DeepseekModel

model = DeepseekModel(id="deepseek-chat", name="deepseek-chat")
# Reads DEEPSEEK_API_KEY from environment
```

### `NvidiaModel`

```python
from agiwo.llm import NvidiaModel

model = NvidiaModel(id="meta/llama-3.1-70b-instruct", name="llama-3.1-70b")
```

### `BedrockAnthropicModel`

```python
from agiwo.llm import BedrockAnthropicModel

model = BedrockAnthropicModel(id="anthropic.claude-3-sonnet", name="claude-3-sonnet")
# Uses AWS credentials (boto3)
```

## Compatible Endpoints

For OpenAI-compatible or Anthropic-compatible endpoints, use the factory:

### OpenAI-Compatible

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="openai-compatible",
    model_name="custom-model",
    params={
        "base_url": "https://api.example.com/v1",
        "api_key_env_name": "MY_API_KEY",
    },
)
```

### Anthropic-Compatible

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="anthropic-compatible",
    model_name="custom-model",
    params={
        "base_url": "https://api.example.com/v1",
        "api_key_env_name": "MY_API_KEY",
    },
)
```

---

## Model Factory

### `create_model()`

```python
from agiwo.llm.factory import create_model, ModelSpec

model = create_model(ModelSpec(
    provider="openai",
    model_name="gpt-4o",
    temperature=0.5,
    max_output_tokens=8192,
))
```

### `create_model_from_dict()`

For use with loose dicts or config files:

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="openai",
    model_name="gpt-4o",
    params={
        "temperature": 0.5,
        "max_output_tokens": 8192,
    },
)
```

Supported provider strings: `"openai"`, `"openai-response"`, `"anthropic"`, `"deepseek"`, `"nvidia"`, `"bedrock-anthropic"`, `"openai-compatible"`, `"anthropic-compatible"`.
