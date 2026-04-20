# Model API Reference

## Public Exports

`agiwo.llm` currently exports:

- `LLMConfig`
- `Model`
- `StreamChunk`
- `ModelSpec`
- `ModelProvider`
- `create_model(...)`
- `create_model_from_dict(...)`
- `OpenAIModel`
- `OpenAIResponsesModel`
- `AnthropicModel`
- `BedrockAnthropicModel`
- `DeepseekModel`
- `NvidiaModel`

## `LLMConfig`

Runtime configuration held by a live model instance.

```python
@dataclass
class LLMConfig:
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
```

Validation performed by `LLMConfig.__post_init__`:

- `temperature` must be in `[0.0, 2.0]`
- `max_output_tokens` must be at least `1` when set
- `max_context_window` must be at least `1` when set
- token price fields must be non-negative

## `Model`

Abstract base class for all LLM providers.

```python
class Model(ABC):
    config: LLMConfig

    def __init__(self, config: LLMConfig | None = None, **kwargs: Any) -> None: ...

    @abstractmethod
    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    async def close(self) -> None: ...
```

Important behavior:

- subclasses hold config on `model.config`
- config fields are also exposed via `__getattr__`, so `model.name` delegates to `model.config.name`
- callers pass either a `config` object or keyword args, not both

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

## Provider Classes

### `OpenAIModel`

```python
from agiwo.llm import OpenAIModel

model = OpenAIModel(name="gpt-5.4")
```

Uses Chat Completions style streaming and reads `OPENAI_API_KEY` from environment unless an explicit `api_key` is supplied.

### `OpenAIResponsesModel`

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="openai-response",
    model_name="gpt-4.1-mini",
)
```

Uses the OpenAI Responses API while preserving Agiwo's `StreamChunk` contract.

### `AnthropicModel`

```python
from agiwo.llm import AnthropicModel

model = AnthropicModel(id="claude-sonnet-4-20250514", name="claude-sonnet-4")
```

### `DeepseekModel`

```python
from agiwo.llm import DeepseekModel

model = DeepseekModel()
```

Defaults to `id="deepseek/deepseek-chat"` and `name="deepseek-chat"`, and adds DeepSeek-specific preprocessing for reasoning mode plus DSML tool-call recovery.

### `NvidiaModel`

```python
from agiwo.llm import NvidiaModel

model = NvidiaModel(id="meta/llama-3.1-70b-instruct", name="llama-3.1-70b")
```

### `BedrockAnthropicModel`

```python
from agiwo.llm import BedrockAnthropicModel

model = BedrockAnthropicModel(
    id="anthropic.claude-3-sonnet",
    name="claude-3-sonnet",
)
```

## Factory APIs

### `ModelSpec`

Serializable construction spec used by `create_model(...)`.

```python
from agiwo.llm import ModelSpec

spec = ModelSpec(
    provider="openai",
    model_name="gpt-5.4",
    temperature=0.5,
    max_output_tokens=8192,
)
```

### `create_model()`

```python
from agiwo.llm import ModelSpec, create_model

model = create_model(
    ModelSpec(
        provider="openai",
        model_name="gpt-5.4",
        temperature=0.5,
    )
)
```

### `create_model_from_dict()`

For loose dict inputs or config files:

```python
from agiwo.llm import create_model_from_dict

model = create_model_from_dict(
    provider="openai-compatible",
    model_name="custom-model",
    params={
        "base_url": "https://api.example.com/v1",
        "api_key_env_name": "MY_API_KEY",
        "temperature": 0.2,
    },
)
```

Supported provider strings:

- `"openai"`
- `"openai-response"`
- `"openai-compatible"`
- `"deepseek"`
- `"anthropic"`
- `"anthropic-compatible"`
- `"nvidia"`
- `"bedrock-anthropic"`

Compatible endpoint notes:

- `openai-compatible` and `anthropic-compatible` require an explicit `base_url`
- they also require `api_key_env_name` to resolve credentials at runtime
