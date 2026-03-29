from typing import Any, AsyncIterator
from dataclasses import dataclass
from abc import abstractmethod, ABC


@dataclass
class StreamChunk:
    """Standardized streaming chunk from any LLM provider."""

    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


@dataclass
class LLMConfig:
    """Pure data container for LLM model configuration.

    Extracted so that the configuration can be serialized, compared, and
    passed around independently of the live client held by ``Model``.

    .. note::

       Not to be confused with ``agiwo.llm.factory.ModelConfig`` (Pydantic),
       which is the *construction spec* used to create a ``Model`` instance.
       ``LLMConfig`` is the runtime config held by the model after creation.
    """

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

    def __post_init__(self) -> None:
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be at least 1")
        if self.max_context_window is not None and self.max_context_window < 1:
            raise ValueError("max_context_window must be at least 1")
        if self.cache_hit_price < 0 or self.input_price < 0 or self.output_price < 0:
            raise ValueError("token prices must be non-negative")


class Model(ABC):
    """Abstract base for all LLM model implementations.

    Subclasses compose an ``LLMConfig`` via the ``config`` attribute and
    implement ``arun_stream()`` to provide streaming responses.

    All configuration fields from ``LLMConfig`` are available as direct
    properties for backward compatibility.
    """

    config: LLMConfig

    def __init__(self, config: LLMConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise TypeError(
                "Pass either a config object or keyword arguments, not both"
            )
        if config is not None:
            self.config = config
        elif kwargs:
            self.config = LLMConfig(**kwargs)
        else:
            raise TypeError(
                "Model requires either a config argument or keyword arguments"
            )

    # Backward-compatible property accessors
    @property
    def id(self) -> str:
        return self.config.id

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def temperature(self) -> float:
        return self.config.temperature

    @property
    def top_p(self) -> float:
        return self.config.top_p

    @property
    def max_output_tokens(self) -> int:
        return self.config.max_output_tokens

    @property
    def max_context_window(self) -> int:
        return self.config.max_context_window

    @property
    def frequency_penalty(self) -> float:
        return self.config.frequency_penalty

    @property
    def presence_penalty(self) -> float:
        return self.config.presence_penalty

    @property
    def api_key(self) -> str | None:
        return self.config.api_key

    @property
    def base_url(self) -> str | None:
        return self.config.base_url

    @property
    def provider(self) -> str:
        return self.config.provider

    @provider.setter
    def provider(self, value: str) -> None:
        self.config.provider = value

    @property
    def cache_hit_price(self) -> float:
        return self.config.cache_hit_price

    @property
    def input_price(self) -> float:
        return self.config.input_price

    @property
    def output_price(self) -> float:
        return self.config.output_price

    @abstractmethod
    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream LLM responses as standardized chunks."""

    async def close(self) -> None:
        """Close the model's underlying client connection."""
        if hasattr(self, "client") and self.client is not None:
            await self.client.close()


__all__ = ["LLMConfig", "Model", "StreamChunk"]
