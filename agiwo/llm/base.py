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
class Model(ABC):
    """
    Abstract base for all LLM model implementations.

    Subclasses must implement arun_stream() to provide streaming responses.
    """

    id: str
    name: str
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_key: str | None = None
    base_url: str | None = None
    provider: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

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


__all__ = ["Model", "StreamChunk"]
