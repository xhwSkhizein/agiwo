from typing import Any, AsyncIterator
from dataclasses import dataclass, field
from abc import abstractmethod, ABC


@dataclass
class StreamChunk:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[dict] | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None


@dataclass
class Model(ABC):
    id: str
    name: str
    temperature: float
    top_p: float
    max_tokens: int
    frequency_penalty: float
    presence_penalty: float
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    stream: bool = True
    # for Pydantic's extra="allow"
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        validate the model parameters
        """
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
        pass


__all__ = ["Model", "StreamChunk"]
