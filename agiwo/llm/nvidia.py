import os
from typing import AsyncIterator

from agiwo.config.settings import settings
from agiwo.llm.base import StreamChunk
from agiwo.llm.openai import OpenAIModel


class NvidiaModel(OpenAIModel):
    def __init__(
        self,
        id: str,
        name: str,
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ):
        super().__init__(
            id=id,
            name=name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    def _resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if settings.nvidia_api_key:
            return settings.nvidia_api_key.get_secret_value()
        return os.getenv("NVIDIA_BUILD_API_KEY")

    def _resolve_base_url(self) -> str | None:
        return (
            self.base_url
            or settings.nvidia_base_url
            or os.getenv("NVIDIA_BUILD_BASE_URL")
            or "https://integrate.api.nvidia.com/v1"
        )


__all__ = ["NvidiaModel"]
