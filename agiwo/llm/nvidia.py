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

    def __post_init__(self):
        super().__post_init__()

        # Resolve API Key: nvidia_api_key > NVIDIA_BUILD_API_KEY
        resolved_api_key = None
        if self.api_key:
            resolved_api_key = self.api_key
        elif settings.nvidia_api_key:
            resolved_api_key = settings.nvidia_api_key.get_secret_value()
        else:
            resolved_api_key = os.getenv("NVIDIA_BUILD_API_KEY")

        # Resolve Base URL
        resolved_base_url = (
            self.base_url
            or settings.nvidia_base_url
            or os.getenv("NVIDIA_BASE_URL")
            or "https://integrate.api.nvidia.com/v1"
        )

        # Create client
        if not hasattr(self, "client") or self.client is None:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                api_key=resolved_api_key,
                base_url=resolved_base_url,
            )

    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Call NVIDIA API.

        NVIDIA response chunks include reasoning_content which is already handled by OpenAIModel.arun_stream.
        """
        async for chunk in super().arun_stream(messages, tools=tools):
            yield chunk


__all__ = ["NvidiaModel"]
