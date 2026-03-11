from typing import Any, AsyncIterator

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
        **model_kwargs: Any,
    ):
        super().__init__(
            id=id,
            name=name,
            api_key=api_key,
            base_url=base_url,
            **model_kwargs,
        )
        self.provider = "nvidia"

    def _resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if settings.nvidia_api_key:
            return settings.nvidia_api_key.get_secret_value()
        return None

    def _resolve_base_url(self) -> str | None:
        return (
            self.base_url
            or settings.nvidia_base_url
            or "https://integrate.api.nvidia.com/v1"
        )


__all__ = ["NvidiaModel"]
