from typing import AsyncIterator
import json

try:
    from openai import (
        AsyncOpenAI,
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
except ImportError:
    raise ImportError("Please install openai package: pip install openai")

from agiwo.llm.base import Model, StreamChunk
from agiwo.llm.helper import normalize_usage_metrics
from agiwo.config.settings import settings
from agiwo.utils.retry import retry_async
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


OPENAI_RETRYABLE = (
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    APITimeoutError,
)


class OpenAIModel(Model):
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
    ):
        super().__init__(
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
            provider="openai",
            cache_hit_price=cache_hit_price,
            input_price=input_price,
            output_price=output_price,
        )
        self.allow_env_fallback = allow_env_fallback
        self.client = self._create_client()

    def _resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.allow_env_fallback and settings.openai_api_key:
            return settings.openai_api_key.get_secret_value()
        return None

    def _resolve_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        if self.allow_env_fallback:
            return settings.openai_base_url
        return None

    def _create_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self._resolve_api_key(),
            base_url=self._resolve_base_url(),
        )

    @retry_async(exceptions=OPENAI_RETRYABLE)
    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Call OpenAI API and return standardized streaming output.

        Args:
            messages: OpenAI format message list
            tools: OpenAI format tool definitions

        Yields:
            StreamChunk: Standardized streaming output chunk
        """
        actual_model = self.id or self.name
        params = {
            "model": actual_model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if self.max_output_tokens:
            params["max_tokens"] = self.max_output_tokens

        if tools:
            params["tools"] = tools

        logger.debug(
            "llm_request",
            model=actual_model,
            messages_count=len(messages),
            tools_count=len(tools) if tools else 0,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            detail=params,
        )

        try:
            stream = await self.client.chat.completions.create(**params)
        except Exception as e:
            logger.error(
                "llm_request_failed",
                model=actual_model,
                error=str(e),
                error_type=type(e).__name__,
                messages_count=len(messages),
                tools_count=len(tools) if tools else 0,
                exc_info=True,
            )
            raise

        async for chunk in stream:
            stream_chunk = StreamChunk()

            if chunk.usage:
                stream_chunk.usage = normalize_usage_metrics(chunk.usage)

            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    stream_chunk.content = delta.content

                # Handle reasoning_content (DeepSeek thinking mode)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    stream_chunk.reasoning_content = delta.reasoning_content

                if delta.tool_calls:
                    stream_chunk.tool_calls = [
                        tc.model_dump(exclude_none=True) for tc in delta.tool_calls
                    ]

                if choice.finish_reason:
                    stream_chunk.finish_reason = choice.finish_reason

            if (
                stream_chunk.content is not None
                or stream_chunk.reasoning_content is not None
                or stream_chunk.tool_calls is not None
                or stream_chunk.usage is not None
            ):
                yield stream_chunk
