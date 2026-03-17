from typing import Any, AsyncIterator


try:
    from anthropic import (
        APIConnectionError,
        APITimeoutError,
        AsyncAnthropic,
        RateLimitError,
    )
except ImportError:
    raise ImportError("Please install anthropic package: uv add anthropic") from None

from agiwo.llm.base import Model, StreamChunk
from agiwo.llm.event_normalizer import (
    AnthropicStreamTranslator,
    normalize_anthropic_sdk_event,
)
from agiwo.llm.message_converter import (
    convert_openai_messages_to_anthropic,
    convert_openai_tools_to_anthropic,
)
from agiwo.config.settings import settings
from agiwo.utils.retry import retry_async
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


# Retryable exceptions for Anthropic
ANTHROPIC_RETRYABLE = (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)


class AnthropicModel(Model):
    def __init__(
        self,
        id: str,
        name: str,
        api_key: str | None = None,
        base_url: str | None = "https://api.anthropic.com/v1",
        allow_env_fallback: bool = True,
        **model_kwargs: Any,
    ):
        super().__init__(
            id=id,
            name=name,
            api_key=api_key,
            base_url=base_url,
            provider="anthropic",
            **model_kwargs,
        )
        self.allow_env_fallback = allow_env_fallback

        self.client = self._create_client()

    def _resolve_api_key(self) -> str | None:
        if self.api_key:
            if hasattr(self.api_key, "get_secret_value"):
                return self.api_key.get_secret_value()
            return self.api_key
        if (
            self.allow_env_fallback
            and hasattr(settings, "anthropic_api_key")
            and settings.anthropic_api_key
        ):
            return settings.anthropic_api_key.get_secret_value()
        return None

    def _resolve_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        if self.allow_env_fallback:
            return settings.anthropic_base_url
        return None

    def _create_client(self) -> AsyncAnthropic:
        client_kwargs = {"api_key": self._resolve_api_key()}
        base_url = self._resolve_base_url()
        if base_url:
            client_kwargs["base_url"] = base_url
        return AsyncAnthropic(**client_kwargs)

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        return convert_openai_messages_to_anthropic(
            messages,
            wrap_user_text=False,
            assistant_text_blocks=False,
            include_reasoning=True,
        )

    @retry_async(exceptions=ANTHROPIC_RETRYABLE)
    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Call Anthropic API and return standardized streaming output.

        Args:
            messages: OpenAI format message list
            tools: OpenAI format tool definitions

        Yields:
            StreamChunk: Standardized streaming output chunk
        """
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = convert_openai_tools_to_anthropic(tools)

        actual_model = getattr(self, "model_name", None) or self.name
        params = {
            "model": actual_model,
            "messages": anthropic_messages,
            "max_tokens": self.max_output_tokens or self.max_tokens_to_sample,
            "temperature": self.temperature,
            "stream": True,
        }

        if system_prompt:
            params["system"] = system_prompt

        if self.top_p is not None:
            params["top_p"] = self.top_p

        if anthropic_tools:
            params["tools"] = anthropic_tools

        logger.debug(
            "llm_request",
            model=actual_model,
            messages_count=len(anthropic_messages),
            tools_count=len(anthropic_tools) if anthropic_tools else 0,
            temperature=self.temperature,
            max_tokens=params["max_tokens"],
            detail=params,
        )

        try:
            stream = await self.client.messages.create(**params)
        except Exception as e:
            logger.error(
                "llm_request_failed",
                model=actual_model,
                error=str(e),
                error_type=type(e).__name__,
                messages_count=len(anthropic_messages),
                tools_count=len(anthropic_tools) if anthropic_tools else 0,
                exc_info=True,
            )
            raise

        translator = AnthropicStreamTranslator(include_reasoning=True)

        async for event in stream:
            stream_chunk = translator.process(normalize_anthropic_sdk_event(event))
            if stream_chunk is not None:
                yield stream_chunk


__all__ = ["AnthropicModel"]
