from typing import AsyncIterator
import os
import json


try:
    from anthropic import (
        APIConnectionError,
        APITimeoutError,
        AsyncAnthropic,
        RateLimitError,
    )
except ImportError:
    raise ImportError("Please install anthropic package: uv add anthropic")

from agiwo.llm.base import Model, StreamChunk
from agiwo.llm.helper import normalize_usage_metrics
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
        base_url: str = "https://api.anthropic.com/v1",
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

        resolved_api_key = None
        if self.api_key:
            if hasattr(self.api_key, "get_secret_value"):
                resolved_api_key = self.api_key.get_secret_value()
            else:
                resolved_api_key = self.api_key
        elif hasattr(settings, "anthropic_api_key") and settings.anthropic_api_key:
            resolved_api_key = settings.anthropic_api_key.get_secret_value()
        else:
            resolved_api_key = os.getenv("ANTHROPIC_API_KEY")

        resolved_base_url = (
            self.base_url
            or settings.anthropic_base_url
            or os.getenv("ANTHROPIC_BASE_URL")
        )

        if not hasattr(self, "client") or self.client is None:
            client_kwargs = {"api_key": resolved_api_key}
            if resolved_base_url:
                client_kwargs["base_url"] = resolved_base_url
            self.client = AsyncAnthropic(**client_kwargs)

        logger.info(
            "AnthropicModel initialized",
            model_name=getattr(self, "model_name", None) or self.name,
            use_key=resolved_api_key[:10] if resolved_api_key else "None",
        )

    def _parse_tool_args(self, args: str | dict) -> dict:
        """Parse tool arguments into a dictionary."""
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                logger.error("failed_to_decode_tool_arguments", arguments=args)
                return {"__raw_arguments__": args}

        if not isinstance(args, dict):
            logger.error(
                "invalid_tool_arguments_type",
                arguments=args,
                type=type(args).__name__,
            )
            return {"__raw_arguments__": args}

        return args

    def _update_usage_info(self, usage_obj, usage_info: dict) -> None:
        """Update usage info from Anthropic usage object."""
        if not usage_obj:
            return

        for key, attr in [
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("cache_read_tokens", "cache_read_input_tokens"),
            ("cache_creation_tokens", "cache_creation_input_tokens"),
        ]:
            val = getattr(usage_obj, attr, None)
            if val is not None:
                usage_info[key] = val

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI format messages to Anthropic format."""
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                system_prompt = content
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                reasoning = msg.get("reasoning_content")
                if "tool_calls" in msg or reasoning:
                    content_blocks = []

                    if reasoning:
                        content_blocks.append(
                            {"type": "thinking", "thinking": reasoning}
                        )

                    if content:
                        content_blocks.append({"type": "text", "text": content})

                    if "tool_calls" in msg:
                        for tool_call in msg["tool_calls"]:
                            func = tool_call["function"]
                            args = func["arguments"]
                            args = self._parse_tool_args(args)

                            content_blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": tool_call["id"],
                                    "name": func["name"],
                                    "input": args,
                                }
                            )

                    anthropic_messages.append(
                        {"role": "assistant", "content": content_blocks}
                    )
                else:
                    anthropic_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Ensure tool result content is a string
                tool_result_content = content
                if not isinstance(tool_result_content, str):
                    tool_result_content = json.dumps(tool_result_content)

                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id"),
                                "content": tool_result_content,
                            }
                        ],
                    }
                )

        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Convert OpenAI format tools to Anthropic format."""
        if not tools:
            return None

        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
                    }
                )

        return anthropic_tools if anthropic_tools else None

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
        anthropic_tools = self._convert_tools(tools)

        actual_model = getattr(self, "model_name", None) or self.name
        params = {
            "model": actual_model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens or self.max_tokens_to_sample,
            "temperature": self.temperature,
            "stream": True,
        }

        if system_prompt:
            params["system"] = system_prompt

        if self.top_p is not None:
            params["top_p"] = self.top_p

        if anthropic_tools:
            params["tools"] = anthropic_tools

        logger.info(
            "llm_request",
            model=actual_model,
            messages_count=len(anthropic_messages),
            tools_count=len(anthropic_tools) if anthropic_tools else 0,
            temperature=self.temperature,
            max_tokens=params["max_tokens"],
            detail=json.dumps(params, ensure_ascii=False),
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

        # Track tool calls being built during streaming
        tool_calls_buffer = {}
        # Track usage
        usage_info = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }

        async for event in stream:
            stream_chunk = StreamChunk()

            if event.type == "message_start":
                if hasattr(event.message, "usage"):
                    self._update_usage_info(event.message.usage, usage_info)
                    stream_chunk.usage = normalize_usage_metrics(usage_info)

            elif event.type == "content_block_start":
                if event.content_block.type == "tool_use":
                    index = event.index
                    tool_calls_buffer[index] = {
                        "id": event.content_block.id,
                        "name": event.content_block.name,
                        "input": "",
                    }

            elif event.type == "content_block_delta":
                delta = event.delta

                if delta.type == "text_delta":
                    stream_chunk.content = delta.text
                elif delta.type == "thinking_delta":
                    stream_chunk.reasoning_content = delta.thinking
                elif delta.type == "input_json_delta":
                    index = event.index
                    if index in tool_calls_buffer:
                        tool_calls_buffer[index]["input"] += delta.partial_json

            elif event.type == "content_block_stop":
                index = event.index
                if index in tool_calls_buffer:
                    tool_call = tool_calls_buffer[index]
                    stream_chunk.tool_calls = [
                        {
                            "index": index,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": tool_call["input"],
                            },
                        }
                    ]
                    # Note: We emit the tool call when it's complete
                    # or should we emit deltas? Agio seems to prefer chunks.
                    # Given the current structure, we'll emit the full tool call at block stop.

            elif event.type == "message_delta":
                if hasattr(event, "usage"):
                    self._update_usage_info(event.usage, usage_info)
                    stream_chunk.usage = normalize_usage_metrics(usage_info)

                if event.delta.stop_reason:
                    # Anthropic stop reasons: end_turn, max_tokens, stop_sequence, tool_use
                    stop_reason = event.delta.stop_reason
                    if stop_reason == "tool_use":
                        stream_chunk.finish_reason = "tool_calls"
                    else:
                        stream_chunk.finish_reason = stop_reason

            elif event.type == "message_stop":
                if stream_chunk.finish_reason is None:
                    stream_chunk.finish_reason = "stop"

            if (
                stream_chunk.content is not None
                or stream_chunk.reasoning_content is not None
                or stream_chunk.tool_calls is not None
                or stream_chunk.usage is not None
                or stream_chunk.finish_reason is not None
            ):
                yield stream_chunk


__all__ = ["AnthropicModel"]
