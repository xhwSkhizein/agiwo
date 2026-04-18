from contextlib import AsyncExitStack
from typing import AsyncIterator
import json
import httpx
from agiwo.config.settings import get_settings
from agiwo.llm.base import LLMConfig, Model, StreamChunk
from agiwo.llm.event_normalizer import normalize_usage_metrics
from agiwo.llm.openai_response_converter import (
    convert_messages_to_responses_input,
    convert_tools_to_responses_tools,
    split_system_instructions,
)
from agiwo.utils.logging import get_logger
from agiwo.utils.retry import retry_async

logger = get_logger(__name__)


class HTTPXError(Exception):
    """Custom exception for httpx errors."""

    pass


OPENAI_RETRYABLE = (
    httpx.HTTPStatusError,
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.NetworkError,
)


OpenAIMessage = dict[str, object]
OpenAITool = dict[str, object]
ToolCallState = dict[int, dict[str, object]]


def _get_attr(obj: object, key: str) -> object | None:
    if isinstance(obj, dict):
        return obj.get(key)
    if obj is None:
        return None
    return getattr(obj, key, None)


def _flatten_response_usage(usage: object) -> object:
    input_tokens_details = _get_attr(usage, "input_tokens_details")
    cached_tokens = _get_attr(input_tokens_details, "cached_tokens")
    if cached_tokens is None:
        return usage
    return {
        "input_tokens": _get_attr(usage, "input_tokens"),
        "output_tokens": _get_attr(usage, "output_tokens"),
        "total_tokens": _get_attr(usage, "total_tokens"),
        "input_tokens_details": {"cached_tokens": cached_tokens},
    }


def _extract_response_error_message(response: object) -> str:
    error = _get_attr(response, "error")
    if error is None:
        return "OpenAI Responses request failed"
    message = _get_attr(error, "message")
    if isinstance(message, str) and message:
        return message
    return str(error)


class OpenAIResponsesModel(Model):
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
        provider: str = "openai-response",
    ) -> None:
        config = LLMConfig(
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
            provider=provider,
            cache_hit_price=cache_hit_price,
            input_price=input_price,
            output_price=output_price,
        )
        super().__init__(config)
        self.allow_env_fallback = allow_env_fallback

    def _resolve_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.allow_env_fallback:
            settings = get_settings()
            if settings.openai_api_key:
                return settings.openai_api_key.get_secret_value()
        return None

    def _resolve_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        if self.allow_env_fallback:
            return get_settings().openai_base_url
        return None

    def _build_params(
        self,
        messages: list[OpenAIMessage],
        tools: list[OpenAITool] | None,
    ) -> dict[str, object]:
        instructions, remaining_messages = split_system_instructions(messages)
        params: dict[str, object] = {
            "model": self.id or self.name,
            "input": convert_messages_to_responses_input(remaining_messages),
            "stream": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if instructions is not None:
            params["instructions"] = instructions
        if self.max_output_tokens:
            params["max_output_tokens"] = self.max_output_tokens
        if tools:
            params["tools"] = convert_tools_to_responses_tools(tools)
        return params

    def _map_finish_reason(self, response: object) -> str:
        output_items = _get_attr(response, "output") or []
        if any(_get_attr(item, "type") == "function_call" for item in output_items):
            return "tool_calls"

        incomplete_details = _get_attr(response, "incomplete_details")
        incomplete_reason = _get_attr(incomplete_details, "reason")
        if incomplete_reason == "max_output_tokens":
            return "length"
        if isinstance(incomplete_reason, str) and incomplete_reason:
            return incomplete_reason
        return "stop"

    def _handle_output_item_added(
        self,
        event: object,
        tool_calls_state: ToolCallState,
    ) -> None:
        item = _get_attr(event, "item")
        if _get_attr(item, "type") != "function_call":
            return
        output_index = _get_attr(event, "output_index")
        if not isinstance(output_index, int):
            return
        tool_calls_state[output_index] = {
            "id": _get_attr(item, "call_id") or _get_attr(item, "id") or "",
            "name": _get_attr(item, "name") or "",
            "name_emitted": False,
        }

    def _handle_function_call_delta(
        self,
        event: object,
        tool_calls_state: ToolCallState,
    ) -> StreamChunk | None:
        output_index = _get_attr(event, "output_index")
        if not isinstance(output_index, int):
            return None
        state = tool_calls_state.get(output_index)
        if state is None:
            return None
        delta = _get_attr(event, "delta")
        if not isinstance(delta, str):
            return None
        function_payload: dict[str, object] = {"arguments": delta}
        if not state["name_emitted"]:
            function_payload["name"] = state["name"]
            state["name_emitted"] = True

        return StreamChunk(
            tool_calls=[
                {
                    "index": output_index,
                    "id": state["id"],
                    "type": "function",
                    "function": function_payload,
                }
            ]
        )

    @staticmethod
    def _build_text_chunk(delta: object) -> StreamChunk | None:
        if isinstance(delta, str):
            return StreamChunk(content=delta)
        return None

    @staticmethod
    def _build_reasoning_chunk(delta: object) -> StreamChunk | None:
        if isinstance(delta, str):
            return StreamChunk(reasoning_content=delta)
        return None

    def _build_completed_chunk(self, response: object) -> StreamChunk:
        usage = _get_attr(response, "usage")
        normalized_usage = None
        if usage is not None:
            normalized_usage = normalize_usage_metrics(_flatten_response_usage(usage))
        return StreamChunk(
            usage=normalized_usage,
            finish_reason=self._map_finish_reason(response),
        )

    def _raise_failed_response(self, event: object) -> None:
        raise RuntimeError(
            _extract_response_error_message(_get_attr(event, "response"))
        )

    def _event_to_chunk(
        self,
        event: object,
        tool_calls_state: ToolCallState,
    ) -> StreamChunk | None:
        event_type = _get_attr(event, "type")
        if event_type == "response.output_text.delta":
            return self._build_text_chunk(_get_attr(event, "delta"))
        if event_type in {
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        }:
            return self._build_reasoning_chunk(_get_attr(event, "delta"))
        if event_type == "response.output_item.added":
            self._handle_output_item_added(event, tool_calls_state)
            return None
        if event_type == "response.function_call_arguments.delta":
            return self._handle_function_call_delta(event, tool_calls_state)
        if event_type == "response.failed":
            self._raise_failed_response(event)
        if event_type == "response.completed":
            return self._build_completed_chunk(_get_attr(event, "response"))
        return None

    async def _parse_sse_line(self, line: str) -> tuple[str, str] | None:
        """Parse a single SSE line into (event_type, data)."""
        if not line.strip():
            return None
        if line.startswith("event:"):
            return ("event", line[len("event:").strip()])
        if line.startswith("data:"):
            return ("data", line[len("data:").strip()])
        return None

    async def _parse_sse_stream(self, response: httpx.Response) -> AsyncIterator[dict]:
        """Parse SSE stream and yield event objects."""
        current_event = None
        current_data = None

        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                # Empty line marks end of an event
                if current_event and current_data:
                    try:
                        yield {"type": current_event, **json.loads(current_data)}
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse SSE data", data=current_data)
                current_event = None
                current_data = None
                continue

            if line.startswith("event:"):
                current_event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                current_data = line[len("data:") :].strip()

    @retry_async(exceptions=OPENAI_RETRYABLE)
    async def _open_stream(
        self,
        *,
        url: str,
        params: dict[str, object],
        headers: dict[str, str],
    ) -> tuple[AsyncExitStack, httpx.Response]:
        # Keep both client and streaming response contexts open for the caller.
        stack = AsyncExitStack()
        try:
            limits = httpx.Limits(max_keepalive_connections=100, max_connections=100)
            timeout = httpx.Timeout(120.0, connect=30.0)
            client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=timeout, limits=limits)
            )
            response = await stack.enter_async_context(
                client.stream("POST", url, json=params, headers=headers)
            )
            if response.status_code != 200:
                error_text = await response.aread()
                raise httpx.HTTPStatusError(
                    f"Request failed with status {response.status_code}: {error_text.decode()}",
                    request=response.request,
                    response=response,
                )
            return stack, response
        except Exception:
            await stack.aclose()
            raise

    async def arun_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        actual_model = self.id or self.name
        params = self._build_params(messages, tools)
        logger.info(
            "llm_request",
            model=actual_model,
            messages_count=len(messages),
            tools_count=len(tools) if tools else 0,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            detail=params,
        )

        base_url = self._resolve_base_url()
        api_key = self._resolve_api_key()

        if not base_url:
            raise ValueError("base_url is required")
        if not api_key:
            raise ValueError("api_key is required")

        url = f"{base_url.rstrip('/')}/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        chunk_count = 0
        tool_calls_state: ToolCallState = {}
        logger.info("openai_responses_stream_started", model=actual_model)

        stack, response = await self._open_stream(
            url=url,
            params=params,
            headers=headers,
        )
        try:
            async for event in self._parse_sse_stream(response):
                stream_chunk = self._event_to_chunk(event, tool_calls_state)
                if stream_chunk is None:
                    continue
                if (
                    stream_chunk.content is None
                    and stream_chunk.reasoning_content is None
                    and stream_chunk.tool_calls is None
                    and stream_chunk.usage is None
                    and stream_chunk.finish_reason is None
                ):
                    continue
                chunk_count += 1
                yield stream_chunk
        finally:
            await stack.aclose()

        if chunk_count == 0:
            raise RuntimeError(
                "OpenAI Responses stream returned no chunks; "
                "verify the model and base_url support the Responses API"
            )

        logger.info(
            "openai_responses_stream_ended",
            model=actual_model,
            chunk_count=chunk_count,
        )


__all__ = ["OpenAIResponsesModel"]
