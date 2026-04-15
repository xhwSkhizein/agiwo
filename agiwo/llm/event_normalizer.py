"""Anthropic-family event normalization and streaming translation helpers."""

from dataclasses import dataclass
from typing import Any, Callable

from agiwo.llm.base import StreamChunk


@dataclass
class AnthropicStreamEvent:
    """Normalized Anthropic-style stream event used across providers."""

    type: str
    index: int | None = None
    content_block_type: str | None = None
    delta_type: str | None = None
    text: str | None = None
    reasoning: str | None = None
    partial_json: str | None = None
    tool_id: str | None = None
    tool_name: str | None = None
    stop_reason: str | None = None
    usage: Any = None


def _get_val(obj: Any, key: str) -> Any:
    """Get value from dict or object attribute safely."""
    if isinstance(obj, dict):
        return obj.get(key)
    if obj is None:
        return None
    return getattr(obj, key, None)


def _get_nested_val(obj: Any, *path: str) -> Any:
    current = obj
    for key in path:
        current = _get_val(current, key)
        if current is None:
            return None
    return current


def _coerce_optional_int(value: object | None) -> int | None:
    return value if isinstance(value, int) else None


def _resolve_usage_metric(
    usage_data: Any,
    *names: str,
) -> int | None:
    for name in names:
        if "." in name:
            value = _coerce_optional_int(_get_nested_val(usage_data, *name.split(".")))
        else:
            value = _coerce_optional_int(_get_val(usage_data, name))
        if value is not None:
            return value
    return None


_USAGE_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens"),
    "cache_read_tokens": (
        "cache_read_tokens",
        "cache_read_input_tokens",
        "cached_tokens",
        "input_tokens_details.cached_tokens",
    ),
    "cache_creation_tokens": ("cache_creation_tokens", "cache_creation_input_tokens"),
}


def _has_anthropic_cache_fields(usage_data: Any) -> bool:
    if usage_data is None:
        return False
    cache_keys = (
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
    )
    if isinstance(usage_data, dict):
        return any(key in usage_data for key in cache_keys)
    return any(hasattr(usage_data, key) for key in cache_keys)


def normalize_usage_metrics(
    usage_data: Any,
) -> dict[str, int | None]:
    """Normalize model usage metrics to unified format."""
    if not usage_data:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cache_read_tokens": None,
            "cache_creation_tokens": None,
        }

    base_input = _resolve_usage_metric(
        usage_data, *_USAGE_METRIC_ALIASES["input_tokens"]
    )
    output_tokens = _resolve_usage_metric(
        usage_data, *_USAGE_METRIC_ALIASES["output_tokens"]
    )
    cache_read_tokens = _resolve_usage_metric(
        usage_data, *_USAGE_METRIC_ALIASES["cache_read_tokens"]
    )
    cache_creation_tokens = _resolve_usage_metric(
        usage_data, *_USAGE_METRIC_ALIASES["cache_creation_tokens"]
    )

    input_tokens = base_input
    if _has_anthropic_cache_fields(usage_data) and input_tokens is not None:
        input_tokens += (cache_read_tokens or 0) + (cache_creation_tokens or 0)

    total_tokens = _resolve_usage_metric(usage_data, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
    }


def normalize_anthropic_stop_reason(stop_reason: str | None) -> str | None:
    if stop_reason is None:
        return None
    return {
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "end_turn": "stop",
        "stop_sequence": "stop",
    }.get(stop_reason, stop_reason)


class AnthropicStreamTranslator:
    """Translate Anthropic-style stream events into shared StreamChunk semantics."""

    def __init__(self, *, include_reasoning: bool):
        self.include_reasoning = include_reasoning
        self._tool_calls_buffer: dict[int, dict[str, str]] = {}
        self._usage_info: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }

    def _update_usage_info(self, usage_obj: Any) -> None:
        if not usage_obj:
            return

        for key, names in _USAGE_METRIC_ALIASES.items():
            value = _resolve_usage_metric(usage_obj, *names)
            if value is not None:
                self._usage_info[key] = value

    def _start_tool_use(self, event: AnthropicStreamEvent) -> None:
        if event.content_block_type == "tool_use" and event.index is not None:
            self._tool_calls_buffer[event.index] = {
                "id": event.tool_id or "",
                "name": event.tool_name or "",
                "input": "",
            }

    def _apply_content_delta(
        self,
        event: AnthropicStreamEvent,
        stream_chunk: StreamChunk,
    ) -> None:
        if event.delta_type == "text_delta":
            stream_chunk.content = event.text
            return
        if event.delta_type == "thinking_delta" and self.include_reasoning:
            stream_chunk.reasoning_content = event.reasoning
            return
        if (
            event.delta_type == "input_json_delta"
            and event.index is not None
            and event.index in self._tool_calls_buffer
        ):
            self._tool_calls_buffer[event.index]["input"] += event.partial_json or ""

    def _finish_tool_use(
        self,
        event: AnthropicStreamEvent,
        stream_chunk: StreamChunk,
    ) -> None:
        if event.index is None or event.index not in self._tool_calls_buffer:
            return
        tool_call = self._tool_calls_buffer.pop(event.index)
        stream_chunk.tool_calls = [
            {
                "index": event.index,
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": tool_call["input"],
                },
            }
        ]

    @staticmethod
    def _is_empty_chunk(stream_chunk: StreamChunk) -> bool:
        return (
            stream_chunk.content is None
            and stream_chunk.reasoning_content is None
            and stream_chunk.tool_calls is None
            and stream_chunk.usage is None
            and stream_chunk.finish_reason is None
        )

    def process(self, event: AnthropicStreamEvent) -> StreamChunk | None:
        stream_chunk = StreamChunk()

        if event.type == "message_start":
            self._update_usage_info(event.usage)
            if event.usage:
                stream_chunk.usage = normalize_usage_metrics(self._usage_info)
        elif event.type == "content_block_start":
            self._start_tool_use(event)
        elif event.type == "content_block_delta":
            self._apply_content_delta(event, stream_chunk)
        elif event.type == "content_block_stop":
            self._finish_tool_use(event, stream_chunk)
        elif event.type == "message_delta":
            self._update_usage_info(event.usage)
            if event.usage:
                stream_chunk.usage = normalize_usage_metrics(self._usage_info)
            stream_chunk.finish_reason = normalize_anthropic_stop_reason(
                event.stop_reason
            )
        elif event.type == "message_stop":
            stream_chunk.finish_reason = "stop"

        if self._is_empty_chunk(stream_chunk):
            return None
        return stream_chunk


def _normalize_anthropic_event(
    event_type: str,
    index: int | None,
    get_root: Callable[[str], Any],
    get_nested: Callable[[str, str], Any],
) -> AnthropicStreamEvent:
    normalized = AnthropicStreamEvent(type=event_type, index=index)

    if event_type == "message_start":
        normalized.usage = get_nested("message", "usage")
    elif event_type == "content_block_start":
        normalized.content_block_type = get_nested("content_block", "type")
        normalized.tool_id = get_nested("content_block", "id")
        normalized.tool_name = get_nested("content_block", "name")
    elif event_type == "content_block_delta":
        normalized.delta_type = get_nested("delta", "type")
        normalized.text = get_nested("delta", "text")
        normalized.reasoning = get_nested("delta", "thinking")
        normalized.partial_json = get_nested("delta", "partial_json")
    elif event_type == "message_delta":
        normalized.usage = get_root("usage")
        normalized.stop_reason = get_nested("delta", "stop_reason")

    return normalized


def normalize_anthropic_sdk_event(event: Any) -> AnthropicStreamEvent:
    return _normalize_anthropic_event(
        event_type=getattr(event, "type", ""),
        index=getattr(event, "index", None),
        get_root=lambda key: _get_val(event, key),
        get_nested=lambda parent, key: _get_val(_get_val(event, parent), key),
    )


def normalize_bedrock_anthropic_event(
    chunk: dict[str, Any],
) -> AnthropicStreamEvent:
    return _normalize_anthropic_event(
        event_type=chunk.get("type", ""),
        index=chunk.get("index"),
        get_root=lambda key: chunk.get(key),
        get_nested=lambda parent, key: _get_val(chunk.get(parent), key),
    )


__all__ = [
    "AnthropicStreamEvent",
    "AnthropicStreamTranslator",
    "normalize_anthropic_stop_reason",
    "normalize_anthropic_sdk_event",
    "normalize_bedrock_anthropic_event",
    "normalize_usage_metrics",
]
