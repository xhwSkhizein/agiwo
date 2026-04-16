from agiwo.llm.event_normalizer import (
    AnthropicStreamEvent,
    AnthropicStreamTranslator,
    normalize_anthropic_stop_reason,
    normalize_usage_metrics,
)


def test_normalize_usage_metrics_openai_format():
    usage_data = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cached_tokens": 2,
    }

    result = normalize_usage_metrics(usage_data)

    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5
    assert result["total_tokens"] == 15
    assert result["cache_read_tokens"] == 2


def test_normalize_usage_metrics_anthropic_format():
    usage_data = {
        "input_tokens": 8,
        "output_tokens": 5,
        "cache_read_tokens": 2,
        "cache_creation_tokens": 1,
    }

    result = normalize_usage_metrics(usage_data)

    assert result["input_tokens"] == 11
    assert result["output_tokens"] == 5
    assert result["total_tokens"] == 16
    assert result["cache_read_tokens"] == 2
    assert result["cache_creation_tokens"] == 1


def test_normalize_usage_metrics_none():
    result = normalize_usage_metrics(None)

    assert result["input_tokens"] is None
    assert result["output_tokens"] is None
    assert result["total_tokens"] is None
    assert result["cache_read_tokens"] is None
    assert result["cache_creation_tokens"] is None


def test_normalize_usage_metrics_empty_dict():
    result = normalize_usage_metrics({})

    assert result["input_tokens"] is None
    assert result["output_tokens"] is None
    assert result["total_tokens"] is None


def test_normalize_usage_metrics_partial_data():
    usage_data = {
        "input_tokens": 10,
    }

    result = normalize_usage_metrics(usage_data)

    assert result["input_tokens"] == 10
    assert result["output_tokens"] is None
    assert result["total_tokens"] is None


def test_normalize_usage_metrics_openai_responses_nested_cached_tokens():
    usage_data = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 3},
    }

    result = normalize_usage_metrics(usage_data)

    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 5
    assert result["total_tokens"] == 15
    assert result["cache_read_tokens"] == 3


def test_normalize_anthropic_stop_reason_maps_shared_semantics():
    assert normalize_anthropic_stop_reason("tool_use") == "tool_calls"
    assert normalize_anthropic_stop_reason("max_tokens") == "length"
    assert normalize_anthropic_stop_reason("end_turn") == "stop"


def test_anthropic_stream_translator_accumulates_usage_and_maps_finish_reason():
    translator = AnthropicStreamTranslator(include_reasoning=True)

    start_chunk = translator.process(
        AnthropicStreamEvent(
            type="message_start",
            usage={
                "input_tokens": 8,
                "cache_read_tokens": 2,
                "cache_creation_tokens": 1,
            },
        )
    )
    delta_chunk = translator.process(
        AnthropicStreamEvent(
            type="message_delta",
            usage={"output_tokens": 5},
            stop_reason="max_tokens",
        )
    )

    assert start_chunk is not None
    assert start_chunk.usage["input_tokens"] == 11
    assert delta_chunk is not None
    assert delta_chunk.usage["output_tokens"] == 5
    assert delta_chunk.finish_reason == "length"


def test_anthropic_stream_translator_emits_tool_call_with_raw_json_arguments():
    translator = AnthropicStreamTranslator(include_reasoning=False)

    assert (
        translator.process(
            AnthropicStreamEvent(
                type="content_block_start",
                index=0,
                content_block_type="tool_use",
                tool_id="tool_123",
                tool_name="lookup",
            )
        )
        is None
    )
    assert (
        translator.process(
            AnthropicStreamEvent(
                type="content_block_delta",
                index=0,
                delta_type="thinking_delta",
                reasoning="hidden reasoning",
            )
        )
        is None
    )
    assert (
        translator.process(
            AnthropicStreamEvent(
                type="content_block_delta",
                index=0,
                delta_type="input_json_delta",
                partial_json='{"x": 1}',
            )
        )
        is None
    )

    tool_chunk = translator.process(
        AnthropicStreamEvent(
            type="content_block_stop",
            index=0,
        )
    )

    assert tool_chunk is not None
    assert tool_chunk.reasoning_content is None
    assert tool_chunk.tool_calls == [
        {
            "index": 0,
            "id": "tool_123",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": '{"x": 1}',
            },
        }
    ]
