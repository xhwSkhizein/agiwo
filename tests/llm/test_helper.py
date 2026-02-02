import pytest
from agiwo.llm.helper import normalize_usage_metrics


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
