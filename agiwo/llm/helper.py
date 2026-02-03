from typing import Any


def _get_usage_value(usage_data: Any, key: str) -> Any:
    if hasattr(usage_data, "get"):
        return usage_data.get(key)
    return getattr(usage_data, key, None)


def _has_any_key(usage_data: Any, keys: tuple[str, ...]) -> bool:
    if hasattr(usage_data, "get"):
        return any(key in usage_data for key in keys)
    return any(hasattr(usage_data, key) for key in keys)


def normalize_usage_metrics(usage_data: dict[str, Any] | None) -> dict[str, int | None]:
    """
    Normalize model usage metrics to unified format.

    Handles both OpenAI-style and unified-style metrics, including prompt caching.
    Ensures 'input_tokens' always represents the TOTAL input (including cache).

    Args:
        usage_data: Raw usage data from model

    Returns:
        dict with normalized keys:
        - input_tokens (Total input)
        - output_tokens
        - total_tokens (input_tokens + output_tokens)
        - cache_read_tokens
        - cache_creation_tokens
    """
    if not usage_data:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cache_read_tokens": None,
            "cache_creation_tokens": None,
        }

    # Extract base tokens
    # OpenAI: prompt_tokens (Total)
    # Anthropic: input_tokens (Excludes cache)
    base_input = _get_usage_value(usage_data, "input_tokens")
    if base_input is None:
        base_input = _get_usage_value(usage_data, "prompt_tokens")

    output_tokens = _get_usage_value(usage_data, "output_tokens")
    if output_tokens is None:
        output_tokens = _get_usage_value(usage_data, "completion_tokens")

    # Extract cache details
    # OpenAI: cached_tokens
    # Anthropic: cache_read_tokens, cache_creation_tokens
    cache_read_tokens = _get_usage_value(usage_data, "cache_read_tokens")
    if cache_read_tokens is None:
        cache_read_tokens = _get_usage_value(usage_data, "cache_read_input_tokens")
    if cache_read_tokens is None:
        cache_read_tokens = _get_usage_value(usage_data, "cached_tokens")

    cache_creation_tokens = _get_usage_value(usage_data, "cache_creation_tokens")
    if cache_creation_tokens is None:
        cache_creation_tokens = _get_usage_value(
            usage_data, "cache_creation_input_tokens"
        )

    input_tokens = base_input
    # For Anthropic, base_input typically excludes cache tokens.
    # For OpenAI, prompt_tokens already includes them.
    anthropic_cache_keys = (
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
    )
    if _has_any_key(usage_data, anthropic_cache_keys) and input_tokens is not None:
        input_tokens += (cache_read_tokens or 0) + (cache_creation_tokens or 0)

    # Calculate total_tokens (Input + Output)
    total_tokens = _get_usage_value(usage_data, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
    }
