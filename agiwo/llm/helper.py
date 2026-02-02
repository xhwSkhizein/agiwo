from typing import Any


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
    if hasattr(usage_data, "get"):
        base_input = usage_data.get("input_tokens") or usage_data.get("prompt_tokens")
    else:
        base_input = getattr(usage_data, "input_tokens", None) or getattr(usage_data, "prompt_tokens", None)
    if hasattr(usage_data, "get"):
        output_tokens = usage_data.get("output_tokens") or usage_data.get("completion_tokens")
    else:
        output_tokens = getattr(usage_data, "output_tokens", None) or getattr(usage_data, "completion_tokens", None)

    # Extract cache details
    # OpenAI: cached_tokens
    # Anthropic: cache_read_tokens, cache_creation_tokens
    if hasattr(usage_data, "get"):
        cache_read_tokens = (
            usage_data.get("cache_read_tokens")
            or usage_data.get("cache_read_input_tokens")
            or usage_data.get("cached_tokens")
        )
        cache_creation_tokens = usage_data.get("cache_creation_tokens") or usage_data.get("cache_creation_input_tokens")
    else:
        cache_read_tokens = (
            getattr(usage_data, "cache_read_tokens", None)
            or getattr(usage_data, "cache_read_input_tokens", None)
            or getattr(usage_data, "cached_tokens", None)
        )
        cache_creation_tokens = getattr(usage_data, "cache_creation_tokens", None) or getattr(usage_data, "cache_creation_input_tokens", None)

    def _has_any_key(keys: tuple[str, ...]) -> bool:
        if hasattr(usage_data, "get"):
            return any(key in usage_data for key in keys)
        else:
            return any(hasattr(usage_data, key) for key in keys)

    input_tokens = base_input
    # For Anthropic, base_input typically excludes cache tokens.
    # For OpenAI, prompt_tokens already includes them.
    anthropic_cache_keys = (
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
    )
    if _has_any_key(anthropic_cache_keys) and input_tokens is not None:
        input_tokens += (cache_read_tokens or 0) + (cache_creation_tokens or 0)

    # Calculate total_tokens (Input + Output)
    if hasattr(usage_data, "get"):
        total_tokens = usage_data.get("total_tokens")
    else:
        total_tokens = getattr(usage_data, "total_tokens", None)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
    }
